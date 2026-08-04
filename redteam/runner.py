"""Runs the four attack categories and reports whether the guardrails held.

An on-demand / scheduled job, never inline with production traffic:

    python -m redteam.runner                 # run, persist, print, set exit code
    python -m redteam.runner --dry-run       # list what would run, send nothing
    python -m redteam.runner --category xpia

PyRIT drives the attacks. Each category maps to the executor built for it:
single-turn sending for jailbreak and XPIA, the purpose-built skeleton-key
executor, and the crescendo executor (which needs an adversarial model -- ours
is the existing gateway, not a second client).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from pyrit.executor.attack import (
    AttackAdversarialConfig,
    CrescendoAttack,
    PromptSendingAttack,
    SkeletonKeyAttack,
)
from pyrit.models import Message, MessagePiece

from app.config import get_settings
from app.observability import configure_logging, get_logger, log_event
from gateway.client import LLMGateway
from guardrails.engine import GuardrailEngine
from redteam.config import RedTeamSettings, get_redteam_settings
from redteam.corpus import Attack, attacks_for
from redteam.pyrit_setup import ensure_pyrit_memory
from redteam.safety import UnsafeTargetError, assert_safe_target
from redteam.schemas import (
    AttackAttempt,
    AttackCategory,
    Outcome,
    RunSummary,
    TrendReport,
    _now,
)
from redteam.scoring import AttackScorer, summarize
from redteam.store import RunStore, build_store, compute_trend
from redteam.targets import GatewayChatTarget, PipelineTarget
from redteam.thresholds import EXIT_ERROR, EXIT_UNSAFE, check_thresholds

logger = get_logger("agentforge.redteam.runner")

SINGLE_TURN = {AttackCategory.JAILBREAK, AttackCategory.XPIA}

CRESCENDO_NEEDS_PROVIDER = (
    "crescendo needs a live LLM provider for the adversarial model that composes "
    "each escalating turn; only the offline stub is configured"
)


def has_live_provider(gateway: LLMGateway) -> bool:
    """A configured provider that is not the offline stub."""
    return any(p.available() and p.name != "stub" for p in gateway.providers)


def _seed(prompt: str) -> Message:
    return Message(message_pieces=[MessagePiece(role="user", original_value=prompt)])


class RedTeamRunner:
    def __init__(
        self,
        settings: RedTeamSettings | None = None,
        gateway: LLMGateway | None = None,
        guardrails: GuardrailEngine | None = None,
        store: RunStore | None = None,
        target: PipelineTarget | None = None,
    ) -> None:
        ensure_pyrit_memory()
        self.settings = settings or get_redteam_settings()
        self.gateway = gateway or LLMGateway(get_settings())
        self.guardrails = guardrails or GuardrailEngine(get_settings(), gateway=self.gateway)
        self.scorer = AttackScorer(self.guardrails, self.gateway, self.settings)
        self.store = store
        # Constructing this validates the target. Nothing runs if it is unsafe.
        self.target = target or PipelineTarget(settings=self.settings)

    def _strategy_for(self, category: AttackCategory, attack: Attack):
        """Return a zero-arg coroutine factory that runs `attack` via PyRIT."""
        if category in SINGLE_TURN:
            strategy = PromptSendingAttack(objective_target=self.target)

            async def execute() -> None:
                await strategy.execute_async(
                    objective=attack.objective, next_message=_seed(attack.prompt)
                )

        elif category is AttackCategory.SKELETON_KEY:
            strategy = SkeletonKeyAttack(
                objective_target=self.target, skeleton_key_prompt=attack.prompt
            )

            async def execute() -> None:
                await strategy.execute_async(objective=attack.objective)

        else:  # crescendo -- multi-turn, needs an adversary to compose the turns
            strategy = CrescendoAttack(
                objective_target=self.target,
                attack_adversarial_config=AttackAdversarialConfig(
                    target=GatewayChatTarget(gateway=self.gateway)
                ),
                max_turns=self.settings.crescendo_max_turns,
                max_backtracks=self.settings.crescendo_max_backtracks,
            )

            async def execute() -> None:
                await strategy.execute_async(objective=attack.objective)

        return execute

    async def _attempt(self, attack: Attack, execute) -> AttackAttempt:
        """Run one attack through PyRIT, then score whatever the target said."""
        self.target.begin()
        failure = ""
        try:
            await execute()
        except Exception as exc:  # noqa: BLE001 -- one bad attack must not end the run
            failure = str(exc)[:300]
            log_event(logger, "redteam.attack_error", attack=attack.id, error=failure)

        collected = self.target.collected()
        attempt = await self.scorer.score(attack, collected)
        # A strategy that blew up with nothing to show cannot be scored either way.
        if failure and not collected.text:
            attempt.outcome = Outcome.ERROR
            attempt.error = failure
        return attempt

    async def run_category(self, category: AttackCategory) -> list[AttackAttempt]:
        attacks = attacks_for(category, self.settings.attempts_per_category)
        attempts: list[AttackAttempt] = []

        # Escalating against a deterministic stub proves nothing, and PyRIT would
        # burn ten retries on the adversary's malformed JSON before giving up.
        # Record the gap honestly instead: an unrunnable category is not a pass.
        if category is AttackCategory.CRESCENDO and not has_live_provider(self.gateway):
            log_event(logger, "redteam.category_skipped", category=category.value)
            return [
                AttackAttempt(
                    category=category,
                    objective=a.objective,
                    prompt=a.prompt,
                    outcome=Outcome.ERROR,
                    error=CRESCENDO_NEEDS_PROVIDER,
                )
                for a in attacks
            ]

        for attack in attacks:
            attempt = await self._attempt(attack, self._strategy_for(category, attack))
            attempts.append(attempt)
            log_event(
                logger,
                "redteam.attempt",
                attack=attack.id,
                category=category.value,
                outcome=attempt.outcome.value,
                turns=attempt.turns,
            )
        return attempts

    async def run(
        self, categories: list[AttackCategory] | None = None
    ) -> tuple[RunSummary, TrendReport]:
        categories = categories or list(AttackCategory)
        if self.store is None:
            self.store = await build_store(get_settings(), self.settings.persist)

        summary = RunSummary(
            target=self.settings.run_endpoint, threshold=self.settings.block_rate_min
        )
        attempts: list[AttackAttempt] = []
        for category in categories:
            attempts.extend(await self.run_category(category))

        for attempt in attempts:
            attempt.run_id = summary.run_id
        summary.attempts = attempts
        summary.categories = summarize(attempts)
        summary.finished_at = _now()

        # Read the previous run before writing this one, or we would compare
        # against ourselves.
        previous_id, previous_rates = await self.store.previous_run(
            summary.target, summary.run_id
        )
        await self.store.save(summary)
        return summary, compute_trend(summary, previous_rates, previous_id)

    async def aclose(self) -> None:
        await self.target.aclose()
        await self.gateway.aclose()


def format_summary(summary: RunSummary, trend: TrendReport) -> str:
    lines = [
        f"Red-team run {summary.run_id}",
        f"  target      {summary.target}",
        f"  attempts    {summary.total}  (leaked {summary.leaked})",
        f"  block rate  {summary.block_rate:.1%}",
        "",
    ]
    by_category = {t.category: t for t in trend.categories}
    for category in summary.categories:
        t = by_category.get(category.category)
        if t is not None and t.delta is not None:
            delta = f"  ({t.delta:+.1%} vs previous, {t.direction})"
        else:
            delta = "  (no previous run)"
        lines.append(
            f"  {category.category.value:<13} block {category.block_rate:6.1%}  "
            f"blocked={category.blocked} refused={category.refused} "
            f"leaked={category.leaked} err={category.errors}{delta}"
        )
    if summary.failures:
        lines += ["", "  Failing transcripts:"]
        for attempt in summary.failures[:10]:
            lines.append(f"    [{attempt.category.value}] {attempt.objective[:80]}")
            lines.append(f"      evidence: {attempt.judge_evidence[:120]}")
    return "\n".join(lines)


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redteam", description="AgentForge red-team suite")
    parser.add_argument(
        "--category",
        action="append",
        choices=[c.value for c in AttackCategory],
        help="Restrict to a category (repeatable). Default: all four.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would run; send nothing."
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override the minimum block rate."
    )
    args = parser.parse_args(argv)

    settings = get_redteam_settings()
    configure_logging(get_settings().log_level)
    categories = [AttackCategory(c) for c in (args.category or [])] or list(AttackCategory)

    if args.dry_run:
        print(f"Target: {settings.target_url or '<unset>'}")
        try:
            print(f"Safety: {assert_safe_target(settings.target_url, settings.allowed_hosts)}")
        except UnsafeTargetError as exc:
            print(f"Safety: REFUSED -- {exc}")
            return EXIT_UNSAFE
        for category in categories:
            names = [a.id for a in attacks_for(category, settings.attempts_per_category)]
            print(f"  {category.value:<13} {len(names)} attacks: {', '.join(names)}")
        return 0

    try:
        runner = RedTeamRunner(settings)
    except UnsafeTargetError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_UNSAFE

    try:
        summary, trend = await runner.run(categories)
    except Exception as exc:  # noqa: BLE001
        print(f"red-team run failed: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        await runner.aclose()

    print(format_summary(summary, trend))

    threshold = args.threshold if args.threshold is not None else settings.block_rate_min
    result = check_thresholds(summary, threshold)
    print()
    print(result.report())
    return result.exit_code


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
