"""Pure evaluators: data in, `Alert | None` (or a list, for circuit_breaker_trips) out.

Each rule reads only data Phases 4 and 6 already produce -- no new signal
source, no touched schema. Thresholds and severity come from `RuleConfig`
(monitoring/config.py), never a literal in this file.
"""

from __future__ import annotations

from collections import Counter

from instrumentation.schemas import Resolution, StepKind, Trajectory
from monitoring.config import RuleConfig
from monitoring.schemas import Alert, Severity
from redteam.thresholds import ThresholdResult


def guardrail_intervention_rate(
    trajectories: list[Trajectory], cfg: RuleConfig
) -> Alert | None:
    """Share of trajectories where a guardrail tier intervened at least once."""
    if len(trajectories) < cfg.min_samples:
        return None
    interventions = sum(1 for t in trajectories if t.reward.guardrail_interventions > 0)
    rate = interventions / len(trajectories)
    if rate <= cfg.threshold:
        return None
    return Alert(
        rule=cfg.name,
        severity=cfg.severity,
        summary=f"Guardrail intervention rate {rate:.0%} exceeds {cfg.threshold:.0%}",
        detail=f"{interventions}/{len(trajectories)} runs in window had a guardrail intervene",
        value=rate,
        threshold=cfg.threshold,
    )


def escalation_rate(trajectories: list[Trajectory], cfg: RuleConfig) -> Alert | None:
    """Share of trajectories that failed, halted, or needed human intervention."""
    if len(trajectories) < cfg.min_samples:
        return None
    escalated = sum(
        1
        for t in trajectories
        if t.reward.escalated or t.reward.resolution in (Resolution.FAILED, Resolution.HALTED)
    )
    rate = escalated / len(trajectories)
    if rate <= cfg.threshold:
        return None
    return Alert(
        rule=cfg.name,
        severity=cfg.severity,
        summary=f"Escalation/failure rate {rate:.0%} exceeds {cfg.threshold:.0%}",
        detail=f"{escalated}/{len(trajectories)} runs in window escalated, halted, or failed",
        value=rate,
        threshold=cfg.threshold,
    )


def circuit_breaker_trips(trajectories: list[Trajectory], cfg: RuleConfig) -> list[Alert]:
    """Proxy for a real circuit breaker: N+ consecutive `tool_call` failures for
    the same action name, in `at` order across the window, counts as a trip.

    ponytail: actions/registry.py has no breaker of its own to read a trip
    count from -- this infers trips from Phase 6 trajectory data instead.
    Ceiling: a genuinely concurrent action can interleave failures across
    trajectories in a way "N in a row by `at` timestamp" doesn't capture
    perfectly. Replace with a real signal if/when actions/ grows a breaker.
    """
    steps = sorted(
        (step for t in trajectories for step in t.steps_of(StepKind.TOOL_CALL)),
        key=lambda s: s.at,
    )
    consecutive: Counter[str] = Counter()
    tripped: dict[str, int] = {}
    for step in steps:
        if step.success:
            consecutive[step.name] = 0
            continue
        consecutive[step.name] += 1
        if consecutive[step.name] >= cfg.breaker_failures:
            tripped[step.name] = max(tripped.get(step.name, 0), consecutive[step.name])

    return [
        Alert(
            rule=cfg.name,
            severity=cfg.severity,
            summary=f"Action '{name}' failed {count} times in a row",
            detail=(
                "circuit-breaker proxy: consecutive tool_call failures in "
                "trajectory data -- no real breaker exists in actions/ yet"
            ),
            value=float(count),
            threshold=float(cfg.breaker_failures),
            fingerprint=f"{cfg.name}:{name}",
        )
        for name, count in tripped.items()
    ]


def redteam_block_rate(result: ThresholdResult, severity: Severity) -> Alert | None:
    """Wraps Phase 4's check_thresholds -- any category failure fires once."""
    if result.passed:
        return None
    categories = ", ".join(v.category.value for v in result.failures)
    worst = min(result.failures, key=lambda v: v.block_rate)
    return Alert(
        rule="redteam_block_rate",
        severity=severity,
        summary=f"Red-team block rate fell below threshold: {categories}",
        detail=result.report(),
        value=worst.block_rate,
        threshold=worst.threshold,
    )
