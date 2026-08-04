"""Turns a Phase 1 PipelineReport into a stored trajectory.

Reads the report as data; nothing in agents/ is modified. The report already
carries everything needed -- per-node traces with tool calls and latencies,
guardrail verdicts, contracts, errors -- so this is a projection, not new
instrumentation bolted into the graph.

Reward derivation is deliberately explicit and documented rather than clever:
a training signal nobody can explain is a training signal nobody should trust.
"""

from __future__ import annotations

from typing import Any

from app.observability import get_logger, log_event
from instrumentation.schemas import (
    Resolution,
    RewardSignal,
    StepKind,
    Trajectory,
    TrajectoryStep,
    _now,
)

logger = get_logger("agentforge.instrumentation.recorder")

STAGES = ("analysis", "develop", "test", "deploy")

_RESOLUTION_FOR_STATUS = {
    "completed": Resolution.COMPLETED,
    "halted": Resolution.HALTED,
    "failed": Resolution.FAILED,
}


def build_steps(report: dict[str, Any]) -> list[TrajectoryStep]:
    """Flatten a report into ordered steps.

    Order follows the pipeline: for each stage, the guardrail verdicts that
    gated it, then the agent's reasoning, its tool calls, and its handoff.
    """
    steps: list[TrajectoryStep] = []
    traces = {t.get("stage"): t for t in report.get("traces") or []}
    guardrails = report.get("guardrail_reports") or []
    run_id = str(report.get("run_id", ""))

    def add(**kwargs) -> None:
        steps.append(TrajectoryStep(run_id=run_id, ordinal=len(steps), **kwargs))

    for stage in STAGES:
        for guard in [g for g in guardrails if g.get("stage") == stage]:
            # Only interventions are worth a step; an "allow" is the absence of news.
            if guard.get("action") == "allow":
                continue
            add(
                kind=StepKind.GUARDRAIL,
                stage=stage,
                name=f"guardrail.{guard.get('direction', 'input')}",
                input={"direction": guard.get("direction", "")},
                output={
                    "action": guard.get("action", ""),
                    "risk_score": guard.get("risk_score", 0.0),
                    "categories": [f.get("category") for f in (guard.get("findings") or [])],
                },
                latency_ms=float(guard.get("latency_ms", 0.0)),
                success=guard.get("action") != "block",
            )

        trace = traces.get(stage)
        if trace is None:
            continue

        add(
            kind=StepKind.REASONING,
            stage=stage,
            name=str(trace.get("agent", stage)),
            output={
                "summary": str(trace.get("reasoning", "")),
                "provider": trace.get("provider", ""),
                "model": trace.get("model", ""),
            },
            latency_ms=float(trace.get("latency_ms", 0.0)),
            success=bool(trace.get("success", True)),
            error=str(trace.get("error", "")),
        )

        for call in trace.get("tool_calls") or []:
            name = str(call.get("name", "tool"))
            outcome = call.get("outcome", "ok")
            add(
                # An llm.complete is a model call; anything else is a tool/action.
                kind=StepKind.LLM_CALL if name.startswith("llm.") else StepKind.TOOL_CALL,
                stage=stage,
                name=name,
                input=dict(call.get("arguments") or {}),
                output={"detail": str(call.get("detail", ""))},
                latency_ms=float(call.get("latency_ms", 0.0)),
                success=outcome == "ok",
                error="" if outcome == "ok" else str(call.get("detail", "")),
            )

        contract = report.get(stage)
        if contract:
            handoff = contract.get("handoff") or {}
            add(
                kind=StepKind.HANDOFF,
                stage=stage,
                name=f"{stage}.handoff",
                output={
                    "summary": str(handoff.get("summary", "")),
                    "to": handoff.get("to", ""),
                    "blocking": bool(handoff.get("blocking", False)),
                    "open_questions": handoff.get("open_questions") or [],
                    "confidence": contract.get("confidence", 0.0),
                },
            )

    for error in report.get("errors") or []:
        add(kind=StepKind.ERROR, name="pipeline.error", success=False, error=str(error))

    return steps


def derive_reward(report: dict[str, Any], human_intervention: bool = False) -> RewardSignal:
    """Implicit reward from what the run did.

    The score is a documented weighting, not a learned one:

        start at stages_completed / stages_total   (progress actually made)
        -0.25  the run did not reach a completed resolution
        -0.10  per guardrail intervention, capped at two
        -0.10  per error, capped at two
        -0.15  the run escalated (an agent raised a blocking question)
        -0.10  a human had to intervene

    Explicit user feedback, when it arrives later, overrides this -- a human
    saying "this was wrong" outranks any heuristic we compute.
    """
    status = str(report.get("status", ""))
    resolution = _RESOLUTION_FOR_STATUS.get(status, Resolution.UNKNOWN)
    stages_completed = sum(1 for s in STAGES if report.get(s))

    interventions = [
        g for g in (report.get("guardrail_reports") or []) if g.get("action") != "allow"
    ]
    errors = report.get("errors") or []
    retries = sum(
        1
        for t in (report.get("traces") or [])
        for c in (t.get("tool_calls") or [])
        if int((c.get("arguments") or {}).get("attempt", 1)) > 1
    )
    escalated = (
        any(bool(((report.get(s) or {}).get("handoff") or {}).get("blocking")) for s in STAGES)
        or resolution is not Resolution.COMPLETED
    )

    score = stages_completed / len(STAGES)
    if resolution is not Resolution.COMPLETED:
        score -= 0.25
    score -= 0.10 * min(len(interventions), 2)
    score -= 0.10 * min(len(errors), 2)
    if escalated:
        score -= 0.15
    if human_intervention:
        score -= 0.10

    return RewardSignal(
        resolution=resolution,
        resolved=resolution is Resolution.COMPLETED,
        escalated=escalated,
        human_intervention=human_intervention,
        guardrail_interventions=len(interventions),
        stages_completed=stages_completed,
        stages_total=len(STAGES),
        retries=retries,
        errors=len(errors),
        implicit_score=round(max(0.0, min(1.0, score)), 4),
    )


def build_trajectory(
    report: dict[str, Any],
    *,
    agent_config_id: str = "",
    agent_config_version: int = 0,
    human_intervention: bool = False,
) -> Trajectory:
    """Project a report into a trajectory. Not yet scrubbed -- the store does that."""
    return Trajectory(
        run_id=str(report.get("run_id", "")),
        session_id=str(report.get("session_id", "")),
        topic=str(report.get("topic", "")),
        agent_config_id=agent_config_id,
        agent_config_version=agent_config_version,
        status=str(report.get("status", "")),
        started_at=str(report.get("created_at") or _now()),
        finished_at=_now(),
        total_latency_ms=float(report.get("total_latency_ms", 0.0)),
        steps=build_steps(report),
        reward=derive_reward(report, human_intervention),
    )


class TrajectoryRecorder:
    """Builds and persists trajectories. Scrubbing happens inside the store."""

    def __init__(self, store) -> None:
        self.store = store

    async def record(
        self,
        report: dict[str, Any],
        *,
        agent_config_id: str = "",
        agent_config_version: int = 0,
        human_intervention: bool = False,
    ) -> Trajectory:
        trajectory = build_trajectory(
            report,
            agent_config_id=agent_config_id,
            agent_config_version=agent_config_version,
            human_intervention=human_intervention,
        )
        stored = await self.store.save_trajectory(trajectory)
        log_event(
            logger,
            "instrumentation.trajectory_recorded",
            run_id=stored.run_id,
            steps=stored.step_count,
            resolution=stored.reward.resolution.value,
            reward=stored.reward.implicit_score,
            pii_findings=stored.pii_findings,
        )
        return stored

    async def record_feedback(
        self, run_id: str, feedback: str, note: str = ""
    ) -> Trajectory | None:
        """Explicit user feedback, which outranks the implicit score."""
        updated = await self.store.set_feedback(run_id, feedback, note)
        if updated is not None:
            log_event(logger, "instrumentation.feedback", run_id=run_id, feedback=feedback)
        return updated
