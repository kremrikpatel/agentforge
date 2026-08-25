"""One test per custom rule that triggers it, plus the boundary cases that
keep a rule quiet when it should be (too few samples, rate at the threshold)."""

from __future__ import annotations

from instrumentation.schemas import (
    Resolution,
    RewardSignal,
    StepKind,
    Trajectory,
    TrajectoryStep,
)
from monitoring import rules
from monitoring.config import RuleConfig
from monitoring.schemas import Severity
from redteam.thresholds import check_thresholds

from tests.test_redteam_thresholds import summary_with


def _trajectory(run_id: str, **reward_kwargs) -> Trajectory:
    return Trajectory(run_id=run_id, reward=RewardSignal(**reward_kwargs))


def _tool_call(run_id: str, ordinal: int, name: str, *, success: bool, at: str) -> TrajectoryStep:
    return TrajectoryStep(
        run_id=run_id, ordinal=ordinal, kind=StepKind.TOOL_CALL, name=name,
        success=success, at=at,
    )


# --- guardrail_intervention_rate --------------------------------------------


def test_guardrail_intervention_rate_fires_above_threshold():
    cfg = RuleConfig("guardrail_intervention_rate", Severity.WARNING, threshold=0.3, min_samples=5)
    trajectories = [
        _trajectory(f"r{i}", guardrail_interventions=1 if i < 3 else 0) for i in range(5)
    ]

    alert = rules.guardrail_intervention_rate(trajectories, cfg)

    assert alert is not None
    assert alert.rule == "guardrail_intervention_rate"
    assert alert.severity is Severity.WARNING
    assert alert.value == 3 / 5


def test_guardrail_intervention_rate_quiet_below_threshold():
    cfg = RuleConfig("guardrail_intervention_rate", Severity.WARNING, threshold=0.5, min_samples=5)
    trajectories = [
        _trajectory(f"r{i}", guardrail_interventions=1 if i < 2 else 0) for i in range(5)
    ]

    assert rules.guardrail_intervention_rate(trajectories, cfg) is None


def test_guardrail_intervention_rate_quiet_below_min_samples():
    cfg = RuleConfig("guardrail_intervention_rate", Severity.WARNING, threshold=0.0, min_samples=5)
    trajectories = [_trajectory("r0", guardrail_interventions=1)]

    assert rules.guardrail_intervention_rate(trajectories, cfg) is None


# --- escalation_rate ---------------------------------------------------------


def test_escalation_rate_fires_above_threshold():
    cfg = RuleConfig("escalation_rate", Severity.WARNING, threshold=0.25, min_samples=4)
    trajectories = [
        _trajectory("r0", resolution=Resolution.FAILED),
        _trajectory("r1", resolution=Resolution.HALTED),
        _trajectory("r2", escalated=True, resolution=Resolution.COMPLETED),
        _trajectory("r3", resolution=Resolution.COMPLETED),
    ]

    alert = rules.escalation_rate(trajectories, cfg)

    assert alert is not None
    assert alert.value == 3 / 4


def test_escalation_rate_quiet_when_all_completed():
    cfg = RuleConfig("escalation_rate", Severity.WARNING, threshold=0.1, min_samples=4)
    trajectories = [_trajectory(f"r{i}", resolution=Resolution.COMPLETED) for i in range(4)]

    assert rules.escalation_rate(trajectories, cfg) is None


# --- circuit_breaker_trips (proxy) ------------------------------------------


def test_circuit_breaker_trips_on_consecutive_failures_for_same_action():
    cfg = RuleConfig("circuit_breaker_trips", Severity.CRITICAL, threshold=1.0, breaker_failures=3)
    trajectories = [
        _trajectory("r0"),
    ]
    trajectories[0].steps = [
        _tool_call("r0", i, "create_ticket", success=False, at=f"2026-01-01T00:00:0{i}Z")
        for i in range(3)
    ]

    alerts = rules.circuit_breaker_trips(trajectories, cfg)

    assert len(alerts) == 1
    assert alerts[0].fingerprint == "circuit_breaker_trips:create_ticket"
    assert alerts[0].value == 3.0


def test_circuit_breaker_resets_on_a_success_in_between():
    cfg = RuleConfig("circuit_breaker_trips", Severity.CRITICAL, threshold=1.0, breaker_failures=3)
    trajectories = [_trajectory("r0")]
    trajectories[0].steps = [
        _tool_call("r0", 0, "create_ticket", success=False, at="2026-01-01T00:00:00Z"),
        _tool_call("r0", 1, "create_ticket", success=False, at="2026-01-01T00:00:01Z"),
        _tool_call("r0", 2, "create_ticket", success=True, at="2026-01-01T00:00:02Z"),
        _tool_call("r0", 3, "create_ticket", success=False, at="2026-01-01T00:00:03Z"),
    ]

    assert rules.circuit_breaker_trips(trajectories, cfg) == []


# --- redteam_block_rate ------------------------------------------------------


def test_redteam_block_rate_fires_on_threshold_failure():
    summary = summary_with(jailbreak=1.0, xpia=0.4)
    result = check_thresholds(summary, minimum=0.9)

    alert = rules.redteam_block_rate(result, Severity.CRITICAL)

    assert alert is not None
    assert alert.rule == "redteam_block_rate"
    assert "xpia" in alert.detail


def test_redteam_block_rate_quiet_when_thresholds_pass():
    summary = summary_with(jailbreak=1.0, xpia=0.95)
    result = check_thresholds(summary, minimum=0.9)

    assert rules.redteam_block_rate(result, Severity.CRITICAL) is None
