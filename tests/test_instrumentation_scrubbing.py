"""PII scrubbing before persistence, trajectory capture, and reward derivation.

The scrubbing tests are the compliance-critical ones: they assert not just that
the API scrubs, but that a caller who tries to persist raw PII *cannot*.
"""

from __future__ import annotations

import json

import pytest

from instrumentation.recorder import TrajectoryRecorder, derive_reward
from instrumentation.schemas import (
    Resolution,
    StepKind,
    Trajectory,
    TrajectoryStep,
    to_training_example,
)
from instrumentation.scrubbing import assert_clean, scrub, scrub_text
from instrumentation.store import InMemoryInstrumentationStore

EMAIL = "alex.doe@example.com"
CARD = "4111111111111111"
SSN = "123-45-6789"
PHONE = "555-234-5678"

# A report whose *tool output* carries PII -- the case the brief calls out.
REPORT_WITH_PII = {
    "run_id": "run-pii",
    "session_id": "sess-1",
    "topic": f"Look up the account for {EMAIL}",
    "status": "completed",
    "created_at": "2026-08-04T19:18:00Z",
    "total_latency_ms": 900.0,
    "analysis": {
        "agent": "Ana (Analyst)",
        "confidence": 0.8,
        "handoff": {"summary": f"Customer {EMAIL} reported an issue.", "to": "develop"},
    },
    "traces": [
        {
            "stage": "analysis",
            "agent": "Ana (Analyst)",
            "reasoning": f"Reached the customer on {PHONE}.",
            "provider": "anthropic",
            "latency_ms": 400.0,
            "success": True,
            "tool_calls": [
                {
                    "name": "search_knowledge_base",
                    "arguments": {"query": f"account {EMAIL}"},
                    "outcome": "ok",
                    "latency_ms": 120.0,
                    "detail": f"row: card {CARD}, ssn {SSN}",
                }
            ],
        }
    ],
    "guardrail_reports": [{"stage": "analysis", "direction": "input", "action": "allow"}],
    "errors": [],
}


@pytest.fixture
def store() -> InMemoryInstrumentationStore:
    return InMemoryInstrumentationStore()


# --- scrubbing primitives --------------------------------------------------


def test_scrub_text_redacts_each_pii_category():
    text = f"{EMAIL} {PHONE} {SSN} {CARD}"
    cleaned, categories, count = scrub_text(text)

    assert set(categories) == {"email", "phone", "ssn", "credit_card"}
    assert count == 4
    for secret in (EMAIL, PHONE, SSN, CARD):
        assert secret not in cleaned
    assert "[REDACTED:EMAIL]" in cleaned


def test_scrub_walks_nested_structures_including_dict_keys():
    payload = {EMAIL: "keyed by email", "rows": [{"note": f"ssn {SSN}"}, ["deep", CARD]]}

    report = scrub(payload)

    blob = json.dumps(report.value)
    assert EMAIL not in blob and SSN not in blob and CARD not in blob
    assert report.findings == 3
    assert report.sorted_categories == ["credit_card", "email", "ssn"]


def test_scrub_leaves_non_pii_and_numerics_alone():
    payload = {"latency_ms": 42.5, "ok": True, "count": None, "order": "1234567812345678"}

    report = scrub(payload)

    # A 16-digit order number is not Luhn-valid, so it is not a card.
    assert report.clean
    assert report.value == payload


def test_assert_clean_raises_when_pii_survives():
    with pytest.raises(AssertionError, match="still contains PII"):
        assert_clean({"note": f"call {PHONE}"}, "payload")


# --- the hard requirement --------------------------------------------------


async def test_pii_in_tool_output_is_scrubbed_before_storage(store):
    """Acceptance: injected PII in a tool call never reaches the store."""
    recorder = TrajectoryRecorder(store)

    stored = await recorder.record(REPORT_WITH_PII)

    blob = json.dumps(stored.model_dump(mode="json"))
    for secret in (EMAIL, PHONE, SSN, CARD):
        assert secret not in blob, f"{secret} survived into the stored trajectory"

    assert stored.scrubbed is True
    assert stored.pii_findings >= 5
    assert set(stored.pii_categories) >= {"email", "phone", "ssn", "credit_card"}

    # And the same holds for what is read back out.
    reloaded = await store.get_trajectory("run-pii")
    assert_clean(reloaded.model_dump(mode="json"), "reloaded trajectory")


async def test_the_store_scrubs_even_when_the_caller_does_not(store):
    """A caller cannot persist raw PII by skipping the recorder.

    Scrubbing lives in the save path, so 'forgot to scrub' is not reachable.
    """
    raw = Trajectory(
        run_id="raw-1",
        topic=f"contact {EMAIL}",
        steps=[
            TrajectoryStep(
                kind=StepKind.TOOL_CALL, name="lookup", output={"detail": f"ssn {SSN}"}
            )
        ],
    )
    assert raw.scrubbed is False

    stored = await store.save_trajectory(raw)
    blob = json.dumps(stored.model_dump(mode="json"))

    assert stored.scrubbed is True
    assert EMAIL not in blob
    assert SSN not in blob


async def test_feedback_notes_are_scrubbed_too(store):
    recorder = TrajectoryRecorder(store)
    await recorder.record(REPORT_WITH_PII)

    updated = await recorder.record_feedback(
        "run-pii", "negative", f"wrong account, mine is {EMAIL}"
    )

    assert updated.reward.user_feedback == "negative"
    assert EMAIL not in updated.reward.user_feedback_note
    assert "[REDACTED:EMAIL]" in updated.reward.user_feedback_note


def test_export_refuses_an_unscrubbed_trajectory():
    """The export boundary re-checks; it does not trust the flag blindly."""
    with pytest.raises(ValueError, match="refusing to export unscrubbed"):
        to_training_example(Trajectory(run_id="x", scrubbed=False))


# --- trajectory capture ----------------------------------------------------


async def test_a_run_produces_steps_tool_calls_and_timings(store):
    """Acceptance: a full run yields a trajectory with steps, calls and timings."""
    stored = await TrajectoryRecorder(store).record(REPORT_WITH_PII)

    kinds = [s.kind for s in stored.steps]
    assert StepKind.REASONING in kinds
    assert StepKind.TOOL_CALL in kinds
    assert StepKind.HANDOFF in kinds

    tool = stored.steps_of(StepKind.TOOL_CALL)[0]
    assert tool.name == "search_knowledge_base"
    assert tool.latency_ms == 120.0
    assert tool.success is True

    reasoning = stored.steps_of(StepKind.REASONING)[0]
    assert reasoning.latency_ms == 400.0
    assert reasoning.output["provider"] == "anthropic"

    assert [s.ordinal for s in stored.steps] == list(range(len(stored.steps)))


async def test_guardrail_interventions_become_steps_but_allows_do_not(store):
    report = {
        **REPORT_WITH_PII,
        "run_id": "run-guard",
        "guardrail_reports": [
            {"stage": "analysis", "direction": "input", "action": "allow"},
            {
                "stage": "analysis",
                "direction": "input",
                "action": "block",
                "risk_score": 0.9,
                "findings": [{"category": "instruction_override"}],
            },
        ],
    }

    stored = await TrajectoryRecorder(store).record(report)
    guards = stored.steps_of(StepKind.GUARDRAIL)

    assert len(guards) == 1, "an 'allow' is the absence of news, not a step"
    assert guards[0].output["action"] == "block"
    assert guards[0].success is False


# --- reward ----------------------------------------------------------------


def test_a_clean_completed_run_scores_well():
    # Phase 1 emits a full contract or None per stage -- never an empty dict.
    contract = {"agent": "x", "confidence": 0.8, "handoff": {"summary": "s"}}
    reward = derive_reward(
        {**REPORT_WITH_PII, "develop": contract, "test": contract, "deploy": contract}
    )

    assert reward.resolution is Resolution.COMPLETED
    assert reward.resolved is True
    assert reward.escalated is False
    assert reward.implicit_score == pytest.approx(1.0)


def test_a_halted_run_with_interventions_scores_lower():
    reward = derive_reward(
        {
            "status": "halted",
            "analysis": {"handoff": {"blocking": True}},
            "guardrail_reports": [{"action": "block"}],
            "errors": ["analysis: input blocked by guardrails"],
        }
    )

    assert reward.resolution is Resolution.HALTED
    assert reward.resolved is False
    assert reward.escalated is True
    assert reward.guardrail_interventions == 1
    assert reward.errors == 1
    assert reward.implicit_score < 0.3


def test_human_intervention_is_recorded_and_penalised():
    base = derive_reward(REPORT_WITH_PII)
    with_human = derive_reward(REPORT_WITH_PII, human_intervention=True)

    assert with_human.human_intervention is True
    assert with_human.implicit_score < base.implicit_score


def test_retries_are_counted_from_tool_call_attempts():
    report = {
        "status": "completed",
        "traces": [
            {
                "stage": "analysis",
                "tool_calls": [
                    {"name": "llm.complete", "arguments": {"attempt": 1}},
                    {"name": "llm.complete", "arguments": {"attempt": 2}},
                ],
            }
        ],
    }

    assert derive_reward(report).retries == 1


# --- training export -------------------------------------------------------


async def test_training_export_carries_turns_reward_and_its_scrubbed_flag(store):
    stored = await TrajectoryRecorder(store).record(REPORT_WITH_PII)

    example = to_training_example(stored)

    assert example.schema_version == "1.0"
    assert example.run_id == "run-pii"
    assert example.outcome is Resolution.COMPLETED
    assert example.reward == stored.reward.implicit_score
    assert example.pii_scrubbed is True
    assert {t.role for t in example.turns} >= {"assistant", "tool"}
    # The instruction is the scrubbed topic, not the original.
    assert EMAIL not in example.instruction
    assert_clean(example.model_dump(mode="json"), "training example")
