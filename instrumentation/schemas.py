"""Trajectory, reward, audit, and version types -- plus the training export shape.

The export schema is defined now even though no training pipeline consumes it
yet. Deciding it late is how you discover the trajectories you have been storing
for six months are missing the one field the trainer needs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_id() -> str:
    return uuid.uuid4().hex


Score = Annotated[float, Field(ge=0.0, le=1.0)]


class StepKind(StrEnum):
    REASONING = "reasoning"      # an agent's own summary of what it decided
    LLM_CALL = "llm_call"        # a model invocation
    TOOL_CALL = "tool_call"      # retrieval, action, or other tool
    GUARDRAIL = "guardrail"      # an inline guardrail verdict
    HANDOFF = "handoff"          # one agent passing a contract to the next
    ERROR = "error"


class Resolution(StrEnum):
    COMPLETED = "completed"
    HALTED = "halted"
    FAILED = "failed"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------


class TrajectoryStep(BaseModel):
    """One observable event. `input`/`output` are scrubbed before persistence."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    run_id: str = ""
    ordinal: int = 0
    kind: StepKind
    stage: str = ""
    name: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    at: str = Field(default_factory=_now)


class RewardSignal(BaseModel):
    """Implicit and explicit signals a future trainer can optimise against.

    Kept separate from the trajectory because reward is a *label*: it can be
    revised (a user leaves feedback an hour later) without rewriting the
    observed steps, which must stay immutable.
    """

    model_config = ConfigDict(extra="forbid")

    resolution: Resolution = Resolution.UNKNOWN
    resolved: bool = False
    # True when the pipeline could not finish on its own.
    escalated: bool = False
    # True when a human had to act -- e.g. approving Text2SQL before it ran.
    human_intervention: bool = False
    guardrail_interventions: int = 0
    stages_completed: int = 0
    stages_total: int = 0
    retries: int = 0
    errors: int = 0

    user_feedback: Literal["positive", "negative", "neutral"] | None = None
    user_feedback_note: str = ""   # scrubbed like everything else
    feedback_at: str = ""

    implicit_score: Score = 0.0


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    session_id: str = ""
    topic: str = ""                      # scrubbed
    agent_config_id: str = ""
    agent_config_version: int = 0
    status: str = ""
    started_at: str = Field(default_factory=_now)
    finished_at: str = ""
    total_latency_ms: float = 0.0

    steps: list[TrajectoryStep] = Field(default_factory=list)
    reward: RewardSignal = Field(default_factory=RewardSignal)

    # Set by the store when it scrubs. A row with scrubbed=False never persists.
    scrubbed: bool = False
    pii_findings: int = 0
    pii_categories: list[str] = Field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def steps_of(self, kind: StepKind) -> list[TrajectoryStep]:
        return [s for s in self.steps if s.kind is kind]


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


class FieldChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    before: Any = None
    after: Any = None
    change: Literal["added", "removed", "changed"]


class AuditEntry(BaseModel):
    """Who did what, when, and what changed."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    actor: str = "unknown"
    action: str                      # agent_config.update, action.rollback, ...
    entity_type: str = ""
    entity_id: str = ""
    at: str = Field(default_factory=_now)
    summary: str = ""
    changes: list[FieldChange] = Field(default_factory=list)
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    scrubbed: bool = False
    pii_findings: int = 0


# --------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------


class EntityType(StrEnum):
    AGENT_CONFIG = "agent_config"
    ACTION = "action"
    KNOWLEDGE_BASE = "knowledge_base"


class VersionRecord(BaseModel):
    """One immutable snapshot. Rollback appends a new version, never rewrites."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    entity_type: EntityType
    entity_id: str
    version: int
    payload: dict[str, Any] = Field(default_factory=dict)
    actor: str = "unknown"
    note: str = ""
    # Set when this version was produced by rolling back to an earlier one.
    rolled_back_from: int | None = None
    created_at: str = Field(default_factory=_now)


# --------------------------------------------------------------------------
# Training export
# --------------------------------------------------------------------------

EXPORT_SCHEMA_VERSION = "1.0"


class TrainingTurn(BaseModel):
    """One trajectory step, flattened into a trainer-friendly turn."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "assistant", "tool", "guardrail"]
    stage: str = ""
    name: str = ""
    content: str = ""
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = True


class TrainingExample(BaseModel):
    """The unit a future fine-tuning or RL pipeline consumes.

    One row per completed run. `reward` is the scalar to optimise; the component
    signals are kept alongside so a trainer can re-weight them without a
    re-export.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = EXPORT_SCHEMA_VERSION
    run_id: str
    created_at: str = Field(default_factory=_now)

    instruction: str = ""                       # the user's topic, scrubbed
    turns: list[TrainingTurn] = Field(default_factory=list)

    outcome: Resolution = Resolution.UNKNOWN
    reward: Score = 0.0
    reward_components: RewardSignal = Field(default_factory=RewardSignal)

    metadata: dict[str, Any] = Field(default_factory=dict)
    # Redundant with the store, but an exported file must carry its own proof
    # that it was scrubbed -- the guarantee has to travel with the data.
    pii_scrubbed: bool = True


_ROLE_FOR_KIND: dict[StepKind, str] = {
    StepKind.REASONING: "assistant",
    StepKind.LLM_CALL: "assistant",
    StepKind.HANDOFF: "assistant",
    StepKind.TOOL_CALL: "tool",
    StepKind.GUARDRAIL: "guardrail",
    StepKind.ERROR: "tool",
}


def to_training_example(trajectory: Trajectory) -> TrainingExample:
    """Flatten a stored trajectory into the export shape.

    Refuses unscrubbed input: an export is exactly where unscrubbed data would
    escape, so the check lives here as well as at the storage boundary.
    """
    if not trajectory.scrubbed:
        raise ValueError(
            f"refusing to export unscrubbed trajectory {trajectory.run_id}; "
            "trajectories must be scrubbed before they leave storage"
        )

    turns = [
        TrainingTurn(
            role=_ROLE_FOR_KIND.get(step.kind, "tool"),  # type: ignore[arg-type]
            stage=step.stage,
            name=step.name,
            content=str(step.output.get("summary") or step.output.get("text") or "")[:8000],
            tool_input=step.input,
            tool_output=step.output,
            latency_ms=step.latency_ms,
            success=step.success,
        )
        for step in trajectory.steps
    ]

    return TrainingExample(
        run_id=trajectory.run_id,
        created_at=trajectory.started_at,
        instruction=trajectory.topic,
        turns=turns,
        outcome=trajectory.reward.resolution,
        reward=trajectory.reward.implicit_score,
        reward_components=trajectory.reward,
        metadata={
            "agent_config_id": trajectory.agent_config_id,
            "agent_config_version": trajectory.agent_config_version,
            "status": trajectory.status,
            "steps": trajectory.step_count,
            "total_latency_ms": trajectory.total_latency_ms,
            "pii_categories": trajectory.pii_categories,
        },
    )
