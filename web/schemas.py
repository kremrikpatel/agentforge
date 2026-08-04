"""Types for agent configuration and run history.

Stage names are imported from agents.contracts rather than re-declared, so the
UI cannot drift out of sync with the pipeline it configures. Read-only import;
nothing in agents/ is modified.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.contracts import STAGE_ORDER, Stage

STAGE_NAMES: tuple[str, ...] = tuple(s.value for s in STAGE_ORDER)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_id() -> str:
    return uuid.uuid4().hex


# --------------------------------------------------------------------------
# Agent configuration
# --------------------------------------------------------------------------


class StagePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Stage
    instructions: str = Field(default="", max_length=20000)
    notes: str = Field(default="", max_length=2000)
    enabled: bool = True


class KnowledgeRef(BaseModel):
    """A knowledge base this agent may retrieve from."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=100)
    label: str = ""
    documents: int = 0
    chunks: int = 0
    last_ingested_at: str = ""


class ActionRef(BaseModel):
    """A third-party action.

    Stored as configuration only. There is no actions runtime yet -- that is a
    later phase -- so nothing here is dispatched.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    endpoint: str = ""
    method: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    enabled: bool = False


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    stages: list[StagePrompt] = Field(default_factory=list)
    knowledge: list[KnowledgeRef] = Field(default_factory=list)
    actions: list[ActionRef] = Field(default_factory=list)
    version: int = 1
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    @classmethod
    def blank(cls, name: str = "Default agent") -> AgentConfig:
        """A config pre-populated with one entry per pipeline stage."""
        return cls(name=name, stages=[StagePrompt(stage=s) for s in STAGE_ORDER])

    def stage(self, name: str) -> StagePrompt | None:
        return next((s for s in self.stages if s.stage.value == name), None)


class AgentConfigInput(BaseModel):
    """What the UI submits. The server owns id/version/timestamps."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    stages: list[StagePrompt] = Field(default_factory=list)
    knowledge: list[KnowledgeRef] = Field(default_factory=list)
    actions: list[ActionRef] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=8000)
    agent_config_id: str = ""
    run_id: str = Field(default_factory=new_id)


class StageOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    reached: bool = False
    agent: str = ""
    confidence: float = 0.0
    handoff: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    contract: dict[str, Any] | None = None


class RunRecord(BaseModel):
    """One pipeline execution, as the admin UI shows it."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    topic: str
    status: str = "unknown"
    agent_config_id: str = ""
    agent_config_name: str = ""
    started_at: str = Field(default_factory=_now)
    latency_ms: float = 0.0
    cached: bool = False
    stages: list[StageOutcome] = Field(default_factory=list)
    guardrail_interventions: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)

    @property
    def stages_reached(self) -> int:
        return sum(1 for s in self.stages if s.reached)


def summarize_report(
    report: dict[str, Any], *, agent_config_id: str = "", agent_config_name: str = ""
) -> RunRecord:
    """Fold a Phase 1 PipelineReport into the shape the UI lists and renders."""
    stages: list[StageOutcome] = []
    traces = {t.get("stage"): t for t in report.get("traces") or []}

    for name in STAGE_NAMES:
        contract = report.get(name)
        trace = traces.get(name, {})
        if contract:
            stages.append(
                StageOutcome(
                    stage=name,
                    reached=True,
                    agent=str(contract.get("agent", "")),
                    confidence=float(contract.get("confidence", 0.0)),
                    handoff=str((contract.get("handoff") or {}).get("summary", "")),
                    provider=str(trace.get("provider", "")),
                    latency_ms=float(trace.get("latency_ms", 0.0)),
                    contract=contract,
                )
            )
        else:
            stages.append(StageOutcome(stage=name, reached=False))

    interventions = [
        g for g in (report.get("guardrail_reports") or []) if g.get("action") != "allow"
    ]

    return RunRecord(
        run_id=str(report.get("run_id") or new_id()),
        topic=str(report.get("topic", "")),
        status=str(report.get("status", "unknown")),
        agent_config_id=agent_config_id,
        agent_config_name=agent_config_name,
        started_at=str(report.get("created_at") or _now()),
        latency_ms=float(report.get("total_latency_ms", 0.0)),
        cached=bool(report.get("cached", False)),
        stages=stages,
        guardrail_interventions=interventions,
        errors=[str(e) for e in (report.get("errors") or [])],
        report=report,
    )


# --------------------------------------------------------------------------
# Knowledge ingestion
# --------------------------------------------------------------------------


class IngestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=200000)
    title: str = ""
    source: str = ""


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str = Field(min_length=1, max_length=100)
    documents: list[IngestDocument] = Field(min_length=1, max_length=50)


class IngestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str
    documents: int
    chunks: int
    at: str = Field(default_factory=_now)
