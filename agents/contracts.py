"""Typed contracts exchanged between agents.

Each agent publishes exactly one contract. The next agent accepts only that
contract plus the shared blackboard -- never raw upstream state. A malformed
handoff therefore fails at the boundary that produced it instead of silently
degrading stage 4.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_id() -> str:
    return uuid.uuid4().hex


class Stage(StrEnum):
    ANALYSIS = "analysis"
    DEVELOP = "develop"
    TEST = "test"
    DEPLOY = "deploy"


AGENT_NAMES: dict[Stage, str] = {
    Stage.ANALYSIS: "Ana (Analyst)",
    Stage.DEVELOP: "Dev (Solution Architect)",
    Stage.TEST: "Tess (QA Engineer)",
    Stage.DEPLOY: "Dep (Release Engineer)",
}

STAGE_ORDER: tuple[Stage, ...] = (Stage.ANALYSIS, Stage.DEVELOP, Stage.TEST, Stage.DEPLOY)


# --------------------------------------------------------------------------
# Team coordination primitives
# --------------------------------------------------------------------------


class TeamMessage(BaseModel):
    """One utterance on the shared blackboard. This is what the UI renders."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    sender: Stage | Literal["orchestrator", "guardrails"]
    recipient: Stage | Literal["team", "orchestrator"] = "team"
    kind: Literal["status", "handoff", "question", "answer", "concern", "verdict"]
    content: str
    at: str = Field(default_factory=_now)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict = Field(default_factory=dict)
    outcome: Literal["ok", "error", "skipped"] = "ok"
    latency_ms: float = 0.0
    detail: str = ""


class NodeTrace(BaseModel):
    """Instrumentation record. Phase 5 trains on these; today they are logged."""

    model_config = ConfigDict(extra="forbid")

    stage: Stage
    agent: str
    reasoning: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    guardrail_input: str = "allow"
    guardrail_output: str = "allow"
    at: str = Field(default_factory=_now)


class Handoff(BaseModel):
    """The explicit part of the contract: what this agent tells the next one."""

    model_config = ConfigDict(extra="forbid")

    to: Stage | Literal["orchestrator"]
    summary: str = Field(description="What the next agent must know, in one paragraph.")
    accepted_inputs: list[str] = Field(
        default_factory=list, description="Upstream items this agent actually consumed."
    )
    open_questions: list[str] = Field(default_factory=list)
    blocking: bool = False


class AgentContract(BaseModel):
    """Base every stage contract shares."""

    model_config = ConfigDict(extra="forbid")

    stage: Stage
    agent: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    handoff: Handoff
    produced_at: str = Field(default_factory=_now)


# --------------------------------------------------------------------------
# Per-stage payloads
# --------------------------------------------------------------------------


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    rationale: str
    priority: Literal["high", "medium", "low"] = "medium"


class AnalysisContract(AgentContract):
    stage: Literal[Stage.ANALYSIS] = Stage.ANALYSIS
    problem_statement: str
    objectives: list[Objective] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class Component(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    responsibility: str
    depends_on: list[str] = Field(default_factory=list)


class DevelopContract(AgentContract):
    stage: Literal[Stage.DEVELOP] = Stage.DEVELOP
    approach: str
    components: list[Component] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    addressed_objectives: list[str] = Field(
        default_factory=list, description="Objective names from AnalysisContract."
    )


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    given: str
    when: str
    then: str
    priority: Literal["high", "medium", "low"] = "medium"
    targets_component: str = ""


class TestContract(AgentContract):
    stage: Literal[Stage.TEST] = Stage.TEST
    strategy: str
    test_cases: list[TestCase] = Field(default_factory=list)
    coverage_notes: str = ""
    uncovered_components: list[str] = Field(default_factory=list)
    verdict: Literal["pass", "pass_with_risk", "fail"] = "pass_with_risk"


class DeployContract(AgentContract):
    stage: Literal[Stage.DEPLOY] = Stage.DEPLOY
    strategy: str
    environments: list[str] = Field(default_factory=list)
    rollout_steps: list[str] = Field(default_factory=list)
    monitoring: list[str] = Field(default_factory=list)
    rollback_plan: str = ""
    go_no_go: Literal["go", "conditional_go", "no_go"] = "conditional_go"


STAGE_CONTRACTS: dict[Stage, type[AgentContract]] = {
    Stage.ANALYSIS: AnalysisContract,
    Stage.DEVELOP: DevelopContract,
    Stage.TEST: TestContract,
    Stage.DEPLOY: DeployContract,
}


# --------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------


class PipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(default_factory=new_id)
    # Optional so a client can subscribe to the event stream before the run starts.
    run_id: str = Field(default_factory=new_id)
    bypass_cache: bool = False


class GuardrailFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Literal["regex", "classifier", "llm"]
    category: str
    detail: str
    severity: Annotated[float, Field(ge=0.0, le=1.0)]
    span: str = ""


class GuardrailReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = ""
    direction: Literal["input", "output"] = "input"
    action: Literal["allow", "redact", "block"] = "allow"
    findings: list[GuardrailFinding] = Field(default_factory=list)
    sanitized_text: str = ""
    risk_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    latency_ms: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.action != "block"


class PipelineReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    session_id: str
    topic: str
    status: Literal["completed", "halted", "failed"]
    created_at: str = Field(default_factory=_now)
    cached: bool = False

    analysis: AnalysisContract | None = None
    develop: DevelopContract | None = None
    test: TestContract | None = None
    deploy: DeployContract | None = None

    team_messages: list[TeamMessage] = Field(default_factory=list)
    traces: list[NodeTrace] = Field(default_factory=list)
    guardrail_reports: list[GuardrailReport] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0
