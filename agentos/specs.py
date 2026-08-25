"""Typed specs for the AgentForge OS layer.

Everything the OS knows about its agents lives in markdown files with TOML
frontmatter (parsed with stdlib ``tomllib`` -- no new dependencies). These
models are the validated, in-memory form of those files. The pipeline reads
only these; it never parses markdown itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.contracts import Stage


class ModelPolicy(BaseModel):
    """Per-persona LLM call parameters. Applied on top of gateway defaults."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = None
    provider: str = ""


class PersonaSpec(BaseModel):
    """One specialist agent: identity text plus operational metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    role: str
    stage: Stage
    charter: str
    model: ModelPolicy = Field(default_factory=ModelPolicy)
    tools: list[str] = Field(default_factory=list)
    memory_scope: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    version: int = 1

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "stage": self.stage.value,
            "triggers": self.triggers,
            "tools": self.tools,
            "memory_scope": self.memory_scope,
            "model": self.model.model_dump(),
            "version": self.version,
        }


class RoutingRule(BaseModel):
    """Declarative intent match from the kernel: keywords to a destination.

    Exactly one of ``stage`` or ``command`` should be set. Matching is a plain
    keyword intersection -- inspectable, debuggable, no embeddings required.
    """

    model_config = ConfigDict(extra="forbid")

    intent: str
    match: list[str] = Field(min_length=1)
    stage: Stage | None = None
    command: str | None = None

    def matches(self, task: str) -> bool:
        lowered = task.lower()
        return any(kw.lower() in lowered for kw in self.match)


class CommandSpec(BaseModel):
    """A named workflow: an ordered subset of stages plus run policy."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    description: str = ""
    stages: list[Stage] = Field(min_length=1)
    bypass_cache: bool = False

    def ordered_stages(self) -> list[Stage]:
        """Canonical pipeline order regardless of how the file lists them."""
        return [s for s in Stage if s in set(self.stages)]


class RouteDecision(BaseModel):
    """Outcome of kernel routing for one free-form task."""

    model_config = ConfigDict(extra="forbid")

    task: str
    intent: str = ""
    target_kind: Literal["pipeline", "command", "fallback"] = "fallback"
    stages: list[Stage]
    command: str | None = None
    rationale: str = ""


class KernelSpec(BaseModel):
    """The OS kernel: identity, shared team protocol, routing table."""

    model_config = ConfigDict(extra="forbid")

    name: str = "AgentForge OS"
    team_charter: str = ""
    identity: str = ""
    routing: list[RoutingRule] = Field(default_factory=list)

    def route(self, task: str) -> tuple[RoutingRule | None, str]:
        """First matching rule wins. Returns (rule, rationale)."""
        for rule in self.routing:
            if rule.matches(task):
                return (
                    rule,
                    f"matched {rule.intent} via keywords {rule.match}",
                )
        return None, "no routing rule matched; defaulting to full pipeline"
