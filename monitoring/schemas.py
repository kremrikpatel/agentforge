"""Alert and severity types the rule engine, router, and notifiers share."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Severity(StrEnum):
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """One fired condition.

    `fingerprint` identifies the *condition*, not the instance -- dedup and
    silence key off it so a flapping rule collapses to one notification
    instead of one per evaluation. Defaults to `rule` when a rule fires at
    most one alert at a time; rules that can fire several distinct alerts at
    once (circuit_breaker_trips, one per action name) set it explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    rule: str
    severity: Severity
    summary: str
    detail: str = ""
    value: float = 0.0
    threshold: float = 0.0
    fingerprint: str = ""
    fired_at: str = Field(default_factory=_now)

    @model_validator(mode="after")
    def _default_fingerprint(self) -> "Alert":
        if not self.fingerprint:
            self.fingerprint = self.rule
        return self
