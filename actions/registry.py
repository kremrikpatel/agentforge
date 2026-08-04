"""A versioned registry of third-party actions.

Scope note: `actions/` did not exist before this phase. What is here is the
*definition and versioning* half only -- registering an action, editing it,
inspecting its history, rolling it back. There is deliberately no dispatch,
retry, or schema-validation runtime; that is a later phase, and building it now
would be scope this phase did not ask for.

Versions live in the shared instrumentation version store, so an action's
history and an agent config's history are queried the same way and audited by
the same code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from instrumentation.schemas import EntityType, VersionRecord
from instrumentation.versioning import VersionStore


class ActionDefinition(BaseModel):
    """What an action *is*. Nothing here is invoked by this module."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)
    endpoint: str = Field(default="", max_length=2000)
    method: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    timeout_s: float = Field(default=30.0, ge=0.1, le=600.0)
    enabled: bool = False


class ActionRegistry:
    """Register, edit, inspect and roll back action definitions.

    The action's name is its identity, so history and rollback read naturally
    (`rollback("create_ticket", to_version=2)`).
    """

    entity_type = EntityType.ACTION

    def __init__(self, versions: VersionStore) -> None:
        self.versions = versions

    async def register(
        self, action: ActionDefinition, *, actor: str = "unknown", note: str = ""
    ) -> VersionRecord:
        """First registration is v1; re-registering an existing name is an edit."""
        return await self.versions.record(
            self.entity_type,
            action.name,
            action.model_dump(mode="json"),
            actor=actor,
            note=note,
            action="action.register",
        )

    async def update(
        self, action: ActionDefinition, *, actor: str = "unknown", note: str = ""
    ) -> VersionRecord:
        return await self.versions.record(
            self.entity_type,
            action.name,
            action.model_dump(mode="json"),
            actor=actor,
            note=note,
            action="action.update",
        )

    async def get(self, name: str) -> ActionDefinition | None:
        record = await self.versions.current(self.entity_type, name)
        return ActionDefinition.model_validate(record.payload) if record else None

    async def at_version(self, name: str, version: int) -> ActionDefinition | None:
        record = await self.versions.get(self.entity_type, name, version)
        return ActionDefinition.model_validate(record.payload) if record else None

    async def history(self, name: str) -> list[VersionRecord]:
        return await self.versions.history(self.entity_type, name)

    async def rollback(
        self, name: str, to_version: int, *, actor: str = "unknown"
    ) -> ActionDefinition:
        """Restore an earlier definition as a new version. Raises VersionNotFound."""
        record = await self.versions.rollback(self.entity_type, name, to_version, actor=actor)
        return ActionDefinition.model_validate(record.payload)

    async def snapshot(self, name: str) -> dict[str, Any]:
        """Current definition plus its version number, for display."""
        record = await self.versions.current(self.entity_type, name)
        if record is None:
            return {}
        return {"version": record.version, "action": record.payload}
