"""Versioning with rollback, for agent configs and registered actions.

Rollback appends a *new* version carrying an older payload rather than deleting
the versions in between. Destructive rollback would erase the record of what was
briefly live -- which is precisely the question an incident review asks. The new
version records `rolled_back_from` so the history explains itself.

Every version write also emits an audit entry, so "who changed this" and "what
did it look like before" are answerable from either side.
"""

from __future__ import annotations

from typing import Any

from app.observability import get_logger, log_event
from instrumentation.audit import AuditLog
from instrumentation.schemas import EntityType, VersionRecord

logger = get_logger("agentforge.instrumentation.versioning")


class VersionNotFound(LookupError):
    """The requested version does not exist for that entity."""


class VersionStore:
    def __init__(self, store, audit: AuditLog | None = None) -> None:
        self.store = store
        self.audit = audit or AuditLog(store)

    async def current(self, entity_type: EntityType, entity_id: str) -> VersionRecord | None:
        return await self.store.latest_version(entity_type, entity_id)

    async def history(self, entity_type: EntityType, entity_id: str) -> list[VersionRecord]:
        return await self.store.list_versions(entity_type, entity_id)

    async def get(
        self, entity_type: EntityType, entity_id: str, version: int
    ) -> VersionRecord | None:
        return await self.store.get_version(entity_type, entity_id, version)

    async def record(
        self,
        entity_type: EntityType,
        entity_id: str,
        payload: dict[str, Any],
        *,
        actor: str = "unknown",
        note: str = "",
        action: str = "",
    ) -> VersionRecord:
        """Append the next version and audit the change against the previous one."""
        previous = await self.current(entity_type, entity_id)
        next_version = (previous.version + 1) if previous else 1

        record = await self.store.save_version(
            VersionRecord(
                entity_type=entity_type,
                entity_id=entity_id,
                version=next_version,
                payload=payload,
                actor=actor,
                note=note,
            )
        )
        verb = "create" if previous is None else "update"
        await self.audit.record(
            actor=actor,
            action=action or f"{entity_type.value}.{verb}",
            entity_type=entity_type.value,
            entity_id=entity_id,
            before=previous.payload if previous else {},
            after=record.payload,
        )
        log_event(
            logger,
            "instrumentation.version_recorded",
            entity=f"{entity_type.value}:{entity_id}",
            version=record.version,
            actor=actor,
        )
        return record

    async def rollback(
        self,
        entity_type: EntityType,
        entity_id: str,
        to_version: int,
        *,
        actor: str = "unknown",
        note: str = "",
    ) -> VersionRecord:
        """Restore an earlier payload as a new version.

        Raises VersionNotFound rather than silently doing nothing -- a rollback
        that quietly no-ops is how you discover, later, that production never
        actually moved.
        """
        target = await self.get(entity_type, entity_id, to_version)
        if target is None:
            raise VersionNotFound(
                f"{entity_type.value} {entity_id!r} has no version {to_version}"
            )

        current = await self.current(entity_type, entity_id)
        next_version = (current.version + 1) if current else 1

        record = await self.store.save_version(
            VersionRecord(
                entity_type=entity_type,
                entity_id=entity_id,
                version=next_version,
                payload=target.payload,
                actor=actor,
                note=note or f"rollback to v{to_version}",
                rolled_back_from=to_version,
            )
        )
        await self.audit.record(
            actor=actor,
            action=f"{entity_type.value}.rollback",
            entity_type=entity_type.value,
            entity_id=entity_id,
            before=current.payload if current else {},
            after=record.payload,
            summary=f"rolled back to v{to_version} (now v{record.version})",
        )
        log_event(
            logger,
            "instrumentation.rollback",
            entity=f"{entity_type.value}:{entity_id}",
            to_version=to_version,
            new_version=record.version,
            actor=actor,
        )
        return record
