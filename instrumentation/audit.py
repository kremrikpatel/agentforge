"""Audit logging: who changed what, when, and exactly what moved.

The diff is structural rather than textual. A reviewer asking "what changed in
this config?" wants `stages[0].instructions` named, not a unified diff of
pretty-printed JSON they have to read character by character.
"""

from __future__ import annotations

from typing import Any

from app.observability import get_logger, log_event
from instrumentation.schemas import AuditEntry, FieldChange

logger = get_logger("agentforge.instrumentation.audit")

_MISSING = object()


def diff_values(before: Any, after: Any, path: str = "") -> list[FieldChange]:
    """Recursive structural diff, emitting dotted/indexed paths.

    Lists are compared position-wise: for the shapes we version (stages, actions,
    knowledge refs) an index is meaningful, and whole-list replacement would
    report "everything changed" for a one-word edit.
    """
    changes: list[FieldChange] = []

    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            changes.extend(
                diff_values(before.get(key, _MISSING), after.get(key, _MISSING), child)
            )
        return changes

    if isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            child = f"{path}[{index}]"
            changes.extend(
                diff_values(
                    before[index] if index < len(before) else _MISSING,
                    after[index] if index < len(after) else _MISSING,
                    child,
                )
            )
        return changes

    if before is _MISSING and after is not _MISSING:
        return [FieldChange(path=path, before=None, after=after, change="added")]
    if after is _MISSING and before is not _MISSING:
        return [FieldChange(path=path, before=before, after=None, change="removed")]
    if before != after:
        return [FieldChange(path=path, before=before, after=after, change="changed")]
    return []


def summarize(changes: list[FieldChange], limit: int = 4) -> str:
    if not changes:
        return "no field changes"
    head = ", ".join(f"{c.change} {c.path}" for c in changes[:limit])
    extra = len(changes) - limit
    return head + (f", +{extra} more" if extra > 0 else "")


class AuditLog:
    """Thin recorder over the store. Scrubbing happens inside the store."""

    def __init__(self, store) -> None:
        self.store = store

    async def record(
        self,
        *,
        actor: str,
        action: str,
        entity_type: str = "",
        entity_id: str = "",
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        summary: str = "",
    ) -> AuditEntry:
        before = before or {}
        after = after or {}
        changes = diff_values(before, after)

        entry = await self.store.append_audit(
            AuditEntry(
                actor=actor,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                summary=summary or summarize(changes),
                changes=changes,
                before=before,
                after=after,
            )
        )
        log_event(
            logger,
            "instrumentation.audit",
            actor=actor,
            action=action,
            entity=f"{entity_type}:{entity_id}",
            changes=len(changes),
        )
        return entry

    async def list(self, limit: int = 100, **filters: str) -> list[dict[str, Any]]:
        return await self.store.list_audit(limit, **filters)
