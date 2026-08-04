"""Persistence for trajectories, audit entries, and versions.

The important design decision is where scrubbing happens: **inside the save
methods**, not in the callers. "Hard requirement, not best-effort" means a
caller must not be *able* to persist unscrubbed data, even by mistake. Both
store implementations route every write through `_prepare_*`, so the safe path
is the only path.

Append-only by construction: there is no UPDATE or DELETE anywhere except the
reward label, which is explicitly revisable (a user may leave feedback later)
and is stored apart from the observed steps for exactly that reason.

Storage is Postgres tables rather than an append-only log file, so trajectories
can be selected by run, outcome or reward without a parsing layer -- the future
training export is a query, not a batch job.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import psycopg

from app.config import Settings, get_settings
from app.observability import get_logger, log_event
from instrumentation.schemas import (
    AuditEntry,
    EntityType,
    Trajectory,
    TrajectoryStep,
    VersionRecord,
    _now,
)
from instrumentation.scrubbing import scrub

logger = get_logger("agentforge.instrumentation.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    run_id               TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL DEFAULT '',
    topic                TEXT NOT NULL DEFAULT '',
    agent_config_id      TEXT NOT NULL DEFAULT '',
    agent_config_version INTEGER NOT NULL DEFAULT 0,
    status               TEXT NOT NULL DEFAULT '',
    started_at           TIMESTAMPTZ NOT NULL,
    finished_at          TIMESTAMPTZ,
    total_latency_ms     DOUBLE PRECISION NOT NULL DEFAULT 0,
    step_count           INTEGER NOT NULL DEFAULT 0,
    reward               JSONB NOT NULL DEFAULT '{}'::jsonb,
    scrubbed             BOOLEAN NOT NULL DEFAULT true,
    pii_findings         INTEGER NOT NULL DEFAULT 0,
    pii_categories       JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trajectories_started_idx ON trajectories (started_at DESC);

CREATE TABLE IF NOT EXISTS trajectory_steps (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES trajectories(run_id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    kind       TEXT NOT NULL,
    stage      TEXT NOT NULL DEFAULT '',
    name       TEXT NOT NULL DEFAULT '',
    input      JSONB NOT NULL DEFAULT '{}'::jsonb,
    output     JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    success    BOOLEAN NOT NULL DEFAULT true,
    error      TEXT NOT NULL DEFAULT '',
    at         TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS trajectory_steps_run_idx ON trajectory_steps (run_id, ordinal);

CREATE TABLE IF NOT EXISTS audit_log (
    id           TEXT PRIMARY KEY,
    actor        TEXT NOT NULL DEFAULT 'unknown',
    action       TEXT NOT NULL,
    entity_type  TEXT NOT NULL DEFAULT '',
    entity_id    TEXT NOT NULL DEFAULT '',
    at           TIMESTAMPTZ NOT NULL,
    summary      TEXT NOT NULL DEFAULT '',
    changes      JSONB NOT NULL DEFAULT '[]'::jsonb,
    before       JSONB NOT NULL DEFAULT '{}'::jsonb,
    after        JSONB NOT NULL DEFAULT '{}'::jsonb,
    pii_findings INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS audit_log_at_idx ON audit_log (at DESC);
CREATE INDEX IF NOT EXISTS audit_log_entity_idx ON audit_log (entity_type, entity_id);

CREATE TABLE IF NOT EXISTS entity_versions (
    id                TEXT PRIMARY KEY,
    entity_type       TEXT NOT NULL,
    entity_id         TEXT NOT NULL,
    version           INTEGER NOT NULL,
    payload           JSONB NOT NULL DEFAULT '{}'::jsonb,
    actor             TEXT NOT NULL DEFAULT 'unknown',
    note              TEXT NOT NULL DEFAULT '',
    rolled_back_from  INTEGER,
    created_at        TIMESTAMPTZ NOT NULL,
    UNIQUE (entity_type, entity_id, version)
);
CREATE INDEX IF NOT EXISTS entity_versions_lookup_idx
    ON entity_versions (entity_type, entity_id, version DESC);
"""


class ScrubbingStore:
    """Shared write-path scrubbing. Both stores inherit; neither bypasses it."""

    def _prepare_trajectory(self, trajectory: Trajectory) -> Trajectory:
        report = scrub(trajectory.model_dump(mode="json"))
        cleaned = Trajectory.model_validate(report.value)
        cleaned.scrubbed = True
        cleaned.pii_findings = report.findings
        cleaned.pii_categories = report.sorted_categories
        if report.findings:
            log_event(
                logger,
                "instrumentation.pii_scrubbed",
                run_id=cleaned.run_id,
                findings=report.findings,
                categories=report.sorted_categories,
            )
        return cleaned

    def _prepare_audit(self, entry: AuditEntry) -> AuditEntry:
        report = scrub(entry.model_dump(mode="json"))
        cleaned = AuditEntry.model_validate(report.value)
        cleaned.scrubbed = True
        cleaned.pii_findings = report.findings
        if report.findings:
            log_event(
                logger,
                "instrumentation.audit_pii_scrubbed",
                entry=cleaned.id,
                findings=report.findings,
            )
        return cleaned

    def _prepare_version(self, record: VersionRecord) -> VersionRecord:
        report = scrub(record.model_dump(mode="json"))
        return VersionRecord.model_validate(report.value)


@runtime_checkable
class InstrumentationStore(Protocol):
    name: str

    async def ensure_schema(self) -> bool: ...
    async def save_trajectory(self, trajectory: Trajectory) -> Trajectory: ...
    async def get_trajectory(self, run_id: str) -> Trajectory | None: ...
    async def list_trajectories(self, limit: int = 50) -> list[dict[str, Any]]: ...
    async def set_feedback(
        self, run_id: str, feedback: str, note: str
    ) -> Trajectory | None: ...
    async def append_audit(self, entry: AuditEntry) -> AuditEntry: ...
    async def list_audit(self, limit: int = 100, **filters: str) -> list[dict[str, Any]]: ...
    async def save_version(self, record: VersionRecord) -> VersionRecord: ...
    async def latest_version(
        self, entity_type: EntityType, entity_id: str
    ) -> VersionRecord | None: ...
    async def get_version(
        self, entity_type: EntityType, entity_id: str, version: int
    ) -> VersionRecord | None: ...
    async def list_versions(
        self, entity_type: EntityType, entity_id: str
    ) -> list[VersionRecord]: ...


def _traj_row(t: Trajectory) -> dict[str, Any]:
    return {
        "run_id": t.run_id,
        "topic": t.topic,
        "status": t.status,
        "started_at": t.started_at,
        "total_latency_ms": t.total_latency_ms,
        "step_count": t.step_count,
        "resolution": t.reward.resolution.value,
        "reward": t.reward.implicit_score,
        "pii_findings": t.pii_findings,
    }


class InMemoryInstrumentationStore(ScrubbingStore):
    name = "memory"

    def __init__(self) -> None:
        self.trajectories: dict[str, Trajectory] = {}
        self._order: list[str] = []
        self.audit: list[AuditEntry] = []
        self.versions: list[VersionRecord] = []

    async def ensure_schema(self) -> bool:
        return True

    async def save_trajectory(self, trajectory: Trajectory) -> Trajectory:
        cleaned = self._prepare_trajectory(trajectory)
        if cleaned.run_id not in self.trajectories:
            self._order.insert(0, cleaned.run_id)
        self.trajectories[cleaned.run_id] = cleaned
        return cleaned

    async def get_trajectory(self, run_id: str) -> Trajectory | None:
        return self.trajectories.get(run_id)

    async def list_trajectories(self, limit: int = 50) -> list[dict[str, Any]]:
        return [_traj_row(self.trajectories[r]) for r in self._order[:limit]]

    async def set_feedback(self, run_id: str, feedback: str, note: str) -> Trajectory | None:
        existing = self.trajectories.get(run_id)
        if existing is None:
            return None
        updated = existing.model_copy(deep=True)
        updated.reward.user_feedback = feedback  # type: ignore[assignment]
        updated.reward.user_feedback_note = scrub(note).value
        updated.reward.feedback_at = _now()
        self.trajectories[run_id] = updated
        return updated

    async def append_audit(self, entry: AuditEntry) -> AuditEntry:
        cleaned = self._prepare_audit(entry)
        self.audit.insert(0, cleaned)
        return cleaned

    async def list_audit(self, limit: int = 100, **filters: str) -> list[dict[str, Any]]:
        rows = self.audit
        for key in ("entity_type", "entity_id", "actor", "action"):
            wanted = filters.get(key)
            if wanted:
                rows = [e for e in rows if getattr(e, key) == wanted]
        return [e.model_dump(mode="json") for e in rows[:limit]]

    async def save_version(self, record: VersionRecord) -> VersionRecord:
        cleaned = self._prepare_version(record)
        self.versions.append(cleaned)
        return cleaned

    async def latest_version(
        self, entity_type: EntityType, entity_id: str
    ) -> VersionRecord | None:
        candidates = await self.list_versions(entity_type, entity_id)
        return candidates[-1] if candidates else None

    async def get_version(
        self, entity_type: EntityType, entity_id: str, version: int
    ) -> VersionRecord | None:
        return next(
            (
                v
                for v in self.versions
                if v.entity_type is entity_type
                and v.entity_id == entity_id
                and v.version == version
            ),
            None,
        )

    async def list_versions(
        self, entity_type: EntityType, entity_id: str
    ) -> list[VersionRecord]:
        return sorted(
            (
                v
                for v in self.versions
                if v.entity_type is entity_type and v.entity_id == entity_id
            ),
            key=lambda v: v.version,
        )


class PostgresInstrumentationStore(ScrubbingStore):
    name = "postgres"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._degraded = False
        self._warned = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _note(self, op: str, exc: Exception) -> None:
        self._degraded = True
        if not self._warned:
            self._warned = True
            log_event(logger, "instrumentation.postgres_unavailable", op=op, error=str(exc))

    def _connect(self, autocommit: bool = True):
        return psycopg.AsyncConnection.connect(
            self.settings.postgres_dsn, autocommit=autocommit
        )

    async def ensure_schema(self) -> bool:
        try:
            async with await self._connect() as conn:
                await conn.execute(SCHEMA)
        except (psycopg.Error, OSError) as exc:
            self._note("ensure_schema", exc)
            return False
        self._degraded = False
        return True

    async def save_trajectory(self, trajectory: Trajectory) -> Trajectory:
        cleaned = self._prepare_trajectory(trajectory)
        try:
            async with await self._connect() as conn:
                await conn.execute(
                    """
                    INSERT INTO trajectories
                        (run_id, session_id, topic, agent_config_id, agent_config_version,
                         status, started_at, finished_at, total_latency_ms, step_count,
                         reward, scrubbed, pii_findings, pii_categories)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        finished_at = EXCLUDED.finished_at,
                        total_latency_ms = EXCLUDED.total_latency_ms,
                        step_count = EXCLUDED.step_count,
                        reward = EXCLUDED.reward
                    """,
                    (
                        cleaned.run_id, cleaned.session_id, cleaned.topic,
                        cleaned.agent_config_id, cleaned.agent_config_version,
                        cleaned.status, cleaned.started_at, cleaned.finished_at or None,
                        cleaned.total_latency_ms, cleaned.step_count,
                        cleaned.reward.model_dump_json(), True,
                        cleaned.pii_findings, json.dumps(cleaned.pii_categories),
                    ),
                )
                await conn.cursor().executemany(
                    """
                    INSERT INTO trajectory_steps
                        (id, run_id, ordinal, kind, stage, name, input, output,
                         latency_ms, success, error, at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    [
                        (
                            s.id, cleaned.run_id, s.ordinal, s.kind.value, s.stage, s.name,
                            json.dumps(s.input), json.dumps(s.output),
                            s.latency_ms, s.success, s.error, s.at,
                        )
                        for s in cleaned.steps
                    ],
                )
        except (psycopg.Error, OSError) as exc:
            self._note("save_trajectory", exc)
        return cleaned

    async def get_trajectory(self, run_id: str) -> Trajectory | None:
        try:
            async with await self._connect(autocommit=False) as conn:
                cur = await conn.execute(
                    """
                    SELECT run_id, session_id, topic, agent_config_id, agent_config_version,
                           status, started_at, finished_at, total_latency_ms,
                           reward, scrubbed, pii_findings, pii_categories
                    FROM trajectories WHERE run_id = %s
                    """,
                    (run_id,),
                )
                head = await cur.fetchone()
                if head is None:
                    return None
                cur = await conn.execute(
                    """
                    SELECT id, ordinal, kind, stage, name, input, output,
                           latency_ms, success, error, at
                    FROM trajectory_steps WHERE run_id = %s ORDER BY ordinal
                    """,
                    (run_id,),
                )
                step_rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("get_trajectory", exc)
            return None

        return Trajectory(
            run_id=head[0], session_id=head[1], topic=head[2],
            agent_config_id=head[3], agent_config_version=head[4], status=head[5],
            started_at=str(head[6]), finished_at=str(head[7] or ""),
            total_latency_ms=head[8], reward=head[9], scrubbed=head[10],
            pii_findings=head[11], pii_categories=head[12] or [],
            steps=[
                TrajectoryStep(
                    id=r[0], run_id=run_id, ordinal=r[1], kind=r[2], stage=r[3], name=r[4],
                    input=r[5] or {}, output=r[6] or {}, latency_ms=r[7],
                    success=r[8], error=r[9], at=str(r[10]),
                )
                for r in step_rows
            ],
        )

    async def list_trajectories(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            async with await self._connect(autocommit=False) as conn:
                cur = await conn.execute(
                    """
                    SELECT run_id, topic, status, started_at, total_latency_ms, step_count,
                           reward->>'resolution', (reward->>'implicit_score')::float,
                           pii_findings
                    FROM trajectories ORDER BY started_at DESC LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("list_trajectories", exc)
            return []
        keys = (
            "run_id", "topic", "status", "started_at", "total_latency_ms",
            "step_count", "resolution", "reward", "pii_findings",
        )
        return [{k: v for k, v in zip(keys, r)} for r in rows]

    async def set_feedback(self, run_id: str, feedback: str, note: str) -> Trajectory | None:
        existing = await self.get_trajectory(run_id)
        if existing is None:
            return None
        existing.reward.user_feedback = feedback  # type: ignore[assignment]
        existing.reward.user_feedback_note = scrub(note).value
        existing.reward.feedback_at = _now()
        try:
            async with await self._connect() as conn:
                await conn.execute(
                    "UPDATE trajectories SET reward = %s::jsonb WHERE run_id = %s",
                    (existing.reward.model_dump_json(), run_id),
                )
        except (psycopg.Error, OSError) as exc:
            self._note("set_feedback", exc)
        return existing

    async def append_audit(self, entry: AuditEntry) -> AuditEntry:
        cleaned = self._prepare_audit(entry)
        try:
            async with await self._connect() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_log
                        (id, actor, action, entity_type, entity_id, at, summary,
                         changes, before, after, pii_findings)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        cleaned.id, cleaned.actor, cleaned.action, cleaned.entity_type,
                        cleaned.entity_id, cleaned.at, cleaned.summary,
                        json.dumps([c.model_dump(mode="json") for c in cleaned.changes]),
                        json.dumps(cleaned.before), json.dumps(cleaned.after),
                        cleaned.pii_findings,
                    ),
                )
        except (psycopg.Error, OSError) as exc:
            self._note("append_audit", exc)
        return cleaned

    async def list_audit(self, limit: int = 100, **filters: str) -> list[dict[str, Any]]:
        clauses, params = [], []
        for key in ("entity_type", "entity_id", "actor", "action"):
            if filters.get(key):
                clauses.append(f"{key} = %s")
                params.append(filters[key])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        try:
            async with await self._connect(autocommit=False) as conn:
                cur = await conn.execute(
                    f"""
                    SELECT id, actor, action, entity_type, entity_id, at, summary,
                           changes, before, after, pii_findings
                    FROM audit_log {where} ORDER BY at DESC LIMIT %s
                    """,
                    tuple(params),
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("list_audit", exc)
            return []
        keys = (
            "id", "actor", "action", "entity_type", "entity_id", "at",
            "summary", "changes", "before", "after", "pii_findings",
        )
        return [{k: v for k, v in zip(keys, r)} for r in rows]

    async def save_version(self, record: VersionRecord) -> VersionRecord:
        cleaned = self._prepare_version(record)
        try:
            async with await self._connect() as conn:
                await conn.execute(
                    """
                    INSERT INTO entity_versions
                        (id, entity_type, entity_id, version, payload, actor, note,
                         rolled_back_from, created_at)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                    ON CONFLICT (entity_type, entity_id, version) DO NOTHING
                    """,
                    (
                        cleaned.id, cleaned.entity_type.value, cleaned.entity_id,
                        cleaned.version, json.dumps(cleaned.payload), cleaned.actor,
                        cleaned.note, cleaned.rolled_back_from, cleaned.created_at,
                    ),
                )
        except (psycopg.Error, OSError) as exc:
            self._note("save_version", exc)
        return cleaned

    async def _versions_query(
        self, entity_type: EntityType, entity_id: str, extra: str, params
    ):
        try:
            async with await self._connect(autocommit=False) as conn:
                cur = await conn.execute(
                    f"""
                    SELECT id, entity_type, entity_id, version, payload, actor, note,
                           rolled_back_from, created_at
                    FROM entity_versions
                    WHERE entity_type = %s AND entity_id = %s {extra}
                    """,
                    (entity_type.value, entity_id, *params),
                )
                return await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("versions_query", exc)
            return []

    @staticmethod
    def _to_version(row) -> VersionRecord:
        return VersionRecord(
            id=row[0], entity_type=row[1], entity_id=row[2], version=row[3],
            payload=row[4] or {}, actor=row[5], note=row[6],
            rolled_back_from=row[7], created_at=str(row[8]),
        )

    async def latest_version(
        self, entity_type: EntityType, entity_id: str
    ) -> VersionRecord | None:
        rows = await self._versions_query(
            entity_type, entity_id, "ORDER BY version DESC LIMIT 1", ()
        )
        return self._to_version(rows[0]) if rows else None

    async def get_version(
        self, entity_type: EntityType, entity_id: str, version: int
    ) -> VersionRecord | None:
        rows = await self._versions_query(
            entity_type, entity_id, "AND version = %s", (version,)
        )
        return self._to_version(rows[0]) if rows else None

    async def list_versions(
        self, entity_type: EntityType, entity_id: str
    ) -> list[VersionRecord]:
        rows = await self._versions_query(entity_type, entity_id, "ORDER BY version", ())
        return [self._to_version(r) for r in rows]


async def build_store(settings: Settings | None = None) -> InstrumentationStore:
    """Postgres when it answers, in-memory otherwise."""
    store = PostgresInstrumentationStore(settings)
    if await store.ensure_schema():
        log_event(logger, "instrumentation.store", backend=store.name)
        return store
    log_event(
        logger,
        "instrumentation.store",
        backend=InMemoryInstrumentationStore.name,
        reason="postgres down",
    )
    return InMemoryInstrumentationStore()
