"""Persistence for agent configs and run history.

Same shape as the other phases: a protocol with a Postgres implementation and an
in-memory one, chosen at startup by whether Postgres answers. The in-memory store
is what makes the console usable (and testable) with no database at all -- it
just forgets everything on restart, which the UI says out loud.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import psycopg

from app.config import Settings, get_settings
from app.observability import get_logger, log_event
from web.schemas import AgentConfig, RunRecord

logger = get_logger("agentforge.web.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS web_agent_configs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    payload     JSONB NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS web_runs (
    run_id            TEXT PRIMARY KEY,
    topic             TEXT NOT NULL,
    status            TEXT NOT NULL,
    agent_config_id   TEXT NOT NULL DEFAULT '',
    agent_config_name TEXT NOT NULL DEFAULT '',
    started_at        TIMESTAMPTZ NOT NULL,
    latency_ms        DOUBLE PRECISION NOT NULL DEFAULT 0,
    cached            BOOLEAN NOT NULL DEFAULT false,
    stages_reached    INTEGER NOT NULL DEFAULT 0,
    interventions     INTEGER NOT NULL DEFAULT 0,
    record            JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS web_runs_started_idx ON web_runs (started_at DESC);
"""


def _row(record: RunRecord) -> dict[str, Any]:
    """The listing projection -- what the runs table shows, without the report."""
    return {
        "run_id": record.run_id,
        "topic": record.topic,
        "status": record.status,
        "agent_config_id": record.agent_config_id,
        "agent_config_name": record.agent_config_name,
        "started_at": record.started_at,
        "latency_ms": record.latency_ms,
        "cached": record.cached,
        "stages_reached": record.stages_reached,
        "interventions": len(record.guardrail_interventions),
    }


@runtime_checkable
class WebStore(Protocol):
    name: str

    async def ensure_schema(self) -> bool: ...
    async def save_config(self, config: AgentConfig) -> bool: ...
    async def get_config(self, config_id: str) -> AgentConfig | None: ...
    async def list_configs(self) -> list[AgentConfig]: ...
    async def save_run(self, record: RunRecord) -> bool: ...
    async def get_run(self, run_id: str) -> RunRecord | None: ...
    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]: ...


class InMemoryWebStore:
    name = "memory"

    def __init__(self) -> None:
        self.configs: dict[str, AgentConfig] = {}
        self.runs: dict[str, RunRecord] = {}
        self._order: list[str] = []

    async def ensure_schema(self) -> bool:
        return True

    async def save_config(self, config: AgentConfig) -> bool:
        self.configs[config.id] = config
        return True

    async def get_config(self, config_id: str) -> AgentConfig | None:
        return self.configs.get(config_id)

    async def list_configs(self) -> list[AgentConfig]:
        return sorted(self.configs.values(), key=lambda c: c.updated_at, reverse=True)

    async def save_run(self, record: RunRecord) -> bool:
        if record.run_id not in self.runs:
            self._order.insert(0, record.run_id)
        self.runs[record.run_id] = record
        return True

    async def get_run(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return [_row(self.runs[r]) for r in self._order[:limit] if r in self.runs]


class PostgresWebStore:
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
            log_event(logger, "web.postgres_unavailable", op=op, error=str(exc))

    async def ensure_schema(self) -> bool:
        try:
            async with await psycopg.AsyncConnection.connect(
                self.settings.postgres_dsn, autocommit=True
            ) as conn:
                await conn.execute(SCHEMA)
        except (psycopg.Error, OSError) as exc:
            self._note("ensure_schema", exc)
            return False
        self._degraded = False
        return True

    async def save_config(self, config: AgentConfig) -> bool:
        try:
            async with await psycopg.AsyncConnection.connect(
                self.settings.postgres_dsn, autocommit=True
            ) as conn:
                await conn.execute(
                    """
                    INSERT INTO web_agent_configs
                        (id, name, description, payload, version, created_at, updated_at)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        payload = EXCLUDED.payload,
                        version = EXCLUDED.version,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        config.id,
                        config.name,
                        config.description,
                        config.model_dump_json(),
                        config.version,
                        config.created_at,
                        config.updated_at,
                    ),
                )
        except (psycopg.Error, OSError) as exc:
            self._note("save_config", exc)
            return False
        self._degraded = False
        return True

    async def get_config(self, config_id: str) -> AgentConfig | None:
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    "SELECT payload FROM web_agent_configs WHERE id = %s", (config_id,)
                )
                row = await cur.fetchone()
        except (psycopg.Error, OSError) as exc:
            self._note("get_config", exc)
            return None
        return AgentConfig.model_validate(row[0]) if row else None

    async def list_configs(self) -> list[AgentConfig]:
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    "SELECT payload FROM web_agent_configs ORDER BY updated_at DESC LIMIT 200"
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("list_configs", exc)
            return []
        out: list[AgentConfig] = []
        for row in rows:
            try:
                out.append(AgentConfig.model_validate(row[0]))
            except ValueError:
                continue  # written by an older shape; skip rather than fail the list
        return out

    async def save_run(self, record: RunRecord) -> bool:
        try:
            async with await psycopg.AsyncConnection.connect(
                self.settings.postgres_dsn, autocommit=True
            ) as conn:
                await conn.execute(
                    """
                    INSERT INTO web_runs
                        (run_id, topic, status, agent_config_id, agent_config_name,
                         started_at, latency_ms, cached, stages_reached, interventions, record)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        latency_ms = EXCLUDED.latency_ms,
                        stages_reached = EXCLUDED.stages_reached,
                        interventions = EXCLUDED.interventions,
                        record = EXCLUDED.record
                    """,
                    (
                        record.run_id,
                        record.topic,
                        record.status,
                        record.agent_config_id,
                        record.agent_config_name,
                        record.started_at,
                        record.latency_ms,
                        record.cached,
                        record.stages_reached,
                        len(record.guardrail_interventions),
                        record.model_dump_json(),
                    ),
                )
        except (psycopg.Error, OSError) as exc:
            self._note("save_run", exc)
            return False
        self._degraded = False
        return True

    async def get_run(self, run_id: str) -> RunRecord | None:
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    "SELECT record FROM web_runs WHERE run_id = %s", (run_id,)
                )
                row = await cur.fetchone()
        except (psycopg.Error, OSError) as exc:
            self._note("get_run", exc)
            return None
        return RunRecord.model_validate(row[0]) if row else None

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    """
                    SELECT run_id, topic, status, agent_config_id, agent_config_name,
                           started_at, latency_ms, cached, stages_reached, interventions
                    FROM web_runs ORDER BY started_at DESC LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("list_runs", exc)
            return []
        keys = (
            "run_id", "topic", "status", "agent_config_id", "agent_config_name",
            "started_at", "latency_ms", "cached", "stages_reached", "interventions",
        )
        return [{k: v for k, v in zip(keys, r)} for r in rows]


async def build_store(settings: Settings | None = None) -> WebStore:
    """Postgres when it answers, in-memory otherwise."""
    store = PostgresWebStore(settings)
    if await store.ensure_schema():
        log_event(logger, "web.store", backend=store.name)
        return store
    log_event(logger, "web.store", backend=InMemoryWebStore.name, reason="postgres down")
    return InMemoryWebStore()


def config_as_json(config: AgentConfig) -> str:
    return json.dumps(config.model_dump(mode="json"), indent=2)
