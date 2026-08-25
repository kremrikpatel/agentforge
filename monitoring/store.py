"""Durable history of alerts that actually fired.

Separate from `dedup.py` on purpose, because the two answer different
questions with different lifetimes: Redis holds *live suppression* (TTL
shaped -- is this fingerprint muted right now?), Postgres holds the *log*
(append-only -- what fired, when, and where was it delivered?). The console
lists the log and overlays the suppression flags.

Fails open exactly like `instrumentation/store.py`: a dead database loses
alert history, never a notification. The notification is sent first and
recorded second, so the record is a side effect of alerting and not a
precondition for it.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import psycopg

from app.config import Settings, get_settings
from app.observability import get_logger, log_event
from monitoring.schemas import Alert

logger = get_logger("agentforge.monitoring.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id          TEXT PRIMARY KEY,
    rule        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '',
    value       DOUBLE PRECISION NOT NULL DEFAULT 0,
    threshold   DOUBLE PRECISION NOT NULL DEFAULT 0,
    fingerprint TEXT NOT NULL,
    channels    JSONB NOT NULL DEFAULT '[]'::jsonb,
    fired_at    TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_fired_idx ON alerts (fired_at DESC);
CREATE INDEX IF NOT EXISTS alerts_fingerprint_idx ON alerts (fingerprint, fired_at DESC);
"""


def _row(alert: Alert, channels: list[str]) -> dict[str, Any]:
    return {
        "id": alert.id,
        "rule": alert.rule,
        "severity": alert.severity.value,
        "summary": alert.summary,
        "detail": alert.detail,
        "value": alert.value,
        "threshold": alert.threshold,
        "fingerprint": alert.fingerprint,
        "channels": channels,
        "fired_at": alert.fired_at,
    }


class AlertStore(Protocol):
    name: str

    async def ensure_schema(self) -> bool:
        """Create the table if absent; False means the backend is unreachable."""

    async def record(self, alert: Alert, channels: list[str]) -> None:
        """Append one fired alert and the channels that delivered it."""

    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest-first rows for the console list view."""


class InMemoryAlertStore:
    """Process-local fallback, so a dead Postgres costs history, not alerting."""

    name = "memory"

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def ensure_schema(self) -> bool:
        return True

    async def record(self, alert: Alert, channels: list[str]) -> None:
        self.rows.insert(0, _row(alert, channels))

    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.rows[:limit]


class PostgresAlertStore:
    name = "postgres"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._warned = False

    def _note(self, op: str, exc: Exception) -> None:
        if not self._warned:
            self._warned = True
            log_event(logger, "monitoring.postgres_unavailable", op=op, error=str(exc))

    async def _connect(self, autocommit: bool = True):
        return await psycopg.AsyncConnection.connect(
            self.settings.postgres_dsn, autocommit=autocommit
        )

    async def ensure_schema(self) -> bool:
        try:
            async with await self._connect() as conn:
                await conn.execute(SCHEMA)
            return True
        except (psycopg.Error, OSError) as exc:
            self._note("ensure_schema", exc)
            return False

    async def record(self, alert: Alert, channels: list[str]) -> None:
        row = _row(alert, channels)
        try:
            async with await self._connect() as conn:
                await conn.execute(
                    """
                    INSERT INTO alerts (id, rule, severity, summary, detail, value,
                                        threshold, fingerprint, channels, fired_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        row["id"], row["rule"], row["severity"], row["summary"],
                        row["detail"], row["value"], row["threshold"],
                        row["fingerprint"], json.dumps(channels), row["fired_at"],
                    ),
                )
        except (psycopg.Error, OSError) as exc:
            self._note("record", exc)

    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            async with await self._connect(autocommit=False) as conn:
                cur = await conn.execute(
                    """
                    SELECT id, rule, severity, summary, detail, value, threshold,
                           fingerprint, channels, fired_at
                    FROM alerts ORDER BY fired_at DESC LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("recent", exc)
            return []
        keys = (
            "id", "rule", "severity", "summary", "detail", "value",
            "threshold", "fingerprint", "channels", "fired_at",
        )
        return [{k: v for k, v in zip(keys, r)} for r in rows]


async def build_alert_store(settings: Settings | None = None) -> AlertStore:
    """Postgres when it answers, in-memory otherwise."""
    store = PostgresAlertStore(settings)
    if await store.ensure_schema():
        log_event(logger, "monitoring.alert_store", backend=store.name)
        return store
    log_event(
        logger, "monitoring.alert_store", backend="memory", reason="postgres down"
    )
    return InMemoryAlertStore()
