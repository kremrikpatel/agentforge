"""Persistence and cross-run trends.

Trend arithmetic is a pure function over two summaries, so it is testable
without a database. Two stores implement the same protocol: Postgres for real
runs, in-memory so the trend path can be exercised offline.

Fails open on write, like the rest of the platform -- losing a row must not fail
the security check that produced it. Reads return empty rather than raising.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import psycopg

from app.config import Settings, get_settings
from app.observability import get_logger, log_event
from redteam.schemas import (
    AttackCategory,
    CategorySummary,
    CategoryTrend,
    RunSummary,
    TrendReport,
)

logger = get_logger("agentforge.redteam.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS redteam_runs (
    run_id      TEXT PRIMARY KEY,
    target      TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    threshold   DOUBLE PRECISION NOT NULL DEFAULT 0.9,
    total       INTEGER NOT NULL DEFAULT 0,
    leaked      INTEGER NOT NULL DEFAULT 0,
    block_rate  DOUBLE PRECISION NOT NULL DEFAULT 0,
    categories  JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS redteam_runs_started_idx ON redteam_runs (target, started_at DESC);

CREATE TABLE IF NOT EXISTS redteam_attempts (
    id               TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES redteam_runs(run_id) ON DELETE CASCADE,
    category         TEXT NOT NULL,
    objective        TEXT NOT NULL,
    prompt           TEXT NOT NULL DEFAULT '',
    response         TEXT NOT NULL DEFAULT '',
    outcome          TEXT NOT NULL,
    guardrail_action TEXT NOT NULL DEFAULT '',
    guardrail_risk   DOUBLE PRECISION NOT NULL DEFAULT 0,
    judge_leaked     BOOLEAN,
    judge_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    judge_evidence   TEXT NOT NULL DEFAULT '',
    turns            INTEGER NOT NULL DEFAULT 1,
    latency_ms       DOUBLE PRECISION NOT NULL DEFAULT 0,
    error            TEXT NOT NULL DEFAULT '',
    at               TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS redteam_attempts_run_idx ON redteam_attempts (run_id);
CREATE INDEX IF NOT EXISTS redteam_attempts_outcome_idx ON redteam_attempts (outcome);
"""


# --------------------------------------------------------------------------
# Trend arithmetic -- pure, no store involved
# --------------------------------------------------------------------------


def rates_from_rows(rows: list[dict]) -> dict[str, float]:
    """Extract {category: block_rate} from a stored `categories` JSONB payload."""
    out: dict[str, float] = {}
    for row in rows or []:
        category = row.get("category")
        if category:
            out[category] = float(row.get("block_rate", 0.0))
    return out


def compute_trend(
    current: RunSummary,
    previous_rates: dict[str, float] | None,
    previous_run_id: str | None,
) -> TrendReport:
    """Per-category block-rate delta against the previous run.

    A category absent from the previous run reports `new` rather than a bogus
    delta against zero.
    """
    previous_rates = previous_rates or {}
    report = TrendReport(run_id=current.run_id, previous_run_id=previous_run_id)
    for summary in current.categories:
        report.categories.append(
            CategoryTrend(
                category=summary.category,
                current=round(summary.block_rate, 4),
                previous=previous_rates.get(summary.category.value),
            )
        )
    return report


# --------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------


@runtime_checkable
class RunStore(Protocol):
    async def ensure_schema(self) -> bool: ...
    async def save(self, summary: RunSummary) -> bool: ...
    async def previous_run(self, target: str, before_run_id: str) -> tuple[str | None, dict]: ...
    async def recent_runs(self, limit: int = 20) -> list[dict]: ...
    async def attempts_for(self, run_id: str) -> list[dict]: ...


class InMemoryRunStore:
    """Offline store. Same protocol, so the trend path is exercisable with no DB."""

    name = "memory"

    def __init__(self) -> None:
        self.runs: list[dict] = []
        self.attempts: dict[str, list[dict]] = {}

    async def ensure_schema(self) -> bool:
        return True

    async def save(self, summary: RunSummary) -> bool:
        self.runs.insert(
            0,
            {
                "run_id": summary.run_id,
                "target": summary.target,
                "started_at": summary.started_at,
                "finished_at": summary.finished_at,
                "threshold": summary.threshold,
                "total": summary.total,
                "leaked": summary.leaked,
                "block_rate": round(summary.block_rate, 4),
                "categories": [c.as_row() for c in summary.categories],
            },
        )
        self.attempts[summary.run_id] = [a.model_dump(mode="json") for a in summary.attempts]
        return True

    async def previous_run(self, target: str, before_run_id: str) -> tuple[str | None, dict]:
        for run in self.runs:
            if run["target"] == target and run["run_id"] != before_run_id:
                return run["run_id"], rates_from_rows(run["categories"])
        return None, {}

    async def recent_runs(self, limit: int = 20) -> list[dict]:
        return self.runs[:limit]

    async def attempts_for(self, run_id: str) -> list[dict]:
        return self.attempts.get(run_id, [])


class PostgresRunStore:
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
            log_event(logger, "redteam.postgres_unavailable", op=op, error=str(exc))

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

    async def save(self, summary: RunSummary) -> bool:
        try:
            async with await psycopg.AsyncConnection.connect(
                self.settings.postgres_dsn, autocommit=True
            ) as conn:
                await conn.execute(
                    """
                    INSERT INTO redteam_runs
                        (run_id, target, started_at, finished_at, threshold,
                         total, leaked, block_rate, categories)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (run_id) DO UPDATE SET
                        finished_at = EXCLUDED.finished_at,
                        total = EXCLUDED.total,
                        leaked = EXCLUDED.leaked,
                        block_rate = EXCLUDED.block_rate,
                        categories = EXCLUDED.categories
                    """,
                    (
                        summary.run_id,
                        summary.target,
                        summary.started_at,
                        summary.finished_at or None,
                        summary.threshold,
                        summary.total,
                        summary.leaked,
                        round(summary.block_rate, 4),
                        json.dumps([c.as_row() for c in summary.categories]),
                    ),
                )
                await conn.cursor().executemany(
                    """
                    INSERT INTO redteam_attempts
                        (id, run_id, category, objective, prompt, response, outcome,
                         guardrail_action, guardrail_risk, judge_leaked, judge_confidence,
                         judge_evidence, turns, latency_ms, error, at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    [
                        (
                            a.id, summary.run_id, a.category.value, a.objective,
                            a.prompt, a.response, a.outcome.value, a.guardrail_action,
                            a.guardrail_risk, a.judge_leaked, a.judge_confidence,
                            a.judge_evidence, a.turns, a.latency_ms, a.error, a.at,
                        )
                        for a in summary.attempts
                    ],
                )
        except (psycopg.Error, OSError) as exc:
            self._note("save", exc)
            return False
        self._degraded = False
        return True

    async def previous_run(self, target: str, before_run_id: str) -> tuple[str | None, dict]:
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    """
                    SELECT run_id, categories FROM redteam_runs
                    WHERE target = %s AND run_id <> %s
                    ORDER BY started_at DESC LIMIT 1
                    """,
                    (target, before_run_id),
                )
                row = await cur.fetchone()
        except (psycopg.Error, OSError) as exc:
            self._note("previous_run", exc)
            return None, {}
        return (row[0], rates_from_rows(row[1])) if row else (None, {})

    async def recent_runs(self, limit: int = 20) -> list[dict]:
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    """
                    SELECT run_id, target, started_at, finished_at, threshold,
                           total, leaked, block_rate, categories
                    FROM redteam_runs ORDER BY started_at DESC LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("recent_runs", exc)
            return []
        keys = (
            "run_id", "target", "started_at", "finished_at", "threshold",
            "total", "leaked", "block_rate", "categories",
        )
        return [{k: v for k, v in zip(keys, r)} for r in rows]

    async def attempts_for(self, run_id: str) -> list[dict]:
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    """
                    SELECT id, category, objective, prompt, response, outcome,
                           guardrail_action, guardrail_risk, judge_confidence,
                           judge_evidence, turns, error, at
                    FROM redteam_attempts WHERE run_id = %s ORDER BY category, at
                    """,
                    (run_id,),
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("attempts_for", exc)
            return []
        keys = (
            "id", "category", "objective", "prompt", "response", "outcome",
            "guardrail_action", "guardrail_risk", "judge_confidence",
            "judge_evidence", "turns", "error", "at",
        )
        return [{k: v for k, v in zip(keys, r)} for r in rows]


async def build_store(settings: Settings | None = None, persist: bool = True) -> RunStore:
    """Postgres when it answers, in-memory otherwise."""
    if not persist:
        return InMemoryRunStore()
    store = PostgresRunStore(settings)
    if await store.ensure_schema():
        log_event(logger, "redteam.store", backend=store.name)
        return store
    log_event(logger, "redteam.store", backend=InMemoryRunStore.name, reason="postgres down")
    return InMemoryRunStore()


def summarize_categories(rows: list[dict]) -> list[CategorySummary]:
    """Rehydrate stored category rows for the dashboard."""
    out: list[CategorySummary] = []
    for row in rows or []:
        try:
            out.append(
                CategorySummary(
                    category=AttackCategory(row["category"]),
                    total=int(row.get("total", 0)),
                    blocked=int(row.get("blocked", 0)),
                    refused=int(row.get("refused", 0)),
                    leaked=int(row.get("leaked", 0)),
                    errors=int(row.get("errors", 0)),
                )
            )
        except (KeyError, ValueError):
            continue
    return out
