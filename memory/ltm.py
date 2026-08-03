"""Tier 2 memory: durable long-term memory in Postgres + pgvector.

Every finished run is embedded and stored, so later runs can recall what the
team decided about a similar topic. Like the other tiers this fails open: a
Postgres outage costs recall, not availability.
"""

from __future__ import annotations

import uuid

import psycopg

from app.config import Settings, get_settings
from app.observability import get_logger, log_event
from memory.embedding import embed

logger = get_logger("agentforge.memory.ltm")

# Mirrors db/init.sql so a bare database still works without the compose init.
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS ltm_records (
    id          UUID PRIMARY KEY,
    session_id  TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    topic       TEXT NOT NULL,
    stage       TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(%(dim)s),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ltm_records_session_idx ON ltm_records (session_id);
CREATE INDEX IF NOT EXISTS ltm_records_embedding_idx
    ON ltm_records USING hnsw (embedding vector_cosine_ops);
"""


def _vector_literal(values: list[float]) -> str:
    """pgvector's text input format; cast with ::vector at the call site."""
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


class LongTermMemory:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._degraded = False
        self._warned = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _note_failure(self, op: str, exc: Exception) -> None:
        self._degraded = True
        if not self._warned:
            self._warned = True
            log_event(logger, "memory.postgres_unavailable", op=op, error=str(exc))

    async def ensure_schema(self) -> bool:
        try:
            async with await psycopg.AsyncConnection.connect(
                self.settings.postgres_dsn, autocommit=True
            ) as conn:
                await conn.execute(SCHEMA % {"dim": self.settings.embedding_dim})
        except (psycopg.Error, OSError) as exc:
            self._note_failure("ensure_schema", exc)
            return False
        self._degraded = False
        return True

    async def remember(
        self, *, session_id: str, run_id: str, topic: str, stage: str, content: str
    ) -> bool:
        vector = _vector_literal(embed(content or topic, self.settings.embedding_dim))
        try:
            # ponytail: connection per call. Fine at Phase 1 volume; add
            # psycopg_pool.AsyncConnectionPool when request rate justifies it.
            async with await psycopg.AsyncConnection.connect(
                self.settings.postgres_dsn, autocommit=True
            ) as conn:
                await conn.execute(
                    """
                    INSERT INTO ltm_records
                        (id, session_id, run_id, topic, stage, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (uuid.uuid4(), session_id, run_id, topic, stage, content[:20000], vector),
                )
        except (psycopg.Error, OSError) as exc:
            self._note_failure("remember", exc)
            return False
        self._degraded = False
        return True

    async def recall(self, query: str, k: int = 5) -> list[dict]:
        vector = _vector_literal(embed(query, self.settings.embedding_dim))
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    """
                    SELECT run_id, topic, stage, content,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM ltm_records
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector, vector, k),
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note_failure("recall", exc)
            return []

        self._degraded = False
        return [
            {
                "run_id": r[0],
                "topic": r[1],
                "stage": r[2],
                "content": r[3],
                "similarity": float(r[4]),
            }
            for r in rows
        ]

    async def as_prompt_context(self, query: str, k: int = 3, floor: float = 0.6) -> str:
        hits = [h for h in await self.recall(query, k) if h["similarity"] >= floor]
        if not hits:
            return ""
        lines = [
            f"- ({h['stage']}, similarity {h['similarity']:.2f}) {h['content'][:300]}"
            for h in hits
        ]
        return "Relevant conclusions from earlier runs:\n" + "\n".join(lines)
