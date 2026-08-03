"""The sparse half of hybrid search.

Two interchangeable backends behind one protocol:

- InMemoryBM25 -- real Okapi BM25, no server. The default, and what the tests
  exercise, so hybrid ranking is verified against actual scoring math.
- PostgresFTS  -- tsvector + GIN, for corpora that outgrow a process. Note this
  is coverage-density ranking (ts_rank_cd), not literally BM25.

Both return (chunk_id, score); the vector store owns the payloads and hydrates.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, runtime_checkable

import psycopg

from app.config import Settings, get_settings
from app.observability import get_logger, log_event
from memory.embedding import tokenize  # same tokenizer as the dense side
from rag.schemas import Chunk

logger = get_logger("agentforge.rag.lexical")

# Okapi BM25 defaults from the TREC literature.
K1 = 1.5
B = 0.75

# New table, owned by the RAG subsystem. No Phase 1 table is touched.
SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id   TEXT PRIMARY KEY,
    kb_id      TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    text       TEXT NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rag_chunks_kb_idx ON rag_chunks (kb_id);
CREATE INDEX IF NOT EXISTS rag_chunks_tsv_idx ON rag_chunks USING gin (tsv);
"""


@runtime_checkable
class LexicalIndex(Protocol):
    async def index(self, kb_id: str, chunks: list[Chunk]) -> None: ...
    async def search(self, kb_id: str, query: str, limit: int) -> list[tuple[str, float]]: ...
    async def drop(self, kb_id: str) -> None: ...


class InMemoryBM25:
    """Okapi BM25 over an in-process corpus, partitioned by kb_id."""

    name = "in_memory_bm25"

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Counter]] = {}   # kb -> chunk_id -> term freqs
        self._lengths: dict[str, dict[str, int]] = {}    # kb -> chunk_id -> length
        self._df: dict[str, Counter] = {}                # kb -> term -> doc frequency

    async def index(self, kb_id: str, chunks: list[Chunk]) -> None:
        docs = self._docs.setdefault(kb_id, {})
        lengths = self._lengths.setdefault(kb_id, {})
        df = self._df.setdefault(kb_id, Counter())

        for chunk in chunks:
            # Re-ingesting the same chunk id must not double-count its terms.
            if chunk.id in docs:
                for term in docs[chunk.id]:
                    df[term] -= 1
                    if df[term] <= 0:
                        del df[term]
            terms = Counter(tokenize(f"{chunk.title} {chunk.text}"))
            docs[chunk.id] = terms
            lengths[chunk.id] = sum(terms.values())
            df.update(terms.keys())

    async def search(self, kb_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        docs = self._docs.get(kb_id)
        if not docs:
            return []
        lengths = self._lengths[kb_id]
        df = self._df[kb_id]
        n = len(docs)
        avgdl = (sum(lengths.values()) / n) or 1.0

        query_terms = set(tokenize(query))
        scored: list[tuple[str, float]] = []
        for chunk_id, terms in docs.items():
            dl = lengths[chunk_id] or 1
            total = 0.0
            for term in query_terms:
                tf = terms.get(term, 0)
                if not tf:
                    continue
                # Probabilistic IDF, +1 inside the log so it can never go negative.
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                total += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
            if total > 0:
                scored.append((chunk_id, total))

        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:limit]

    async def drop(self, kb_id: str) -> None:
        self._docs.pop(kb_id, None)
        self._lengths.pop(kb_id, None)
        self._df.pop(kb_id, None)


class PostgresFTS:
    """tsvector-backed lexical search.

    Fails open: a Postgres outage costs the sparse half of the hybrid, not the
    request.
    """

    name = "postgres_fts"

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
            log_event(logger, "rag.postgres_unavailable", op=op, error=str(exc))

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

    async def index(self, kb_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        try:
            async with await psycopg.AsyncConnection.connect(
                self.settings.postgres_dsn, autocommit=True
            ) as conn:
                await conn.cursor().executemany(
                    """
                    INSERT INTO rag_chunks (chunk_id, kb_id, doc_id, text)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text
                    """,
                    [(c.id, kb_id, c.doc_id, f"{c.title}\n{c.text}".strip()) for c in chunks],
                )
        except (psycopg.Error, OSError) as exc:
            self._note("index", exc)

    async def search(self, kb_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        try:
            async with await psycopg.AsyncConnection.connect(self.settings.postgres_dsn) as conn:
                cur = await conn.execute(
                    """
                    SELECT chunk_id, ts_rank_cd(tsv, q) AS score
                    FROM rag_chunks, websearch_to_tsquery('english', %s) AS q
                    WHERE kb_id = %s AND tsv @@ q
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, kb_id, limit),
                )
                rows = await cur.fetchall()
        except (psycopg.Error, OSError) as exc:
            self._note("search", exc)
            return []
        self._degraded = False
        return [(r[0], float(r[1])) for r in rows]

    async def drop(self, kb_id: str) -> None:
        try:
            async with await psycopg.AsyncConnection.connect(
                self.settings.postgres_dsn, autocommit=True
            ) as conn:
                await conn.execute("DELETE FROM rag_chunks WHERE kb_id = %s", (kb_id,))
        except (psycopg.Error, OSError) as exc:
            self._note("drop", exc)


async def build_lexical_index(settings: Settings | None = None) -> LexicalIndex:
    """Postgres when it answers, in-process BM25 otherwise."""
    fts = PostgresFTS(settings)
    if await fts.ensure_schema():
        log_event(logger, "rag.lexical_backend", backend=fts.name)
        return fts
    log_event(logger, "rag.lexical_backend", backend=InMemoryBM25.name)
    return InMemoryBM25()
