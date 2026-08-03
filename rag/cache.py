"""Caches for the two expensive stages of the funnel.

Embeddings and rerank scores are pure functions of their inputs, so they cache
perfectly. Reuses Phase 1's RedisBacked, which already fails open -- a cache
outage costs latency, never correctness.

With the current local hashing embedder the embedding cache saves little; it
earns its keep the moment `embed()` becomes a network call. The rerank cache
pays for itself immediately, because a cross-encoder is the slowest thing here.
"""

from __future__ import annotations

import json

from memory.embedding import cache_key
from memory.redis_conn import RedisBacked
from rag.config import RagSettings, get_rag_settings

_EMB = "af:rag:emb:"
_RR = "af:rag:rr:"


class RagCache(RedisBacked):
    def __init__(self, settings=None, rag_settings: RagSettings | None = None, client=None):
        super().__init__(settings, client)
        self.rag = rag_settings or get_rag_settings()

    @property
    def enabled(self) -> bool:
        return self.rag.cache_enabled

    # --- embeddings --------------------------------------------------------

    async def get_embedding(self, text: str) -> list[float] | None:
        if not self.enabled:
            return None
        raw = await self._safe(
            "rag.emb_get", lambda: self.client.get(_EMB + cache_key(text)), None
        )
        if not raw:
            return None
        try:
            vec = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return vec if isinstance(vec, list) else None

    async def set_embedding(self, text: str, vector: list[float]) -> None:
        if not self.enabled:
            return
        await self._safe(
            "rag.emb_set",
            lambda: self.client.setex(
                _EMB + cache_key(text), self.rag.cache_ttl_s, json.dumps(vector)
            ),
            None,
        )

    # --- rerank scores -----------------------------------------------------

    @staticmethod
    def _rr_key(query: str, chunk_id: str) -> str:
        return f"{_RR}{cache_key(query)}:{chunk_id}"

    async def get_rerank_scores(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """One MGET for the whole candidate set, not one round-trip per chunk."""
        if not self.enabled or not chunk_ids:
            return {}
        keys = [self._rr_key(query, cid) for cid in chunk_ids]
        raw = await self._safe("rag.rr_get", lambda: self.client.mget(keys), [])
        hits: dict[str, float] = {}
        for cid, value in zip(chunk_ids, raw or []):
            if value is None:
                continue
            try:
                hits[cid] = float(value)
            except (TypeError, ValueError):
                continue
        return hits

    async def set_rerank_scores(self, query: str, scores: dict[str, float]) -> None:
        if not self.enabled or not scores:
            return

        async def _write() -> None:
            pipe = self.client.pipeline()
            for cid, score in scores.items():
                pipe.setex(self._rr_key(query, cid), self.rag.cache_ttl_s, str(score))
            await pipe.execute()

        await self._safe("rag.rr_set", _write, None)
