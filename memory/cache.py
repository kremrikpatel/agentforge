"""Tier 3 memory: semantic cache that short-circuits repeated work.

Two lookups. Exact: hash of the normalised topic, so casing and whitespace do
not cause a miss. Semantic: cosine over a small recent-topic index, so a
rephrased request still hits. A hit returns the stored report and no LLM call
is made at all.
"""

from __future__ import annotations

import json

from app.observability import get_logger, log_event
from memory.embedding import cache_key, cosine, embed
from memory.redis_conn import RedisBacked

logger = get_logger("agentforge.memory.cache")

_PREFIX = "af:cache:v1:"
_INDEX = "af:cache:index"


class SemanticCache(RedisBacked):
    async def get(self, topic: str) -> tuple[dict | None, str, float]:
        """Return (payload, hit_kind, similarity). hit_kind is '' on a miss."""
        key = cache_key(topic)

        exact = await self._safe("cache.get", lambda: self.client.get(_PREFIX + key), None)
        if exact:
            payload = _loads(exact)
            if payload is not None:
                log_event(logger, "cache.hit", kind="exact", key=key[:12])
                return payload, "exact", 1.0

        threshold = self.settings.cache_similarity_threshold
        query_vec = embed(topic, self.settings.embedding_dim)
        raw_index = await self._safe("cache.index", lambda: self.client.lrange(_INDEX, 0, -1), [])

        scored: list[tuple[float, str]] = []
        for item in raw_index:
            entry = _loads(item)
            if not entry:
                continue
            score = cosine(query_vec, entry.get("vec", []))
            if score >= threshold:
                scored.append((score, entry["key"]))

        # Best first; entries whose payload already expired are simply skipped.
        for score, entry_key in sorted(scored, key=lambda s: s[0], reverse=True):
            stored = await self._safe(
                "cache.get_semantic", lambda k=entry_key: self.client.get(_PREFIX + k), None
            )
            payload = _loads(stored) if stored else None
            if payload is not None:
                log_event(
                    logger, "cache.hit", kind="semantic", key=entry_key[:12], similarity=score
                )
                return payload, "semantic", round(score, 4)

        return None, "", 0.0

    async def set(self, topic: str, payload: dict) -> None:
        key = cache_key(topic)
        entry = json.dumps(
            {
                "key": key,
                "topic": topic[:200],
                "vec": embed(topic, self.settings.embedding_dim),
            }
        )
        ttl = self.settings.cache_ttl_s

        async def _write() -> None:
            pipe = self.client.pipeline()
            pipe.setex(_PREFIX + key, ttl, json.dumps(payload, default=str))
            pipe.lpush(_INDEX, entry)
            # ponytail: linear scan over a capped list. Swap for Redis vector
            # search (FT.SEARCH) when the index outgrows a few hundred entries.
            pipe.ltrim(_INDEX, 0, self.settings.cache_index_size - 1)
            pipe.expire(_INDEX, ttl * 4)
            await pipe.execute()

        await self._safe("cache.set", _write, None)

    async def invalidate(self, topic: str) -> None:
        key = cache_key(topic)
        await self._safe("cache.invalidate", lambda: self.client.delete(_PREFIX + key), 0)


def _loads(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None
