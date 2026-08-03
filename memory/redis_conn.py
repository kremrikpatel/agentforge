"""Shared Redis plumbing with fail-open degradation.

Memory is an optimisation, not a correctness requirement: if Redis is down the
pipeline must still answer, just without session history or cache hits. Every
call goes through `_safe`, which logs once and returns a default.
"""

from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.config import Settings, get_settings
from app.observability import get_logger, log_event

logger = get_logger("agentforge.memory")

T = TypeVar("T")


class RedisBacked:
    def __init__(self, settings: Settings | None = None, client=None) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._degraded = False
        self._warned = False

    @property
    def client(self):
        if self._client is None:
            self._client = aioredis.from_url(self.settings.redis_url, decode_responses=True)
        return self._client

    @property
    def degraded(self) -> bool:
        return self._degraded

    async def _safe(self, op: str, fn: Callable[[], Awaitable[T]], default: T) -> T:
        try:
            result = await fn()
        except (RedisError, OSError) as exc:
            self._degraded = True
            if not self._warned:
                self._warned = True
                log_event(logger, "memory.redis_unavailable", op=op, error=str(exc))
            return default
        self._degraded = False
        return result

    async def ping(self) -> bool:
        return await self._safe("ping", lambda: self.client.ping(), False)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
