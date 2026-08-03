"""Tier 1 memory: Redis session-scoped short-term memory.

Holds the running conversation for a session so a follow-up run can see what the
team already concluded. Capped and TTL'd -- STM is not an archive, that is LTM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from memory.redis_conn import RedisBacked

MAX_TURNS = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class SessionMemory(RedisBacked):
    @staticmethod
    def _key(session_id: str) -> str:
        return f"af:session:{session_id}"

    async def append(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        turn = json.dumps({"role": role, "content": content[:4000], "at": _now()})

        async def _write() -> None:
            pipe = self.client.pipeline()
            pipe.rpush(key, turn)
            pipe.ltrim(key, -MAX_TURNS, -1)
            pipe.expire(key, self.settings.session_ttl_s)
            await pipe.execute()

        await self._safe("session.append", _write, None)

    async def history(self, session_id: str, limit: int = 10) -> list[dict]:
        raw = await self._safe(
            "session.history",
            lambda: self.client.lrange(self._key(session_id), -limit, -1),
            [],
        )
        turns: list[dict] = []
        for item in raw:
            try:
                turns.append(json.loads(item))
            except (ValueError, TypeError):
                continue
        return turns

    async def as_prompt_context(self, session_id: str, limit: int = 6) -> str:
        turns = await self.history(session_id, limit)
        if not turns:
            return ""
        lines = [f"- [{t.get('role', '?')}] {t.get('content', '')[:400]}" for t in turns]
        return "Earlier in this session:\n" + "\n".join(lines)

    async def clear(self, session_id: str) -> None:
        await self._safe("session.clear", lambda: self.client.delete(self._key(session_id)), 0)
