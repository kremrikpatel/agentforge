"""Alert suppression: dedup window, silence, and acknowledge.

Two backends behind one protocol. In-memory is the local/test default;
Redis is what a deployed install uses, because suppression state that lives
in one process is worthless once the sweep runs as a CronJob (a fresh pod
per run, so the dedup window resets every time) and the web console
acknowledges alerts from a *different* pod.

The Redis implementation leans on `SET NX EX` rather than read-then-write:
that one call is the dedup decision, and it stays correct when the CronJob
and the console race each other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from memory.redis_conn import RedisBacked

# Suppression is keyed by fingerprint, so an alert that can fire per-action
# (circuit_breaker_trips) silences per-action rather than for the whole rule.
_FIRED = "alert:fired:{}"
_SILENCE = "alert:silence:{}"
_ACK = "alert:ack:{}"


class AlertState(Protocol):
    async def should_notify(self, fingerprint: str) -> bool:
        """False when acknowledged, silenced, or still inside the dedup window."""

    async def silence(self, fingerprint: str, duration_s: float) -> None:
        """Suppress notifications for this fingerprint until the window elapses."""

    async def acknowledge(self, fingerprint: str) -> None:
        """Suppress indefinitely, until `clear`."""

    async def clear(self, fingerprint: str) -> None:
        """Drop all suppression for this fingerprint."""

    async def status(self, fingerprint: str) -> dict[str, bool]:
        """Current suppression flags, for the console to render."""


@dataclass
class _Entry:
    last_fired_at: float = 0.0
    silenced_until: float = 0.0
    acknowledged: bool = False


class InMemoryAlertState:
    """Process-local suppression.

    ponytail: single-process only -- state dies with the process and is not
    shared across replicas. That is exactly why `RedisAlertState` exists;
    this one stays for tests and for a local `monitoring.cli sweep`.
    """

    name = "memory"

    def __init__(self, dedup_window_s: float = 900.0) -> None:
        self.dedup_window_s = dedup_window_s
        self._state: dict[str, _Entry] = {}

    async def should_notify(self, fingerprint: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        entry = self._state.setdefault(fingerprint, _Entry())
        if entry.acknowledged or now < entry.silenced_until:
            return False
        if now - entry.last_fired_at < self.dedup_window_s:
            return False
        entry.last_fired_at = now
        return True

    async def silence(
        self, fingerprint: str, duration_s: float, *, now: float | None = None
    ) -> None:
        now = now if now is not None else time.time()
        self._state.setdefault(fingerprint, _Entry()).silenced_until = now + duration_s

    async def acknowledge(self, fingerprint: str) -> None:
        self._state.setdefault(fingerprint, _Entry()).acknowledged = True

    async def clear(self, fingerprint: str) -> None:
        self._state.pop(fingerprint, None)

    async def status(self, fingerprint: str, *, now: float | None = None) -> dict[str, bool]:
        now = now if now is not None else time.time()
        entry = self._state.get(fingerprint)
        if entry is None:
            return {"silenced": False, "acknowledged": False}
        return {
            "silenced": now < entry.silenced_until,
            "acknowledged": entry.acknowledged,
        }


class RedisAlertState(RedisBacked):
    """Shared suppression, so dedup survives a CronJob pod and an
    acknowledgement in the console actually suppresses the next sweep.

    Inherits `RedisBacked`'s fail-open `_safe`, and fails open *towards
    notifying*: if Redis is unreachable we would rather send a duplicate
    alert than silently drop a real one.
    """

    name = "redis"

    def __init__(
        self, dedup_window_s: float = 900.0, settings: Settings | None = None, client=None
    ) -> None:
        super().__init__(settings, client)
        self.dedup_window_s = dedup_window_s

    async def should_notify(self, fingerprint: str) -> bool:
        if any((await self.status(fingerprint)).values()):
            return False

        # SET NX EX is the dedup decision itself: true only for the caller that
        # claimed the window, so two pods evaluating at once notify once.
        claimed = await self._safe(
            "should_notify",
            lambda: self.client.set(
                _FIRED.format(fingerprint), "1", ex=int(self.dedup_window_s), nx=True
            ),
            True,  # Redis down -> notify; a duplicate beats a dropped alert.
        )
        return bool(claimed)

    async def silence(self, fingerprint: str, duration_s: float) -> None:
        await self._safe(
            "silence",
            lambda: self.client.set(
                _SILENCE.format(fingerprint), "1", ex=max(1, int(duration_s))
            ),
            None,
        )

    async def acknowledge(self, fingerprint: str) -> None:
        # No TTL: an acknowledgement holds until someone clears it.
        await self._safe(
            "acknowledge", lambda: self.client.set(_ACK.format(fingerprint), "1"), None
        )

    async def clear(self, fingerprint: str) -> None:
        await self._safe(
            "clear",
            lambda: self.client.delete(
                _FIRED.format(fingerprint),
                _SILENCE.format(fingerprint),
                _ACK.format(fingerprint),
            ),
            None,
        )

    async def status(self, fingerprint: str) -> dict[str, bool]:
        silenced = await self._safe(
            "status", lambda: self.client.exists(_SILENCE.format(fingerprint)), 0
        )
        acknowledged = await self._safe(
            "status", lambda: self.client.exists(_ACK.format(fingerprint)), 0
        )
        return {"silenced": bool(silenced), "acknowledged": bool(acknowledged)}


async def build_state(
    dedup_window_s: float, backend: str, settings: Settings | None = None
) -> AlertState:
    """Redis when configured and reachable, in-memory otherwise."""
    if backend == "redis":
        state = RedisAlertState(dedup_window_s, settings)
        if await state.ping():
            return state
    return InMemoryAlertState(dedup_window_s)
