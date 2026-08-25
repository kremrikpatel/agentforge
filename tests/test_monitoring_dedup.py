"""Flapping-alert dedup/rate-limit and silence/acknowledge.

Runs the same behavioural contract against both backends: the in-memory one
with an injected clock, and the Redis one against a fake that implements the
`SET NX EX` semantics the real client provides. The point of testing both is
that only the Redis path is correct in a deployed install, so its behaviour
has to be pinned, not assumed.
"""

from __future__ import annotations

import pytest

from monitoring.dedup import InMemoryAlertState, RedisAlertState


class FakeRedis:
    """Enough of redis.asyncio for suppression keys: SET with NX/EX, EXISTS, DELETE.

    TTLs are stored as absolute deadlines against a clock the test drives, so
    expiry is deterministic rather than a sleep.
    """

    def __init__(self) -> None:
        self.store: dict[str, float | None] = {}   # key -> expiry (None = no TTL)
        self.now = 1_000_000.0

    def _live(self, key: str) -> bool:
        if key not in self.store:
            return False
        expiry = self.store[key]
        if expiry is not None and self.now >= expiry:
            del self.store[key]
            return False
        return True

    async def set(self, key, value, ex=None, nx=False):
        if nx and self._live(key):
            return None
        self.store[key] = self.now + ex if ex else None
        return True

    async def exists(self, key):
        return 1 if self._live(key) else 0

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
        return len(keys)

    async def ping(self):
        return True


@pytest.fixture
def redis_state() -> tuple[RedisAlertState, FakeRedis]:
    fake = FakeRedis()
    return RedisAlertState(dedup_window_s=900.0, client=fake), fake


# --- in-memory backend -------------------------------------------------------


async def test_first_fire_notifies():
    assert await InMemoryAlertState(dedup_window_s=900).should_notify("rule") is True


async def test_flapping_alert_is_deduped_within_the_window():
    state = InMemoryAlertState(dedup_window_s=900)
    now = 1_000_000.0

    assert await state.should_notify("rule", now=now) is True
    assert await state.should_notify("rule", now=now + 1) is False
    assert await state.should_notify("rule", now=now + 899) is False


async def test_alert_fires_again_after_the_window_elapses():
    state = InMemoryAlertState(dedup_window_s=900)
    now = 1_000_000.0

    assert await state.should_notify("rule", now=now) is True
    assert await state.should_notify("rule", now=now + 901) is True


async def test_silenced_alert_does_not_notify():
    state = InMemoryAlertState(dedup_window_s=1)
    now = 1_000_000.0
    await state.silence("rule", duration_s=3600, now=now)

    assert await state.should_notify("rule", now=now + 10) is False
    assert (await state.status("rule", now=now + 10))["silenced"] is True


async def test_silence_expires():
    state = InMemoryAlertState(dedup_window_s=1)
    now = 1_000_000.0
    await state.silence("rule", duration_s=10, now=now)

    assert await state.should_notify("rule", now=now + 20) is True


async def test_acknowledged_alert_never_re_notifies_until_cleared():
    state = InMemoryAlertState(dedup_window_s=1)
    now = 1_000_000.0
    await state.acknowledge("rule")

    assert await state.should_notify("rule", now=now) is False
    assert await state.should_notify("rule", now=now + 10_000) is False

    await state.clear("rule")
    assert await state.should_notify("rule", now=now + 10_001) is True


async def test_fingerprints_are_independent():
    state = InMemoryAlertState(dedup_window_s=900)
    now = 1_000_000.0
    await state.should_notify("rule_a", now=now)

    assert await state.should_notify("rule_b", now=now) is True


# --- Redis backend (the one a deployed install uses) -------------------------


async def test_redis_flapping_alert_is_deduped_within_the_window(redis_state):
    state, fake = redis_state

    assert await state.should_notify("rule") is True
    assert await state.should_notify("rule") is False
    fake.now += 899
    assert await state.should_notify("rule") is False


async def test_redis_alert_fires_again_after_the_window_elapses(redis_state):
    state, fake = redis_state

    assert await state.should_notify("rule") is True
    fake.now += 901
    assert await state.should_notify("rule") is True


async def test_redis_dedup_survives_a_new_process(redis_state):
    """The CronJob case: a fresh pod, the same Redis, still deduped."""
    state, fake = redis_state
    assert await state.should_notify("rule") is True

    fresh_pod = RedisAlertState(dedup_window_s=900.0, client=fake)
    assert await fresh_pod.should_notify("rule") is False


async def test_redis_silence_suppresses_then_expires(redis_state):
    state, fake = redis_state
    await state.silence("rule", duration_s=600)

    assert await state.should_notify("rule") is False
    assert (await state.status("rule"))["silenced"] is True

    fake.now += 601
    assert await state.should_notify("rule") is True


async def test_redis_acknowledge_holds_until_cleared(redis_state):
    state, fake = redis_state
    await state.acknowledge("rule")

    assert await state.should_notify("rule") is False
    fake.now += 100_000
    assert await state.should_notify("rule") is False, "an ack has no TTL"

    await state.clear("rule")
    assert await state.should_notify("rule") is True


async def test_acknowledgement_from_another_process_suppresses_the_sweep(redis_state):
    """The console and the sweep are different pods sharing one Redis."""
    console, fake = redis_state
    sweep = RedisAlertState(dedup_window_s=900.0, client=fake)

    await console.acknowledge("escalation_rate")

    assert await sweep.should_notify("escalation_rate") is False


async def test_redis_failure_fails_open_towards_notifying():
    """A duplicate alert beats a dropped one when Redis is unreachable."""

    class DeadRedis(FakeRedis):
        async def set(self, *a, **kw):
            raise OSError("connection refused")

        async def exists(self, key):
            raise OSError("connection refused")

    state = RedisAlertState(dedup_window_s=900.0, client=DeadRedis())

    assert await state.should_notify("rule") is True
