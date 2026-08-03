"""Semantic cache, session memory, and the embedder they share."""

from __future__ import annotations

import dataclasses

from memory.cache import SemanticCache
from memory.embedding import cache_key, cosine, embed, normalize_text
from memory.stm import SessionMemory

TOPIC = "Design a rate limiter for a public REST API"
REPORT = {"run_id": "r1", "status": "completed", "topic": TOPIC}


def cache(settings, fake_redis, **overrides):
    return SemanticCache(dataclasses.replace(settings, **overrides), client=fake_redis)


# --- embedder ---------------------------------------------------------------


def test_embedding_is_deterministic_and_unit_length():
    a, b = embed(TOPIC), embed(TOPIC)
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_cosine_ranks_related_text_above_unrelated():
    base = embed(TOPIC)
    related = embed("Design a rate limiter for a public REST API service")
    unrelated = embed("Bake a sourdough loaf with a wet starter")
    assert cosine(base, related) > cosine(base, unrelated)


def test_cache_key_ignores_casing_and_whitespace():
    assert cache_key("  Rate   Limiter ") == cache_key("rate limiter")
    assert normalize_text("  A  B ") == "a b"


# --- cache ------------------------------------------------------------------


async def test_miss_on_an_empty_cache(settings, fake_redis):
    payload, kind, _ = await cache(settings, fake_redis).get(TOPIC)
    assert payload is None
    assert kind == ""


async def test_identical_request_is_an_exact_hit(settings, fake_redis):
    """Acceptance: repeating a request inside the TTL skips the pipeline."""
    c = cache(settings, fake_redis)
    await c.set(TOPIC, REPORT)

    payload, kind, similarity = await c.get(TOPIC)

    assert payload == REPORT
    assert kind == "exact"
    assert similarity == 1.0


async def test_casing_and_spacing_differences_still_hit_exactly(settings, fake_redis):
    c = cache(settings, fake_redis)
    await c.set(TOPIC, REPORT)

    payload, kind, _ = await c.get("  design a RATE limiter for a Public REST API  ")

    assert payload == REPORT
    assert kind == "exact"


async def test_near_identical_request_is_a_semantic_hit(settings, fake_redis):
    c = cache(settings, fake_redis, cache_similarity_threshold=0.8)
    await c.set(TOPIC, REPORT)

    payload, kind, similarity = await c.get(
        "Design a rate limiter for a public REST API endpoint"
    )

    assert payload == REPORT
    assert kind == "semantic"
    assert similarity >= 0.8


async def test_unrelated_request_misses(settings, fake_redis):
    c = cache(settings, fake_redis)
    await c.set(TOPIC, REPORT)

    payload, kind, _ = await c.get("Write a haiku about winter gardening")

    assert payload is None
    assert kind == ""


async def test_expired_payload_is_treated_as_a_miss(settings, fake_redis):
    """The index outlives entries; a dangling pointer must not resurrect one."""
    c = cache(settings, fake_redis)
    await c.set(TOPIC, REPORT)
    fake_redis.store.clear()  # simulate TTL expiry, index untouched

    payload, kind, _ = await c.get(TOPIC)

    assert payload is None
    assert kind == ""


async def test_invalidate_removes_the_entry(settings, fake_redis):
    c = cache(settings, fake_redis)
    await c.set(TOPIC, REPORT)
    await c.invalidate(TOPIC)

    payload, _, _ = await c.get(TOPIC)
    assert payload is None


async def test_cache_fails_open_when_redis_is_down(settings):
    class Broken:
        def __getattr__(self, _name):
            raise OSError("connection refused")

    c = SemanticCache(settings, client=Broken())

    payload, kind, _ = await c.get(TOPIC)
    await c.set(TOPIC, REPORT)  # must not raise

    assert payload is None and kind == ""
    assert c.degraded


# --- session memory ---------------------------------------------------------


async def test_session_history_round_trips(settings, fake_redis):
    stm = SessionMemory(settings, client=fake_redis)
    await stm.append("s1", "analysis", "Ana framed the problem.")
    await stm.append("s1", "develop", "Dev proposed a token bucket.")

    turns = await stm.history("s1")

    assert [t["role"] for t in turns] == ["analysis", "develop"]
    assert "token bucket" in turns[1]["content"]


async def test_session_context_is_empty_for_an_unknown_session(settings, fake_redis):
    stm = SessionMemory(settings, client=fake_redis)
    assert await stm.as_prompt_context("nobody") == ""
