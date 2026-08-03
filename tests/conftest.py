"""Shared fakes. Nothing here touches the network, Redis, or Postgres."""

from __future__ import annotations

import dataclasses
import json

import pytest

from app.config import Settings
from gateway.providers import BaseProvider, GatewayRequest, ProviderError


@pytest.fixture
def settings() -> Settings:
    """Baseline: no provider keys, fast retries, backends pointed nowhere."""
    return dataclasses.replace(
        Settings(),
        anthropic_api_key="",
        openai_api_key="",
        gemini_api_key="",
        groq_api_key="",
        allow_stub_provider=False,
        llm_retries=1,
        guardrails_llm_tier=False,
        guardrails_block_threshold=0.6,
        cache_similarity_threshold=0.85,
    )


class FakeProvider(BaseProvider):
    """Scripted provider. `script` is consumed one entry per attempt.

    Entries: a str (returned), or a ProviderError (raised).
    """

    def __init__(self, name: str, settings: Settings, script: list, key: str = "key") -> None:
        super().__init__(settings)
        self.name = name
        self.script = list(script)
        self._key = key
        self.calls = 0

    @property
    def api_key(self) -> str:
        return self._key

    @property
    def model(self) -> str:
        return f"{self.name}-model"

    async def complete(self, req: GatewayRequest, client) -> str:
        self.calls += 1
        step = self.script.pop(0) if self.script else "default"
        if isinstance(step, Exception):
            raise step
        return step


def retryable(name: str) -> ProviderError:
    return ProviderError(f"{name}: rate limited", retryable=True)


def fatal(name: str) -> ProviderError:
    return ProviderError(f"{name}: HTTP 401 invalid api key", retryable=False)


class FakeRedis:
    """Just enough async Redis for the cache and session tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, _ttl, value):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)
        self.lists.pop(key, None)
        return 1

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    async def ltrim(self, key, start, end):
        items = self.lists.get(key, [])
        self.lists[key] = items[start : (end + 1 if end >= 0 else len(items) + end + 1)]

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start : end + 1]

    async def expire(self, key, ttl):
        return True

    async def ping(self):
        return True

    async def aclose(self):
        return None

    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    """Queues coroutines and runs them on execute(), like redis-py's pipeline."""

    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.ops: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        def _queue(*args):
            self.ops.append((name, args))
            return self

        return _queue

    async def execute(self):
        results = []
        for name, args in self.ops:
            results.append(await getattr(self.redis, name)(*args))
        self.ops.clear()
        return results


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


def contract_json(stage_value: str, **overrides) -> str:
    """Minimal valid contract for a stage, with targeted overrides."""
    from agents.contracts import Stage
    from agents.nodes import _stub_contract

    payload = json.loads(_stub_contract(Stage(stage_value), "test topic", None))
    for key, value in overrides.items():
        if key == "blocking":
            payload["handoff"]["blocking"] = value
        else:
            payload[key] = value
    return json.dumps(payload)
