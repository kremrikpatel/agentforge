"""Fallback chain and retry policy."""

from __future__ import annotations

import dataclasses

import pytest

from app.config import Settings
from gateway.client import AllProvidersFailed, LLMGateway
from gateway.providers import DEFAULT_CHAIN, GatewayRequest, StubProvider, build_chain
from tests.conftest import FakeProvider, fatal, retryable

REQ = GatewayRequest(system="s", user="u")


def gw(settings, providers):
    # backoff_base_s=0 keeps retry tests instant.
    return LLMGateway(settings, providers=providers, client=object(), backoff_base_s=0.0)


def test_chain_order_is_claude_then_gpt_then_gemini_then_groq(settings):
    names = [p.name for p in build_chain(settings)]
    assert names == ["anthropic", "openai", "gemini", "groq", "stub"]
    assert DEFAULT_CHAIN[-1] is StubProvider


async def test_falls_through_when_primary_key_is_unset(settings):
    """Acceptance: env var unset -> primary skipped entirely, no request made."""
    primary = FakeProvider("anthropic", settings, ["never"], key="")
    secondary = FakeProvider("openai", settings, ["from gpt"])

    resp = await gw(settings, [primary, secondary]).complete(REQ)

    assert resp.provider == "openai"
    assert resp.text == "from gpt"
    assert resp.fallback_depth == 1
    assert primary.calls == 0, "unconfigured provider must not be dialled"


async def test_falls_through_when_primary_key_is_invalid(settings):
    """Acceptance: invalid key -> one 401, no retries, next provider answers."""
    primary = FakeProvider("anthropic", settings, [fatal("anthropic")])
    secondary = FakeProvider("openai", settings, ["from gpt"])

    resp = await gw(settings, [primary, secondary]).complete(REQ)

    assert resp.provider == "openai"
    assert primary.calls == 1, "a fatal 401 must not be retried"


async def test_retries_retryable_failure_on_same_provider(settings):
    primary = FakeProvider("anthropic", settings, [retryable("anthropic"), "recovered"])

    resp = await gw(settings, [primary]).complete(REQ)

    assert resp.provider == "anthropic"
    assert resp.text == "recovered"
    assert resp.attempts == 2


async def test_exhausts_retries_then_moves_to_next_provider(settings):
    # llm_retries=1 -> 2 attempts per provider.
    primary = FakeProvider("anthropic", settings, [retryable("a"), retryable("a")])
    secondary = FakeProvider("openai", settings, ["from gpt"])

    resp = await gw(settings, [primary, secondary]).complete(REQ)

    assert primary.calls == 2
    assert resp.provider == "openai"


async def test_walks_the_whole_chain_to_the_last_provider(settings):
    chain = [
        FakeProvider("anthropic", settings, [fatal("anthropic")]),
        FakeProvider("openai", settings, [fatal("openai")]),
        FakeProvider("gemini", settings, [fatal("gemini")]),
        FakeProvider("groq", settings, ["from groq"]),
    ]

    resp = await gw(settings, chain).complete(REQ)

    assert resp.provider == "groq"
    assert resp.fallback_depth == 3


async def test_raises_when_every_provider_fails(settings):
    chain = [
        FakeProvider("anthropic", settings, [fatal("anthropic")]),
        FakeProvider("openai", settings, [fatal("openai")]),
    ]

    with pytest.raises(AllProvidersFailed) as exc:
        await gw(settings, chain).complete(REQ)

    assert len(exc.value.attempts) == 2


async def test_stub_provider_is_opt_in_and_returns_supplied_payload(settings):
    disabled = StubProvider(settings)
    assert not disabled.available(), "stub must stay off unless explicitly enabled"

    enabled_settings = dataclasses.replace(settings, allow_stub_provider=True)
    enabled = StubProvider(enabled_settings)
    assert enabled.available()

    resp = await gw(enabled_settings, [enabled]).complete(
        GatewayRequest(system="s", user="u", stub_response='{"ok": true}')
    )
    assert resp.text == '{"ok": true}'


def test_settings_read_provider_keys_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    fresh = Settings()
    assert fresh.anthropic_api_key == "test-key"
    assert fresh.openai_api_key == ""
