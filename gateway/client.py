"""Fallback chain + retry policy over the provider adapters."""

from __future__ import annotations

import asyncio

import httpx

from app.config import Settings, get_settings
from app.observability import get_logger, log_event, timed
from gateway.providers import (
    BaseProvider,
    GatewayRequest,
    LLMResponse,
    ProviderError,
    build_chain,
)

logger = get_logger("agentforge.gateway")


class AllProvidersFailed(RuntimeError):
    def __init__(self, attempts: list[str]) -> None:
        super().__init__("every provider in the fallback chain failed: " + "; ".join(attempts))
        self.attempts = attempts


class LLMGateway:
    """Try each configured provider in order; retry the retryable failures.

    `providers` is injectable so tests can drive the chain without network.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        providers: list[BaseProvider] | None = None,
        client: httpx.AsyncClient | None = None,
        backoff_base_s: float = 0.5,
    ) -> None:
        self.settings = settings or get_settings()
        self.providers = providers if providers is not None else build_chain(self.settings)
        self._client = client
        self._owns_client = client is None
        self.backoff_base_s = backoff_base_s

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.llm_timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def complete(self, req: GatewayRequest) -> LLMResponse:
        client = await self._http()
        max_attempts = max(1, self.settings.llm_retries + 1)
        failures: list[str] = []

        for depth, provider in enumerate(self.providers):
            if not provider.available():
                failures.append(f"{provider.name}: not configured (no API key)")
                log_event(logger, "gateway.skipped", provider=provider.name, reason="no_api_key")
                continue

            for attempt in range(1, max_attempts + 1):
                try:
                    with timed() as t:
                        text = await provider.complete(req, client)
                except ProviderError as exc:
                    failures.append(str(exc))
                    log_event(
                        logger,
                        "gateway.attempt_failed",
                        provider=provider.name,
                        model=provider.model,
                        attempt=attempt,
                        retryable=exc.retryable,
                        error=str(exc),
                    )
                    if not exc.retryable:
                        break  # provider itself is broken -- move to next provider
                    if attempt < max_attempts:
                        await asyncio.sleep(self.backoff_base_s * (2 ** (attempt - 1)))
                    continue

                log_event(
                    logger,
                    "gateway.success",
                    provider=provider.name,
                    model=provider.model,
                    attempt=attempt,
                    fallback_depth=depth,
                    latency_ms=t["ms"],
                )
                return LLMResponse(
                    text=text,
                    provider=provider.name,
                    model=provider.model,
                    latency_ms=t["ms"],
                    attempts=attempt,
                    fallback_depth=depth,
                )

        log_event(logger, "gateway.exhausted", failures=failures)
        raise AllProvidersFailed(failures)
