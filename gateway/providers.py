"""Provider adapters, one per vendor HTTP API.

Deliberately raw httpx rather than four vendor SDKs: each adapter is ~20 lines,
and it keeps the dependency surface at zero extra packages.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import httpx

from app.config import Settings

# Non-retryable: the request itself is wrong (bad/missing key, unknown model).
# Retrying will not help -- fall through to the next provider immediately.
_FATAL_STATUS = {400, 401, 403, 404, 422}


class ProviderError(Exception):
    """A provider call that failed, carrying whether trying again could help.

    The `retryable` flag is the whole reason this type exists: the gateway needs
    to tell "this vendor is briefly unwell" from "this request will never work",
    and only the adapter that made the call knows which it saw.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        """Record the verdict alongside the message so callers cannot lose it."""
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class GatewayRequest:
    """One vendor-neutral completion request, shaped to the smallest common API."""

    system: str
    user: str
    max_tokens: int = 2048
    temperature: float = 0.2
    # Deterministic payload used only by StubProvider, so the offline chain can
    # still emit a schema-valid contract. Ignored by real providers.
    stub_response: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """A completion plus what it cost to get: which provider, how many tries."""

    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    attempts: int = 1
    fallback_depth: int = 0


class BaseProvider(abc.ABC):
    """Shared HTTP plumbing; subclasses supply only credentials and payload shape.

    Keeping the retryable/fatal classification and the defensive response
    unwrapping here means a new vendor cannot accidentally get either wrong.
    """

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        """Hold settings rather than the key, so a re-read of config takes effect."""
        self.settings = settings

    @property
    @abc.abstractmethod
    def api_key(self) -> str:
        """The configured credential, or empty string when this vendor is unset."""

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """The model id this adapter will ask for."""

    def available(self) -> bool:
        """A blank key means 'not configured' -- skip without burning a retry."""
        return bool(self.api_key)

    @abc.abstractmethod
    async def complete(self, req: GatewayRequest, client: httpx.AsyncClient) -> str:
        """Translate the request into this vendor's dialect and return the text.

        Takes a caller-owned client so connection reuse and timeouts are decided
        once by the gateway rather than five times over in the adapters.
        """

    async def _post(self, client: httpx.AsyncClient, url: str, **kwargs) -> dict:
        try:
            resp = await client.post(url, **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name}: timeout ({exc})", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"{self.name}: transport error ({exc})", retryable=True
            ) from exc

        if resp.status_code in _FATAL_STATUS:
            raise ProviderError(
                f"{self.name}: HTTP {resp.status_code} {resp.text[:200]}", retryable=False
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.name}: HTTP {resp.status_code} {resp.text[:200]}", retryable=True
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.name}: non-JSON body", retryable=True) from exc

    @staticmethod
    def _dig(data: dict, *path, default: str = "") -> str:
        cur: object = data
        for key in path:
            if isinstance(cur, dict):
                cur = cur.get(key)
            elif isinstance(cur, list) and isinstance(key, int) and len(cur) > key:
                cur = cur[key]
            else:
                return default
        return cur if isinstance(cur, str) else default


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API -- system prompt is a top-level field, not a message."""

    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    @property
    def api_key(self) -> str:
        """Return the configured Anthropic key."""
        return self.settings.anthropic_api_key

    @property
    def model(self) -> str:
        """Return the configured Anthropic model id."""
        return self.settings.anthropic_model

    async def complete(self, req: GatewayRequest, client: httpx.AsyncClient) -> str:
        """Post one message turn; treat an empty completion as worth retrying."""
        data = await self._post(
            client,
            self.url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "system": req.system,
                "messages": [{"role": "user", "content": req.user}],
            },
        )
        text = self._dig(data, "content", 0, "text")
        if not text:
            raise ProviderError("anthropic: empty completion", retryable=True)
        return text


class _OpenAICompatible(BaseProvider):
    """OpenAI and Groq speak the same chat-completions dialect."""

    url = ""

    async def complete(self, req: GatewayRequest, client: httpx.AsyncClient) -> str:
        """Post the shared chat-completions body; subclasses differ only in URL."""
        data = await self._post(
            client,
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "messages": [
                    {"role": "system", "content": req.system},
                    {"role": "user", "content": req.user},
                ],
            },
        )
        text = self._dig(data, "choices", 0, "message", "content")
        if not text:
            raise ProviderError(f"{self.name}: empty completion", retryable=True)
        return text


class OpenAIProvider(_OpenAICompatible):
    """OpenAI chat completions."""

    name = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    @property
    def api_key(self) -> str:
        """Return the configured OpenAI key."""
        return self.settings.openai_api_key

    @property
    def model(self) -> str:
        """Return the configured OpenAI model id."""
        return self.settings.openai_model


class GroqProvider(_OpenAICompatible):
    """Groq: last real link in the chain, so it catches whatever the others drop."""

    name = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def api_key(self) -> str:
        """Return the configured Groq key."""
        return self.settings.groq_api_key

    @property
    def model(self) -> str:
        """Return the configured Groq model id."""
        return self.settings.groq_model


class GeminiProvider(BaseProvider):
    """Google Generative Language API -- model id rides in the path, key in a header."""

    name = "gemini"
    base = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def api_key(self) -> str:
        """Return the configured Gemini key."""
        return self.settings.gemini_api_key

    @property
    def model(self) -> str:
        """Return the configured Gemini model id."""
        return self.settings.gemini_model

    async def complete(self, req: GatewayRequest, client: httpx.AsyncClient) -> str:
        """Send the key as a header rather than a query param -- URLs get logged."""
        data = await self._post(
            client,
            f"{self.base}/{self.model}:generateContent",
            headers={"content-type": "application/json", "x-goog-api-key": self.api_key},
            json={
                "systemInstruction": {"parts": [{"text": req.system}]},
                "contents": [{"role": "user", "parts": [{"text": req.user}]}],
                "generationConfig": {
                    "maxOutputTokens": req.max_tokens,
                    "temperature": req.temperature,
                },
            },
        )
        text = self._dig(data, "candidates", 0, "content", "parts", 0, "text")
        if not text:
            raise ProviderError("gemini: empty completion", retryable=True)
        return text


class StubProvider(BaseProvider):
    """Offline last resort so the stack demos with zero API keys.

    Returns whatever deterministic payload the caller supplied. Enabled only via
    ALLOW_STUB_PROVIDER=true (docker-compose sets it; production must not).
    """

    name = "stub"

    @property
    def api_key(self) -> str:
        """Fake a key when stubbing is allowed; empty otherwise so `available` says no."""
        return "stub" if self.settings.allow_stub_provider else ""

    @property
    def model(self) -> str:
        """Name the absence of a model, so a trace shows plainly nothing was called."""
        return "offline-deterministic"

    async def complete(self, req: GatewayRequest, client: httpx.AsyncClient) -> str:
        """Echo the caller's payload; a missing one is fatal, not retryable.

        Retrying would spin forever -- no amount of waiting makes a caller supply
        an offline payload it never had.
        """
        if not req.stub_response:
            raise ProviderError("stub: caller supplied no offline payload", retryable=False)
        return req.stub_response


# Order is the fallback chain: Claude -> GPT-4o -> Gemini -> Groq -> offline stub.
DEFAULT_CHAIN: tuple[type[BaseProvider], ...] = (
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    GroqProvider,
    StubProvider,
)


def build_chain(settings: Settings) -> list[BaseProvider]:
    """Build every provider, configured or not -- `available` does the filtering."""
    return [cls(settings) for cls in DEFAULT_CHAIN]
