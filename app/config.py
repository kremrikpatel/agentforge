"""Environment-backed configuration. No secrets in source -- see .env.example."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw.isdigit() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM gateway -------------------------------------------------------
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))

    anthropic_model: str = field(
        default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    )
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o"))
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash")
    )
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 60.0))
    llm_retries: int = field(default_factory=lambda: _env_int("LLM_RETRIES", 2))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))

    # Deterministic offline provider, last in the chain. Lets `docker compose up`
    # produce a working pipeline with zero API keys.
    allow_stub_provider: bool = field(
        default_factory=lambda: _env_bool("ALLOW_STUB_PROVIDER", False)
    )

    # --- Memory ------------------------------------------------------------
    redis_url: str = field(
        default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0")
    )
    postgres_dsn: str = field(
        default_factory=lambda: _env(
            "POSTGRES_DSN",
            "postgresql://agentforge:agentforge@localhost:5432/agentforge",
        )
    )
    session_ttl_s: int = field(default_factory=lambda: _env_int("SESSION_TTL_S", 3600))
    cache_ttl_s: int = field(default_factory=lambda: _env_int("CACHE_TTL_S", 900))
    cache_similarity_threshold: float = field(
        default_factory=lambda: _env_float("CACHE_SIMILARITY_THRESHOLD", 0.85)
    )
    cache_index_size: int = field(default_factory=lambda: _env_int("CACHE_INDEX_SIZE", 200))
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))

    # --- Guardrails --------------------------------------------------------
    guardrails_llm_tier: bool = field(
        default_factory=lambda: _env_bool("GUARDRAILS_LLM_TIER", False)
    )
    guardrails_block_threshold: float = field(
        default_factory=lambda: _env_float("GUARDRAILS_BLOCK_THRESHOLD", 0.6)
    )

    # --- App ---------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


def get_settings() -> Settings:
    """Read settings fresh. Cheap, and keeps tests free to monkeypatch os.environ."""
    return Settings()
