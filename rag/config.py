"""RAG settings.

Separate from app.config on purpose: Phase 2 lives in rag/, so it owns its own
knobs and only borrows the env helpers. Every stage's top-k is configurable
because the latency/cost profile of the funnel is the main thing you tune.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import _env, _env_bool, _env_float, _env_int, get_settings


@dataclass(frozen=True)
class RagSettings:
    # --- Qdrant ------------------------------------------------------------
    # Blank URL => embedded local mode (":memory:" unless QDRANT_PATH is set),
    # so the module runs with no server at all.
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL"))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    qdrant_path: str = field(default_factory=lambda: _env("QDRANT_PATH"))
    collection_prefix: str = field(
        default_factory=lambda: _env("RAG_COLLECTION_PREFIX", "af_kb_")
    )

    # --- retrieval funnel --------------------------------------------------
    # Wide at the cheap stages, narrow at the expensive ones.
    top_k_dense: int = field(default_factory=lambda: _env_int("RAG_TOP_K_DENSE", 20))
    top_k_lexical: int = field(default_factory=lambda: _env_int("RAG_TOP_K_LEXICAL", 20))
    top_k_fused: int = field(default_factory=lambda: _env_int("RAG_TOP_K_FUSED", 12))
    top_k_final: int = field(default_factory=lambda: _env_int("RAG_TOP_K_FINAL", 5))
    # Reciprocal-rank-fusion damping. 60 is the value from the original RRF paper.
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))

    # --- chunking ----------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 150))

    # --- reranking ---------------------------------------------------------
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("RAG_RERANK", True))
    rerank_model: str = field(
        default_factory=lambda: _env(
            "RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )

    # --- HyDE --------------------------------------------------------------
    hyde_max_tokens: int = field(default_factory=lambda: _env_int("RAG_HYDE_MAX_TOKENS", 400))
    # Blend the hypothetical answer with the raw query so HyDE cannot drift
    # entirely off-topic when the model hallucinates a bad hypothesis.
    hyde_query_weight: float = field(
        default_factory=lambda: _env_float("RAG_HYDE_QUERY_WEIGHT", 0.3)
    )

    # --- CRAG --------------------------------------------------------------
    # >= correct_at   -> answer as-is
    # <= incorrect_at -> discard and take the corrective path
    # in between      -> ambiguous: keep what is good, and also correct
    crag_correct_at: float = field(
        default_factory=lambda: _env_float("RAG_CRAG_CORRECT_AT", 0.6)
    )
    crag_incorrect_at: float = field(
        default_factory=lambda: _env_float("RAG_CRAG_INCORRECT_AT", 0.3)
    )

    # --- Self-RAG ----------------------------------------------------------
    grounding_threshold: float = field(
        default_factory=lambda: _env_float("RAG_GROUNDING_THRESHOLD", 0.6)
    )
    self_rag_max_loops: int = field(default_factory=lambda: _env_int("RAG_SELF_RAG_LOOPS", 2))

    # --- Text2SQL ----------------------------------------------------------
    text2sql_max_rows: int = field(default_factory=lambda: _env_int("RAG_SQL_MAX_ROWS", 200))
    text2sql_timeout_ms: int = field(
        default_factory=lambda: _env_int("RAG_SQL_TIMEOUT_MS", 5000)
    )

    # --- caching -----------------------------------------------------------
    cache_ttl_s: int = field(default_factory=lambda: _env_int("RAG_CACHE_TTL_S", 3600))
    cache_enabled: bool = field(default_factory=lambda: _env_bool("RAG_CACHE", True))

    @property
    def embedding_dim(self) -> int:
        return get_settings().embedding_dim

    def collection(self, kb_id: str) -> str:
        return f"{self.collection_prefix}{kb_id}"


def get_rag_settings() -> RagSettings:
    return RagSettings()
