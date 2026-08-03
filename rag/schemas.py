"""Types crossing the RAG boundary.

`RetrievalResult` is the one shape an agent ever sees, whatever mode ran. The
`steps` trail is what makes a five-strategy funnel debuggable: each stage records
what it received, what it emitted, and how long it took.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


Score = Annotated[float, Field(ge=0.0, le=1.0)]


class RetrievalMode(StrEnum):
    """Each mode adds one stage to the one before it."""

    VECTOR = "vector"       # dense only
    LEXICAL = "lexical"     # BM25 / full-text only
    HYBRID = "hybrid"       # dense + lexical, RRF-fused, then reranked
    HYDE = "hyde"           # hybrid, with a hypothetical-document query rewrite
    CRAG = "crag"           # hyde-capable hybrid + relevance grading + correction
    SELF_RAG = "self_rag"   # crag + a generation-time "retrieve again" loop
    TEXT2SQL = "text2sql"   # natural language -> approval-gated SQL


class Relevance(StrEnum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


class Document(BaseModel):
    """Ingestion input, before chunking."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A stored, retrievable unit. `id` is deterministic so re-ingest upserts."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kb_id: str
    doc_id: str
    ordinal: int
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: str = Field(default_factory=_now)

    @staticmethod
    def make_id(kb_id: str, doc_id: str, ordinal: int) -> str:
        """UUID-shaped so Qdrant accepts it as a point id and re-ingest overwrites."""
        seed = f"{kb_id}:{doc_id}:{ordinal}"
        return str(uuid.UUID(hashlib.sha1(seed.encode()).hexdigest()[:32]))

    @property
    def citation(self) -> str:
        label = self.title or self.source or self.doc_id
        return f"[{label}#{self.ordinal}]"


class ScoredChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float = 0.0
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    relevance: Relevance | None = None
    grade: float | None = None

    @property
    def citation(self) -> str:
        return self.chunk.citation


# --------------------------------------------------------------------------
# Text2SQL
# --------------------------------------------------------------------------


class SqlProposal(BaseModel):
    """Generated SQL awaiting a human. Never carries rows -- nothing has run."""

    model_config = ConfigDict(extra="forbid")

    question: str
    sql: str = ""
    explanation: str = ""
    tables: list[str] = Field(default_factory=list)
    plan: str = Field(default="", description="EXPLAIN output, the human-readable preview.")
    safe: bool = False
    rejection_reason: str = ""
    # Binds an approval to this exact SQL: approving one statement cannot be
    # replayed to execute a different one.
    approval_token: str = ""
    requires_approval: Literal[True] = True
    created_at: str = Field(default_factory=_now)

    @staticmethod
    def token_for(sql: str) -> str:
        return hashlib.sha256(sql.strip().encode("utf-8")).hexdigest()


class SqlExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str = ""


# --------------------------------------------------------------------------
# Strategy verdicts
# --------------------------------------------------------------------------


class CragVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Relevance
    confidence: Score = 0.0
    kept: int = 0
    dropped: int = 0
    reformulated_query: str = ""
    used_fallback: bool = False
    detail: str = ""


class GroundingVerdict(BaseModel):
    """Self-RAG's mid-generation signal."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool = True
    score: Score = 1.0
    action: Literal["accept", "retrieve_again"] = "accept"
    unsupported_claims: list[str] = Field(default_factory=list)
    suggested_query: str = ""
    detail: str = ""


class RetrievalStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    detail: str = ""
    candidates_in: int = 0
    candidates_out: int = 0
    latency_ms: float = 0.0
    provider: str = ""


class RetrievalResult(BaseModel):
    """The single shape every mode returns."""

    model_config = ConfigDict(extra="forbid")

    query: str
    effective_query: str = ""
    kb_id: str = ""
    mode: RetrievalMode = RetrievalMode.HYBRID
    action: Literal["answer", "corrected", "insufficient", "needs_approval"] = "answer"

    chunks: list[ScoredChunk] = Field(default_factory=list)
    context: str = ""
    citations: list[str] = Field(default_factory=list)
    confidence: Score = 0.0

    crag: CragVerdict | None = None
    grounding: GroundingVerdict | None = None
    sql: SqlProposal | None = None

    steps: list[RetrievalStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0

    def as_tool_output(self, max_chars: int = 6000) -> str:
        """Compact rendering for an agent prompt; the full object stays available."""
        if self.sql is not None:
            return (
                f"[text2sql · {self.action}]\n{self.sql.sql}\n\n"
                f"{self.sql.explanation}\n\nPlan:\n{self.sql.plan}\n"
                f"Approval required before execution."
            )[:max_chars]
        if not self.chunks:
            return f"[{self.mode} · no supporting context found] " + "; ".join(self.warnings)
        return self.context[:max_chars]
