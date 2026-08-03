"""Cross-encoder reranking, with a dependency-free fallback.

A cross-encoder reads (query, passage) jointly and is far more accurate than the
bi-encoder retrieval that produced the candidates -- which is exactly why it is
too slow to run over a whole corpus, and belongs here at the narrow end of the
funnel.

sentence-transformers is an optional extra (`pip install '.[rerank]'`) because it
pulls torch. Without it, LexicalReranker keeps the stage functional and the
interface identical.
"""

from __future__ import annotations

import asyncio
import math
from typing import Protocol, runtime_checkable

from app.observability import get_logger, log_event
from memory.embedding import cosine, embed, tokenize
from rag.cache import RagCache
from rag.config import RagSettings, get_rag_settings
from rag.schemas import ScoredChunk

logger = get_logger("agentforge.rag.rerank")


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def score(self, query: str, passages: list[str]) -> list[float]: ...


class LexicalReranker:
    """Query-term coverage blended with embedding similarity.

    ponytail: no joint query-passage attention, so it cannot spot relevance that
    needs real reading. Upgrade path: install the `rerank` extra and
    CrossEncoderReranker is selected automatically.
    """

    name = "lexical"

    async def score(self, query: str, passages: list[str]) -> list[float]:
        terms = set(tokenize(query))
        query_vec = embed(query)
        scores: list[float] = []
        for passage in passages:
            passage_terms = set(tokenize(passage))
            coverage = len(terms & passage_terms) / len(terms) if terms else 0.0
            similarity = (cosine(query_vec, embed(passage)) + 1.0) / 2.0
            # Coverage weighted higher: an exact term match is strong evidence.
            scores.append(round(0.6 * coverage + 0.4 * similarity, 6))
        return scores


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder, loaded lazily and run off-loop."""

    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    async def score(self, query: str, passages: list[str]) -> list[float]:
        def _predict() -> list[float]:
            model = self._load()
            raw = model.predict([(query, p) for p in passages])
            # ms-marco cross-encoders emit unbounded logits; squash to 0..1 so
            # downstream thresholds (CRAG grading) stay comparable across models.
            return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]

        return await asyncio.to_thread(_predict)


def build_reranker(rag_settings: RagSettings | None = None) -> Reranker:
    rag = rag_settings or get_rag_settings()
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        log_event(
            logger,
            "rag.reranker",
            backend=LexicalReranker.name,
            reason="rerank extra not installed",
        )
        return LexicalReranker()
    log_event(logger, "rag.reranker", backend=CrossEncoderReranker.name, model=rag.rerank_model)
    return CrossEncoderReranker(rag.rerank_model)


async def rerank(
    query: str,
    chunks: list[ScoredChunk],
    top_k: int,
    reranker: Reranker,
    cache: RagCache | None = None,
) -> list[ScoredChunk]:
    """Score, sort, truncate. Cached scores skip the model entirely."""
    if not chunks:
        return []

    ids = [c.chunk.id for c in chunks]
    cached = await cache.get_rerank_scores(query, ids) if cache is not None else {}
    pending = [c for c in chunks if c.chunk.id not in cached]

    if pending:
        fresh = await reranker.score(query, [c.chunk.text for c in pending])
        computed = {c.chunk.id: float(s) for c, s in zip(pending, fresh)}
        if cache is not None:
            await cache.set_rerank_scores(query, computed)
        cached.update(computed)

    for chunk in chunks:
        chunk.rerank_score = cached.get(chunk.chunk.id, 0.0)
        # The rerank score becomes the ordering score from here on; the fused
        # RRF value stays available on dense_score/lexical_score for debugging.
        chunk.score = chunk.rerank_score

    chunks.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
    return chunks[:top_k]
