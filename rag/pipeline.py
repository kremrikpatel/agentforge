"""The one entry point: `retrieve(query, kb_id, mode)`.

Every mode walks the same funnel and stops at a different depth, so there is one
code path rather than five:

    HyDE rewrite -> hybrid search -> rerank -> CRAG grading -> Self-RAG loop

  vector/lexical  stop after search (single signal, no fusion)
  hybrid          stop after rerank
  hyde            rewrite the dense query first, then as hybrid
  crag            add grading and the corrective path
  self_rag        add the generation/grounding loop
  text2sql        a different funnel entirely: propose, never execute
"""

from __future__ import annotations

from app.observability import get_logger, log_event, timed
from gateway.client import LLMGateway
from rag.cache import RagCache
from rag.config import RagSettings, get_rag_settings
from rag.crag import CorrectiveRag, WebSearchFn
from rag.hyde import build_hyde_query
from rag.rerank import Reranker, build_reranker
from rag.rerank import rerank as rerank_chunks
from rag.schemas import (
    Document,
    RetrievalMode,
    RetrievalResult,
    RetrievalStep,
    ScoredChunk,
)
from rag.selfrag import GenerateFn, SelfRag, assemble_context
from rag.store import HybridRetriever
from rag.text2sql import SqlSchema, Text2Sql

logger = get_logger("agentforge.rag.pipeline")

_SINGLE_SIGNAL = {RetrievalMode.VECTOR, RetrievalMode.LEXICAL}
_GRADED = {RetrievalMode.CRAG, RetrievalMode.SELF_RAG}


class RagPipeline:
    def __init__(
        self,
        gateway: LLMGateway,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        cache: RagCache | None = None,
        rag_settings: RagSettings | None = None,
        sql_schema: SqlSchema | None = None,
        web_search: WebSearchFn | None = None,
    ) -> None:
        self.rag = rag_settings or get_rag_settings()
        self.gateway = gateway
        self.cache = cache
        self.retriever = retriever or HybridRetriever(rag_settings=self.rag)
        if self.cache is not None and self.retriever.store.cache is None:
            self.retriever.store.cache = self.cache
        self.reranker = reranker or build_reranker(self.rag)
        self.crag = CorrectiveRag(gateway, self.rag, web_search)
        self.self_rag = SelfRag(gateway, self.rag)
        self.sql_schema = sql_schema
        self.text2sql = (
            Text2Sql(gateway, sql_schema, rag_settings=self.rag) if sql_schema else None
        )

    async def ingest(self, kb_id: str, documents: list[Document]) -> int:
        return await self.retriever.ingest(kb_id, documents)

    async def retrieve(
        self,
        query: str,
        kb_id: str = "default",
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        *,
        generate: GenerateFn | None = None,
        use_hyde: bool | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        mode = RetrievalMode(mode)
        top_k = top_k or self.rag.top_k_final
        result = RetrievalResult(query=query, effective_query=query, kb_id=kb_id, mode=mode)

        with timed() as total:
            if mode is RetrievalMode.TEXT2SQL:
                await self._text2sql(query, result)
            else:
                await self._retrieval_funnel(
                    query, kb_id, mode, result, generate, use_hyde, top_k
                )

        result.total_latency_ms = total["ms"]
        log_event(
            logger,
            "rag.retrieved",
            kb_id=kb_id,
            mode=mode.value,
            action=result.action,
            chunks=len(result.chunks),
            confidence=result.confidence,
            latency_ms=total["ms"],
        )
        return result

    # --- funnel stages -----------------------------------------------------

    async def _retrieval_funnel(
        self,
        query: str,
        kb_id: str,
        mode: RetrievalMode,
        result: RetrievalResult,
        generate: GenerateFn | None,
        use_hyde: bool | None,
        top_k: int,
    ) -> None:
        dense_query = query
        wants_hyde = (mode is RetrievalMode.HYDE) if use_hyde is None else use_hyde

        if wants_hyde:
            with timed() as t:
                hyde = await build_hyde_query(query, self.gateway, self.rag)
            dense_query = hyde.dense_query
            result.effective_query = hyde.dense_query
            result.steps.append(
                RetrievalStep(
                    name="hyde",
                    detail=hyde.detail or f"hypothesis: {hyde.hypothesis[:160]}",
                    provider=hyde.provider,
                    latency_ms=t["ms"],
                )
            )
            if not hyde.used:
                result.warnings.append(f"hyde unavailable ({hyde.detail}); used the raw query")

        async def search(q: str) -> list[ScoredChunk]:
            return await self.retriever.search(
                kb_id,
                dense_query=q,
                lexical_query=q,
                use_dense=mode is not RetrievalMode.LEXICAL,
                use_lexical=mode is not RetrievalMode.VECTOR,
            )

        with timed() as t:
            chunks = await self.retriever.search(
                kb_id,
                dense_query=dense_query,
                lexical_query=query,  # lexical always uses the user's own words
                use_dense=mode is not RetrievalMode.LEXICAL,
                use_lexical=mode is not RetrievalMode.VECTOR,
            )
        result.steps.append(
            RetrievalStep(
                name="search",
                detail=mode.value,
                candidates_out=len(chunks),
                latency_ms=t["ms"],
            )
        )

        if mode not in _SINGLE_SIGNAL and chunks:
            with timed() as t:
                before = len(chunks)
                chunks = await rerank_chunks(query, chunks, top_k, self.reranker, self.cache)
            result.steps.append(
                RetrievalStep(
                    name="rerank",
                    detail=self.reranker.name,
                    candidates_in=before,
                    candidates_out=len(chunks),
                    latency_ms=t["ms"],
                )
            )
        else:
            chunks = chunks[:top_k]

        if mode in _GRADED:
            with timed() as t:
                before = len(chunks)
                chunks, verdict = await self.crag.run(query, chunks, search)
                chunks = chunks[:top_k]
            result.crag = verdict
            result.action = (
                "corrected"
                if verdict.used_fallback or verdict.reformulated_query
                else "answer"
            )
            result.steps.append(
                RetrievalStep(
                    name="crag",
                    detail=f"{verdict.action.value}: {verdict.detail}",
                    candidates_in=before,
                    candidates_out=len(chunks),
                    latency_ms=t["ms"],
                )
            )

        if mode is RetrievalMode.SELF_RAG:
            if generate is None:
                result.warnings.append(
                    "self_rag needs a generate callable; ran the crag funnel only"
                )
            else:
                with timed() as t:
                    outcome = await self.self_rag.run(query, chunks, generate, search)
                chunks = outcome.chunks[:top_k]
                result.grounding = outcome.verdict
                if not outcome.verdict.grounded:
                    result.action = "insufficient"
                    result.warnings.append(
                        f"answer still ungrounded after {outcome.loops} retrieval loop(s)"
                    )
                result.steps.append(
                    RetrievalStep(
                        name="self_rag",
                        detail=f"loops={outcome.loops} grounded={outcome.verdict.grounded}",
                        candidates_out=len(chunks),
                        latency_ms=t["ms"],
                    )
                )

        result.chunks = chunks
        result.context = assemble_context(chunks)
        result.citations = [c.citation for c in chunks]
        result.confidence = _confidence(result, chunks)
        if not chunks:
            result.action = "insufficient"
            result.warnings.append("no chunks survived the retrieval funnel")

    async def _text2sql(self, question: str, result: RetrievalResult) -> None:
        if self.text2sql is None:
            result.action = "insufficient"
            result.warnings.append("text2sql needs a SqlSchema; none was configured")
            return

        with timed() as t:
            proposal = await self.text2sql.propose(question)
        result.sql = proposal
        # Even a rejected proposal is returned for a human to look at. Nothing ran.
        result.action = "needs_approval"
        result.confidence = 1.0 if proposal.safe else 0.0
        if not proposal.safe:
            result.warnings.append(f"sql rejected: {proposal.rejection_reason}")
        result.steps.append(
            RetrievalStep(
                name="text2sql",
                detail="safe" if proposal.safe else proposal.rejection_reason,
                latency_ms=t["ms"],
            )
        )


def _confidence(result: RetrievalResult, chunks: list[ScoredChunk]) -> float:
    if result.grounding is not None:
        return result.grounding.score
    if result.crag is not None:
        return result.crag.confidence
    if not chunks:
        return 0.0
    top = chunks[0]
    score = top.rerank_score if top.rerank_score is not None else top.score
    return max(0.0, min(1.0, float(score)))


# --------------------------------------------------------------------------
# Module-level convenience
# --------------------------------------------------------------------------

_default: RagPipeline | None = None


def get_pipeline(gateway: LLMGateway | None = None, **kwargs) -> RagPipeline:
    """Process-wide pipeline.

    Pass a configured one to the tool factory instead when you need per-tenant
    stores or a custom SQL schema.
    """
    global _default
    if _default is None:
        _default = RagPipeline(gateway or LLMGateway(), **kwargs)
    return _default


async def retrieve(
    query: str,
    kb_id: str = "default",
    mode: RetrievalMode | str = RetrievalMode.HYBRID,
    **kwargs,
) -> RetrievalResult:
    """The interface Phase 1 agent nodes call."""
    return await get_pipeline().retrieve(query, kb_id, mode, **kwargs)
