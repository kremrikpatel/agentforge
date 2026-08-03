"""Ingestion, hybrid search, fusion, reranking, and the non-LLM retrieval modes."""

from __future__ import annotations

import dataclasses

import pytest
from qdrant_client import AsyncQdrantClient

from rag.cache import RagCache
from rag.config import RagSettings
from rag.lexical import InMemoryBM25
from rag.pipeline import RagPipeline
from rag.rerank import LexicalReranker, rerank
from rag.schemas import Document, RetrievalMode
from rag.store import HybridRetriever, VectorStore, chunk_document, reciprocal_rank_fusion
from tests.conftest import FakeRedis


class RagFakeRedis(FakeRedis):
    """Phase 1's fake predates the batched rerank-score lookup."""

    async def mget(self, keys):
        return [self.store.get(k) for k in keys]


@pytest.fixture
def rag_redis() -> RagFakeRedis:
    return RagFakeRedis()


CORPUS = [
    Document(
        doc_id="bucket",
        title="Token Bucket",
        text=(
            "A token bucket admits bursts up to a fixed capacity while enforcing a steady "
            "average rate. Tokens refill continuously and a request consumes one token."
        ),
    ),
    Document(
        doc_id="window",
        title="Sliding Window",
        text=(
            "A sliding window counter tracks requests across a rolling interval, which "
            "avoids the boundary spikes that fixed window counters suffer from."
        ),
    ),
    Document(
        doc_id="bread",
        title="Sourdough Starter",
        text=(
            "Autolyse the flour and water before adding the starter. A wet dough needs "
            "several stretch and fold cycles during bulk fermentation."
        ),
    ),
]


@pytest.fixture
def rag_settings() -> RagSettings:
    return dataclasses.replace(
        RagSettings(),
        qdrant_url="",
        qdrant_path="",       # embedded :memory:
        cache_enabled=False,
        top_k_final=3,
        top_k_fused=6,
    )


@pytest.fixture
async def retriever(rag_settings):
    store = VectorStore(rag_settings, client=AsyncQdrantClient(location=":memory:"))
    hybrid = HybridRetriever(store, InMemoryBM25(), rag_settings)
    await hybrid.ingest("kb", CORPUS)
    yield hybrid
    await store.aclose()


@pytest.fixture
def pipeline(rag_settings, retriever):
    # The non-LLM modes never reach the gateway.
    return RagPipeline(
        gateway=None,  # type: ignore[arg-type]
        retriever=retriever,
        reranker=LexicalReranker(),
        rag_settings=rag_settings,
    )


# --- chunking --------------------------------------------------------------


def test_chunking_packs_paragraphs_and_overlaps():
    doc = Document(doc_id="d", text="\n\n".join(f"Paragraph {i} body text." for i in range(20)))
    chunks = chunk_document(doc, "kb", size=120, overlap=30)

    assert len(chunks) > 1
    assert all(len(c.text) <= 160 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic_so_reingest_upserts():
    doc = Document(doc_id="d", text="one body")
    first = chunk_document(doc, "kb", 900, 150)
    second = chunk_document(doc, "kb", 900, 150)

    assert [c.id for c in first] == [c.id for c in second]
    # Different kb -> different point, so knowledge bases cannot collide.
    assert first[0].id != chunk_document(doc, "other", 900, 150)[0].id


def test_oversized_paragraph_is_hard_split():
    doc = Document(doc_id="d", text="x" * 1000)
    chunks = chunk_document(doc, "kb", size=200, overlap=40)
    assert len(chunks) > 1


# --- lexical ---------------------------------------------------------------


async def test_bm25_ranks_the_exact_term_match_first(retriever):
    hits = await retriever.lexical.search("kb", "sliding window counter", 5)
    assert hits, "BM25 returned nothing for a term present in the corpus"
    top_id = hits[0][0]
    hydrated = await retriever.store.hydrate("kb", [top_id])
    assert hydrated[top_id].doc_id == "window"


async def test_reingest_does_not_double_count_document_frequency(retriever):
    before = await retriever.lexical.search("kb", "token bucket", 5)
    await retriever.ingest("kb", CORPUS)  # same docs again
    after = await retriever.lexical.search("kb", "token bucket", 5)

    assert len(before) == len(after)
    assert before[0][1] == pytest.approx(after[0][1]), "re-ingest skewed BM25 statistics"


# --- fusion ----------------------------------------------------------------


def test_rrf_rewards_agreement_between_signals():
    fused = reciprocal_rank_fusion({"dense": ["a", "b"], "lexical": ["b", "a"]}, k=60)
    # b is rank 2 then 1; a is rank 1 then 2 -- symmetric, so they tie.
    assert fused["a"] == pytest.approx(fused["b"])

    lopsided = reciprocal_rank_fusion({"dense": ["a", "b"], "lexical": ["a", "z"]}, k=60)
    assert lopsided["a"] > lopsided["b"], "an id both signals rank highly must win"


async def test_hybrid_populates_both_signal_ranks(retriever):
    hits = await retriever.search("kb", dense_query="token bucket", lexical_query="token bucket")

    assert hits
    top = hits[0]
    assert top.chunk.doc_id == "bucket"
    assert top.dense_rank is not None and top.lexical_rank is not None


async def test_single_signal_modes_skip_fusion(retriever):
    dense = await retriever.search(
        "kb", dense_query="sliding window", lexical_query="ignored", use_lexical=False
    )
    lexical = await retriever.search(
        "kb", dense_query="ignored", lexical_query="sliding window", use_dense=False
    )

    assert dense and dense[0].dense_score is not None and dense[0].lexical_score is None
    assert lexical and lexical[0].lexical_score is not None and lexical[0].dense_score is None


# --- reranking -------------------------------------------------------------


async def test_rerank_scores_and_truncates(retriever):
    candidates = await retriever.search(
        "kb", dense_query="token bucket burst", lexical_query="token bucket burst"
    )
    ranked = await rerank("token bucket burst", candidates, 2, LexicalReranker())

    assert len(ranked) == 2
    assert all(c.rerank_score is not None for c in ranked)
    assert ranked[0].rerank_score >= ranked[1].rerank_score
    assert ranked[0].chunk.doc_id == "bucket"


async def test_rerank_cache_avoids_recomputation(retriever, settings, rag_settings, rag_redis):
    class CountingReranker(LexicalReranker):
        name = "counting"

        def __init__(self):
            self.calls = 0

        async def score(self, query, passages):
            self.calls += 1
            return await LexicalReranker().score(query, passages)

    cache = RagCache(
        settings, dataclasses.replace(rag_settings, cache_enabled=True), client=rag_redis
    )
    reranker = CountingReranker()
    candidates = await retriever.search(
        "kb", dense_query="token bucket", lexical_query="token bucket"
    )

    await rerank("token bucket", list(candidates), 3, reranker, cache)
    assert reranker.calls == 1

    await rerank("token bucket", list(candidates), 3, reranker, cache)
    assert reranker.calls == 1, "second pass should have been served entirely from cache"


# --- pipeline modes --------------------------------------------------------


async def test_hybrid_mode_returns_reranked_chunks_with_citations(pipeline):
    """Acceptance: retrieve() -> hybrid-searched, reranked chunks with citations."""
    result = await pipeline.retrieve("token bucket burst capacity", "kb", RetrievalMode.HYBRID)

    assert result.chunks
    assert result.action == "answer"
    assert result.chunks[0].chunk.doc_id == "bucket"
    assert all(c.rerank_score is not None for c in result.chunks), "rerank stage did not run"
    assert result.citations == [c.citation for c in result.chunks]
    assert result.citations[0].startswith("[Token Bucket#")
    assert result.citations[0] in result.context, "context must carry its citation markers"
    assert [s.name for s in result.steps] == ["search", "rerank"]


async def test_vector_and_lexical_modes_run_a_single_signal(pipeline):
    vector = await pipeline.retrieve("sliding window", "kb", RetrievalMode.VECTOR)
    lexical = await pipeline.retrieve("sliding window", "kb", RetrievalMode.LEXICAL)

    assert vector.chunks and lexical.chunks
    assert all(c.lexical_score is None for c in vector.chunks)
    assert all(c.dense_score is None for c in lexical.chunks)
    # Single-signal modes deliberately skip the rerank stage.
    assert [s.name for s in vector.steps] == ["search"]


async def test_empty_knowledge_base_is_reported_not_crashed(pipeline):
    result = await pipeline.retrieve("anything at all", "no-such-kb", RetrievalMode.HYBRID)

    assert result.chunks == []
    assert result.action == "insufficient"
    assert result.confidence == 0.0
    assert result.warnings


async def test_top_k_is_respected(pipeline):
    result = await pipeline.retrieve("rate limiting", "kb", RetrievalMode.HYBRID, top_k=1)
    assert len(result.chunks) == 1
