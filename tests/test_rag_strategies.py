"""HyDE, CRAG, and Self-RAG -- the three LLM-driven strategies."""

from __future__ import annotations

import dataclasses

import pytest
from qdrant_client import AsyncQdrantClient

from gateway.client import LLMGateway
from rag.config import RagSettings
from rag.crag import CorrectiveRag, grade_chunks
from rag.hyde import blend_query, build_hyde_query
from rag.lexical import InMemoryBM25
from rag.pipeline import RagPipeline
from rag.rerank import LexicalReranker
from rag.schemas import Document, Relevance, RetrievalMode
from rag.selfrag import SelfRag
from rag.store import HybridRetriever, VectorStore
from tests.conftest import FakeProvider

# The question and its answer share almost no vocabulary -- the exact case HyDE
# exists for. The distractor shares the question's words but answers nothing.
SPARSE_MATCH_QUERY = "how do I stop one noisy customer from overwhelming everyone else?"

HYDE_CORPUS = [
    Document(
        doc_id="throttle",
        title="Throttling",
        text=(
            "Token bucket throttling assigns each principal a refill rate and a burst "
            "capacity. Calls beyond the allowance receive HTTP 429 Too Many Requests "
            "with a Retry-After header."
        ),
    ),
    Document(
        doc_id="onboard",
        title="Customer Onboarding",
        text=(
            "When onboarding a noisy customer, greet everyone on the account, stop to "
            "confirm each contact, and check whether one customer needs extra help so "
            "nobody is overwhelming the team."
        ),
    ),
    Document(
        doc_id="bread",
        title="Sourdough",
        text="Autolyse the flour and water, then stretch and fold during bulk fermentation.",
    ),
]

IRRELEVANT_CORPUS = [
    Document(doc_id="b1", title="Sourdough", text="Autolyse flour and water before mixing."),
    Document(doc_id="b2", title="Proofing", text="Cold proof the dough overnight in a banneton."),
]

HYPOTHESIS = (
    '{"hypothesis": "Token bucket throttling assigns each principal a refill rate and '
    'burst capacity. Requests beyond the allowance receive HTTP 429 Too Many Requests."}'
)


@pytest.fixture
def rag_settings() -> RagSettings:
    return dataclasses.replace(
        RagSettings(),
        qdrant_url="",
        qdrant_path="",
        cache_enabled=False,
        top_k_final=3,
        top_k_fused=6,
        self_rag_max_loops=2,
    )


def make_gateway(settings, script: list[str]) -> LLMGateway:
    return LLMGateway(
        settings,
        providers=[FakeProvider("anthropic", settings, script)],
        client=object(),
        backoff_base_s=0.0,
    )


async def build(rag_settings, corpus, gateway) -> RagPipeline:
    store = VectorStore(rag_settings, client=AsyncQdrantClient(location=":memory:"))
    retriever = HybridRetriever(store, InMemoryBM25(), rag_settings)
    await retriever.ingest("kb", corpus)
    return RagPipeline(
        gateway=gateway,
        retriever=retriever,
        reranker=LexicalReranker(),
        rag_settings=rag_settings,
    )


def search_fn(pipe: RagPipeline, kb: str = "kb"):
    """The re-retrieval callable CRAG and Self-RAG are handed."""

    async def _search(query: str):
        return await pipe.retriever.search(kb, dense_query=query, lexical_query=query)

    return _search


# --- HyDE ------------------------------------------------------------------


def test_blend_query_weights_the_original_query_by_repetition():
    query = "alpha beta"
    hypothesis = " ".join(f"word{i}" for i in range(40))

    blended = blend_query(query, hypothesis, weight=0.5)
    assert blended.count("alpha") > 1, "query must be repeated to hold half the token mass"

    assert blend_query(query, hypothesis, weight=0.0) == hypothesis
    assert blend_query(query, hypothesis, weight=1.0) == query
    assert blend_query(query, "", weight=0.3) == query, "no hypothesis -> unchanged query"


async def test_hyde_degrades_to_the_raw_query_when_no_provider(settings):
    gateway = LLMGateway(settings, providers=[], client=object(), backoff_base_s=0.0)

    result = await build_hyde_query("anything", gateway)

    assert result.used is False
    assert result.dense_query == "anything"


async def test_hyde_measurably_changes_retrieval_on_a_sparse_match_query(rag_settings, settings):
    """Acceptance: HyDE changes retrieval where direct matches are sparse.

    Compared on the dense signal alone, which is the only thing HyDE alters.
    """
    baseline = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, []))
    plain = await baseline.retrieve(SPARSE_MATCH_QUERY, "kb", RetrievalMode.VECTOR)

    hyde_pipe = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, [HYPOTHESIS]))
    hyde = await hyde_pipe.retrieve(
        SPARSE_MATCH_QUERY, "kb", RetrievalMode.VECTOR, use_hyde=True
    )

    assert plain.chunks and hyde.chunks
    plain_order = [c.chunk.doc_id for c in plain.chunks]
    hyde_order = [c.chunk.doc_id for c in hyde.chunks]

    assert hyde.effective_query != SPARSE_MATCH_QUERY, "HyDE did not rewrite the dense query"
    assert plain_order != hyde_order, "HyDE made no measurable difference"
    assert plain_order[0] == "onboard", "baseline should be misled by vocabulary overlap"
    assert hyde_order[0] == "throttle", "HyDE should surface the passage that answers it"


async def test_hyde_mode_records_its_step(rag_settings, settings):
    pipe = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, [HYPOTHESIS]))

    result = await pipe.retrieve(SPARSE_MATCH_QUERY, "kb", RetrievalMode.HYDE)

    assert [s.name for s in result.steps] == ["hyde", "search", "rerank"]
    assert result.steps[0].detail.startswith("hypothesis:")


# --- CRAG ------------------------------------------------------------------


def test_crag_classifies_against_the_configured_thresholds(rag_settings, settings):
    crag = CorrectiveRag(make_gateway(settings, []), rag_settings)

    assert crag.classify([0.9, 0.2])[0] is Relevance.CORRECT
    assert crag.classify([0.45])[0] is Relevance.AMBIGUOUS
    assert crag.classify([0.1, 0.0])[0] is Relevance.INCORRECT
    assert crag.classify([])[0] is Relevance.INCORRECT


async def test_grader_falls_back_to_rerank_scores_when_the_llm_is_down(rag_settings, settings):
    pipe = await build(rag_settings, IRRELEVANT_CORPUS, make_gateway(settings, []))
    chunks = await pipe.retriever.search("kb", dense_query="dough", lexical_query="dough")
    for i, chunk in enumerate(chunks):
        chunk.rerank_score = 0.5 - i * 0.1

    gateway = LLMGateway(settings, providers=[], client=object(), backoff_base_s=0.0)
    grades, calibrated = await grade_chunks("dough", chunks, gateway)

    assert grades == [pytest.approx(c.rerank_score) for c in chunks]
    assert calibrated is False, "fallback grades are not on the LLM grade scale"


async def test_an_unavailable_grader_does_not_discard_the_retrieval(rag_settings, settings):
    """Reranker scores are scaled differently than the thresholds expect.

    Judging against them classified everything "incorrect" and returned nothing --
    a degraded grader must cost grading, not the results.
    """
    pipe = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, []))

    result = await pipe.retrieve("throttling and 429s", "kb", RetrievalMode.CRAG)

    assert result.chunks, "an ungradeable retrieval must still be returned"
    assert result.crag.action is Relevance.AMBIGUOUS
    assert "grader unavailable" in result.crag.detail
    assert result.action == "answer"


async def test_crag_triggers_the_fallback_path_on_an_irrelevant_corpus(rag_settings, settings):
    """Acceptance: nothing in the corpus answers the question -> corrective path."""
    grades = '{"grades": [{"index": 0, "score": 0.0}, {"index": 1, "score": 0.05}]}'
    rewrite = '{"query": "API request throttling and quota enforcement"}'
    pipe = await build(
        rag_settings, IRRELEVANT_CORPUS, make_gateway(settings, [grades, rewrite])
    )

    result = await pipe.retrieve("how do I rate limit an API?", "kb", RetrievalMode.CRAG)

    assert result.crag is not None
    assert result.crag.action is Relevance.INCORRECT
    assert result.crag.reformulated_query == "API request throttling and quota enforcement"
    assert result.action == "corrected"
    assert "crag" in [s.name for s in result.steps]


async def test_crag_keeps_good_retrieval_untouched(rag_settings, settings):
    grades = '{"grades": [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.8}]}'
    pipe = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, [grades]))

    result = await pipe.retrieve("throttling and 429 responses", "kb", RetrievalMode.CRAG)

    assert result.crag.action is Relevance.CORRECT
    assert result.crag.reformulated_query == ""
    assert result.action == "answer"
    assert result.chunks and result.chunks[0].relevance is Relevance.CORRECT


async def test_crag_uses_the_web_search_fallback_when_the_corpus_is_graded_incorrect(
    rag_settings, settings
):
    """`incorrect` means the corpus lacks the answer -- go outside it, if we can."""
    grades = '{"grades": [{"index": 0, "score": 0.0}, {"index": 1, "score": 0.0}]}'
    rewrite = '{"query": "API request throttling"}'
    pipe = await build(
        rag_settings, IRRELEVANT_CORPUS, make_gateway(settings, [grades, rewrite])
    )

    called: list[str] = []

    async def fake_web_search(query: str):
        called.append(query)
        return []

    pipe.crag.web_search = fake_web_search
    result = await pipe.retrieve("rate limiting", "kb", RetrievalMode.CRAG)

    assert called == ["API request throttling"], "should search externally, using the rewrite"
    assert result.crag.used_fallback is True


async def test_crag_does_not_reach_outside_when_retrieval_was_good(rag_settings, settings):
    grades = '{"grades": [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.9}]}'
    pipe = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, [grades]))

    called: list[str] = []

    async def fake_web_search(query: str):
        called.append(query)
        return []

    pipe.crag.web_search = fake_web_search
    result = await pipe.retrieve("throttling 429", "kb", RetrievalMode.CRAG)

    assert called == [], "a correct grade must not trigger an external search"
    assert result.crag.used_fallback is False


# --- Self-RAG --------------------------------------------------------------


async def test_self_rag_emits_retrieve_again_on_low_grounding(rag_settings, settings):
    """Acceptance: a mocked low-grounding generation triggers the signal."""
    ungrounded = (
        '{"grounded": false, "score": 0.15, "unsupported_claims": ["invented a quota API"],'
        ' "suggested_query": "token bucket burst capacity"}'
    )
    grounded = '{"grounded": true, "score": 0.92, "unsupported_claims": []}'
    pipe = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, [ungrounded, grounded]))
    self_rag = SelfRag(pipe.gateway, rag_settings)

    generations: list[str] = []

    async def generate(query: str, context: str) -> str:
        generations.append(context)
        return "The service enforces a quota via an undocumented internal API."

    chunks = await pipe.retriever.search("kb", dense_query="throttle", lexical_query="throttle")
    outcome = await self_rag.run(
        "how is throttling enforced?", chunks, generate, search_fn(pipe)
    )

    assert outcome.loops == 1, "should have retrieved again exactly once"
    assert len(generations) == 2, "answer must be regenerated after re-retrieval"
    assert outcome.queries == ["how is throttling enforced?", "token bucket burst capacity"]
    assert outcome.verdict.grounded is True


async def test_self_rag_accepts_a_well_grounded_answer_without_looping(rag_settings, settings):
    grounded = '{"grounded": true, "score": 0.95, "unsupported_claims": []}'
    pipe = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, [grounded]))
    self_rag = SelfRag(pipe.gateway, rag_settings)

    calls = 0

    async def generate(query: str, context: str) -> str:
        nonlocal calls
        calls += 1
        return "Token bucket throttling returns HTTP 429 beyond the allowance."

    chunks = await pipe.retriever.search("kb", dense_query="throttle", lexical_query="throttle")
    outcome = await self_rag.run("q", chunks, generate, search_fn(pipe))

    assert outcome.loops == 0
    assert calls == 1
    assert outcome.verdict.action == "accept"


async def test_self_rag_gives_up_after_the_loop_budget(rag_settings, settings):
    ungrounded = '{"grounded": false, "score": 0.1, "suggested_query": "again"}'
    tight = dataclasses.replace(rag_settings, self_rag_max_loops=1)
    pipe = await build(tight, HYDE_CORPUS, make_gateway(settings, [ungrounded] * 5))
    self_rag = SelfRag(pipe.gateway, tight)

    async def generate(query: str, context: str) -> str:
        return "Persistently unsupported claim about quantum quotas."

    chunks = await pipe.retriever.search("kb", dense_query="throttle", lexical_query="throttle")
    outcome = await self_rag.run("q", chunks, generate, search_fn(pipe))

    assert outcome.loops == 1, "must stop at the configured budget"
    assert outcome.verdict.grounded is False


async def test_self_rag_grounding_falls_back_to_lexical_overlap_offline(rag_settings, settings):
    gateway = LLMGateway(settings, providers=[], client=object(), backoff_base_s=0.0)
    pipe = await build(rag_settings, HYDE_CORPUS, gateway)
    self_rag = SelfRag(gateway, rag_settings)

    chunks = await pipe.retriever.search("kb", dense_query="throttle", lexical_query="throttle")
    verdict = await self_rag.assess("q", "Entirely unrelated vocabulary about marzipan.", chunks)

    assert "lexical grounding fallback" in verdict.detail
    assert verdict.grounded is False
    assert verdict.action == "retrieve_again"


async def test_self_rag_mode_without_a_generate_callable_warns(rag_settings, settings):
    grades = '{"grades": [{"index": 0, "score": 0.9}]}'
    pipe = await build(rag_settings, HYDE_CORPUS, make_gateway(settings, [grades]))

    result = await pipe.retrieve("throttling", "kb", RetrievalMode.SELF_RAG)

    assert any("generate callable" in w for w in result.warnings)
    assert result.grounding is None
