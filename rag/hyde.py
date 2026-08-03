"""HyDE -- Hypothetical Document Embeddings.

A question and its answer live in different regions of embedding space. HyDE
asks the model to write the answer it *expects*, then embeds that instead, so
the query vector lands among real answers. It earns its keep exactly when a
question shares little vocabulary with the passage that answers it.

The rewrite is applied to the dense query only. A hypothetical document is a
poor BM25 query -- it invents vocabulary the corpus may not contain -- so the
lexical half keeps the user's original words.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.jsonio import extract_json
from app.observability import get_logger, log_event
from gateway.client import AllProvidersFailed, LLMGateway
from gateway.providers import GatewayRequest
from memory.embedding import tokenize
from rag.config import RagSettings, get_rag_settings

logger = get_logger("agentforge.rag.hyde")

HYDE_SYSTEM = """You write a short hypothetical passage that would answer the user's question.

Write it as if it were an excerpt from documentation or a reference article: plain
declarative sentences, concrete terminology, no hedging, no preamble. Being factually
wrong is acceptable -- the passage is used only as a retrieval probe, never shown to
anyone. Using the vocabulary such a document would use is what matters.

Reply with JSON only: {"hypothesis": "<2-4 sentences>"}"""


@dataclass(frozen=True)
class HydeResult:
    original_query: str
    hypothesis: str
    dense_query: str
    provider: str = ""
    used: bool = False
    detail: str = ""


def blend_query(query: str, hypothesis: str, weight: float) -> str:
    """Repeat the query so it holds roughly `weight` of the blended token mass.

    The embedder is bag-of-words, so repetition *is* weighting. This keeps a
    hallucinated hypothesis from dragging retrieval entirely off-topic.
    """
    if not hypothesis.strip():
        return query
    if weight <= 0:
        return hypothesis
    if weight >= 1:
        return query

    query_tokens = max(1, len(tokenize(query)))
    hypothesis_tokens = max(1, len(tokenize(hypothesis)))
    # Solve r*qt / (r*qt + ht) = weight  for the repeat count r.
    repeats = max(1, math.ceil((weight * hypothesis_tokens) / ((1 - weight) * query_tokens)))
    return "\n".join([query] * repeats + [hypothesis])


async def build_hyde_query(
    query: str, gateway: LLMGateway, rag_settings: RagSettings | None = None
) -> HydeResult:
    """Generate a hypothesis and return the dense query to search with.

    Never raises: if the model is unreachable or replies with nonsense, HyDE
    silently degrades to the plain query.
    """
    rag = rag_settings or get_rag_settings()
    try:
        resp = await gateway.complete(
            GatewayRequest(
                system=HYDE_SYSTEM,
                user=f"Question:\n<<<{query}>>>",
                max_tokens=rag.hyde_max_tokens,
                temperature=0.3,
                stub_response='{"hypothesis": ""}',
            )
        )
    except AllProvidersFailed as exc:
        log_event(logger, "rag.hyde_failed", error=str(exc)[:200])
        return HydeResult(query, "", query, detail="no provider available")

    payload = extract_json(resp.text) or {}
    hypothesis = str(payload.get("hypothesis", "")).strip()
    if not hypothesis:
        # Some models ignore the JSON instruction; the raw prose is still usable.
        hypothesis = resp.text.strip() if not resp.text.strip().startswith("{") else ""
    if not hypothesis:
        return HydeResult(query, "", query, resp.provider, detail="empty hypothesis")

    dense_query = blend_query(query, hypothesis, rag.hyde_query_weight)
    log_event(
        logger,
        "rag.hyde",
        provider=resp.provider,
        hypothesis_chars=len(hypothesis),
        latency_ms=resp.latency_ms,
    )
    return HydeResult(query, hypothesis, dense_query, resp.provider, used=True)
