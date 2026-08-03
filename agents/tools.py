"""LangGraph tool binding for the RAG subsystem.

This is the only Phase 2 addition under agents/. Nothing in nodes.py, graph.py,
or contracts.py changes -- binding a tool is additive, so an agent that does not
ask for retrieval behaves exactly as it did in Phase 1.

`Text2Sql.execute` is deliberately NOT exposed as a tool. Agents may propose SQL;
approving it is a human action, and a tool an agent can call is by definition not
a human approval.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from rag.pipeline import RagPipeline, get_pipeline
from rag.schemas import RetrievalMode, RetrievalResult

TOOL_DESCRIPTION = """Search the knowledge base for passages that support an answer.

Modes:
  hybrid    (default) dense + keyword search, fused and reranked
  vector    semantic similarity only
  lexical   exact keyword matching only -- best for names, codes, error strings
  hyde      for abstract questions whose wording is unlikely to appear verbatim
  crag      verifies the passages are actually relevant and self-corrects if not
  text2sql  turn the question into SQL over the configured schema

Returns passages with [citation] markers. Cite them when you use them.
text2sql returns SQL for a human to approve -- it never runs the query."""


class RetrieveArgs(BaseModel):
    query: str = Field(description="What to look for, in natural language.")
    kb_id: str = Field(default="default", description="Knowledge base to search.")
    mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID,
        description="Retrieval strategy; see the tool description.",
    )


def make_retrieval_tool(
    pipeline: RagPipeline | None = None,
    *,
    name: str = "search_knowledge_base",
    default_kb_id: str = "default",
    allowed_modes: set[RetrievalMode] | None = None,
) -> StructuredTool:
    """Build the tool.

    Pass a configured pipeline for per-tenant stores or a custom SQL schema; omit
    it to use the process-wide default. `allowed_modes` caps what an agent may
    ask for -- useful to keep text2sql or the more expensive modes out of reach
    of a particular node.
    """
    pipe = pipeline or get_pipeline()

    async def _retrieve(
        query: str,
        kb_id: str = default_kb_id,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
    ) -> tuple[str, dict[str, Any]]:
        requested = RetrievalMode(mode)
        if allowed_modes is not None and requested not in allowed_modes:
            requested = RetrievalMode.HYBRID

        result: RetrievalResult = await pipe.retrieve(query, kb_id, requested)
        # content goes into the agent's prompt; artifact keeps the full object
        # (scores, citations, per-stage trace, SQL proposal) for the caller.
        return result.as_tool_output(), result.model_dump(mode="json")

    return StructuredTool.from_function(
        coroutine=_retrieve,
        name=name,
        description=TOOL_DESCRIPTION,
        args_schema=RetrieveArgs,
        response_format="content_and_artifact",
    )
