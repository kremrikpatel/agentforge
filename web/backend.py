"""Everything the console needs from the rest of the platform.

Three different integration styles, each chosen because it is the cheapest one
that works:

  Phase 1  over HTTP. It is a running service with a stable contract, and the
           console may well be on a different host.
  Phase 2  by direct Python import. RAG has no REST surface, and adding one to
           rag/ would mean editing a module this phase must not touch.
  Phase 3  by reading its store. Same reason, and it avoids requiring the
           red-team dashboard process to be running just to show its numbers.

None of those modules are modified; they are imported and called.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from app.observability import get_logger, log_event, timed
from web.config import WebSettings, get_web_settings
from web.schemas import IngestDocument, IngestResult

logger = get_logger("agentforge.web.backend")


class BackendError(RuntimeError):
    """The upstream pipeline could not be reached or refused the request."""


class BackendClient:
    """HTTP client for the Phase 1 pipeline API."""

    def __init__(self, settings: WebSettings | None = None, client=None) -> None:
        self.settings = settings or get_web_settings()
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.backend_timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health(self) -> dict[str, Any]:
        try:
            resp = await self.client.get(self.settings.health_endpoint, timeout=5)
            return {"reachable": resp.status_code == 200, **resp.json()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"reachable": False, "error": str(exc)[:200]}

    async def run_pipeline(
        self, topic: str, run_id: str, session_id: str = ""
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"topic": topic, "run_id": run_id}
        if session_id:
            payload["session_id"] = session_id
        try:
            with timed() as t:
                resp = await self.client.post(self.settings.run_endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise BackendError(f"pipeline unreachable: {exc}") from exc

        if resp.status_code >= 400:
            raise BackendError(f"pipeline returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            report = resp.json()
        except ValueError as exc:
            raise BackendError("pipeline returned a non-JSON body") from exc

        log_event(logger, "web.run_proxied", run_id=run_id, wall_ms=t["ms"])
        return report

    async def stream(self, run_id: str) -> AsyncIterator[bytes]:
        """Relay the upstream SSE feed.

        Proxied rather than connected to directly from the browser: the pipeline
        API is not necessarily reachable from the client, and this keeps the
        console the single origin the admin talks to.
        """
        url = f"{self.settings.stream_endpoint}/{run_id}"
        try:
            async with self.client.stream("GET", url, timeout=None) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            log_event(logger, "web.stream_failed", run_id=run_id, error=str(exc)[:200])
            yield b'data: {"type": "stream_error"}\n\n'


# --------------------------------------------------------------------------
# Phase 2 -- RAG ingestion (Python API; there is no REST surface)
# --------------------------------------------------------------------------


async def ingest_documents(kb_id: str, documents: list[IngestDocument]) -> IngestResult:
    from gateway.client import LLMGateway
    from rag.pipeline import RagPipeline
    from rag.schemas import Document

    gateway = LLMGateway()
    pipeline = RagPipeline(gateway)
    try:
        chunks = await pipeline.ingest(
            kb_id,
            [Document(text=d.text, title=d.title, source=d.source) for d in documents],
        )
    finally:
        await gateway.aclose()

    log_event(logger, "web.ingested", kb_id=kb_id, documents=len(documents), chunks=chunks)
    return IngestResult(kb_id=kb_id, documents=len(documents), chunks=chunks)


# --------------------------------------------------------------------------
# Phase 3 -- red-team results (read its store directly)
# --------------------------------------------------------------------------


async def redteam_summary(limit: int = 10) -> dict[str, Any]:
    """Latest run's per-category rates, plus the recent run list."""
    from redteam.store import build_store, rates_from_rows

    store = await build_store()
    runs = await store.recent_runs(limit)
    if not runs:
        return {"available": False, "runs": [], "latest": None, "trend": []}

    latest = runs[0]
    previous = next((r for r in runs[1:] if r["target"] == latest["target"]), None)
    trend = []
    if previous is not None:
        before = rates_from_rows(previous["categories"])
        for row in latest["categories"]:
            prior = before.get(row["category"])
            trend.append(
                {
                    "category": row["category"],
                    "current": row["block_rate"],
                    "previous": prior,
                    "delta": None if prior is None else round(row["block_rate"] - prior, 4),
                }
            )

    return {
        "available": True,
        "backend": getattr(store, "name", "unknown"),
        "runs": runs,
        "latest": latest,
        "trend": trend,
    }


async def redteam_failures(run_id: str, limit: int = 20) -> list[dict[str, Any]]:
    from redteam.store import build_store

    store = await build_store()
    attempts = await store.attempts_for(run_id)
    return [a for a in attempts if a.get("outcome") == "leaked"][:limit]
