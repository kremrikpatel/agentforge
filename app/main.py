"""FastAPI surface: run the pipeline, stream team coordination, serve the UI."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import ValidationError

from agents.contracts import PipelineReport, PipelineRequest
from agents.graph import build_graph, initial_state, to_report
from agents.nodes import AgentDeps
from app.config import get_settings
from app.observability import EVENT_BUS, configure_logging, get_logger, log_event, timed
from gateway.client import LLMGateway
from guardrails.engine import GuardrailEngine
from memory.cache import SemanticCache
from memory.ltm import LongTermMemory
from memory.stm import SessionMemory

logger = get_logger("agentforge.api")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    gateway = LLMGateway(settings)
    app.state.settings = settings
    app.state.gateway = gateway
    app.state.guardrails = GuardrailEngine(settings, gateway=gateway)
    app.state.stm = SessionMemory(settings)
    app.state.ltm = LongTermMemory(settings)
    app.state.cache = SemanticCache(settings)
    app.state.graph = build_graph(
        AgentDeps(
            gateway=gateway,
            guardrails=app.state.guardrails,
            stm=app.state.stm,
            ltm=app.state.ltm,
        )
    )

    # Backends are optional: the pipeline degrades rather than refusing to start.
    schema_ok = await app.state.ltm.ensure_schema()
    redis_ok = await app.state.stm.ping()
    log_event(
        logger,
        "startup",
        postgres=schema_ok,
        redis=redis_ok,
        providers=[p.name for p in gateway.providers if p.available()],
    )
    try:
        yield
    finally:
        await gateway.aclose()
        await app.state.stm.aclose()
        await app.state.cache.aclose()


app = FastAPI(
    title="AgentForge",
    version="0.1.0",
    description="Phase 1: four-agent LangGraph pipeline with gateway, guardrails and memory.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/ui")


@app.get("/ui", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "ui.html")


@app.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state
    redis_ok = await state.stm.ping()
    return {
        "status": "ok",
        "redis": "up" if redis_ok else "degraded",
        "postgres": "degraded" if state.ltm.degraded else "up",
        "providers": [p.name for p in state.gateway.providers if p.available()],
    }


@app.post("/pipeline/run", response_model=PipelineReport)
async def run_pipeline(req: PipelineRequest, request: Request) -> PipelineReport:
    state = request.app.state
    run_id = req.run_id

    if not req.bypass_cache:
        cached, kind, similarity = await state.cache.get(req.topic)
        if cached is not None:
            cached["cached"] = True
            EVENT_BUS.publish(
                run_id, {"type": "cache_hit", "kind": kind, "similarity": similarity}
            )
            log_event(logger, "pipeline.cache_hit", run_id=run_id, kind=kind)
            try:
                return PipelineReport.model_validate(cached)
            except ValidationError:
                # Stale shape from an older build -- drop it and run for real.
                await state.cache.invalidate(req.topic)

    EVENT_BUS.publish(run_id, {"type": "run_start", "run_id": run_id, "topic": req.topic})
    with timed() as t:
        final = await state.graph.ainvoke(initial_state(run_id, req.session_id, req.topic))
    report = to_report(final)

    if report.status == "completed":
        await state.cache.set(req.topic, report.model_dump(mode="json"))
        if report.deploy is not None:
            await state.ltm.remember(
                session_id=req.session_id,
                run_id=run_id,
                topic=req.topic,
                stage="deploy",
                content=report.deploy.handoff.summary,
            )

    EVENT_BUS.publish(
        run_id,
        {"type": "run_complete", "status": report.status, "latency_ms": t["ms"]},
    )
    log_event(
        logger,
        "pipeline.completed",
        run_id=run_id,
        status=report.status,
        wall_ms=t["ms"],
        stages=sum(
            1 for s in (report.analysis, report.develop, report.test, report.deploy) if s
        ),
    )
    return report


@app.get("/pipeline/stream/{run_id}")
async def stream(run_id: str, request: Request) -> StreamingResponse:
    """SSE feed of team coordination events. Replays history, then tails live."""
    queue = EVENT_BUS.subscribe(run_id)

    async def events():
        try:
            for event in EVENT_BUS.replay(run_id):
                yield f"data: {json.dumps(event, default=str)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # keep proxies from closing the stream
                    continue
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("type") in {"run_complete", "cache_hit"}:
                    break
        finally:
            EVENT_BUS.unsubscribe(run_id, queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/pipeline/session/{session_id}")
async def session_history(session_id: str, request: Request) -> dict:
    turns = await request.app.state.stm.history(session_id, limit=50)
    if not turns:
        raise HTTPException(status_code=404, detail="no history for this session")
    return {"session_id": session_id, "turns": turns}
