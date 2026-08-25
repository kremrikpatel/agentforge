"""FastAPI surface: run the pipeline, stream team coordination, serve the UI."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from agents.contracts import (
    STAGE_ORDER,
    PipelineReport,
    PipelineRequest,
    Stage,
)
from agents.graph import build_graph, initial_state, to_report
from agents.nodes import AgentDeps
from agentos.journal import Journal
from agentos.loader import AgentOS
from app.config import get_settings
from app.observability import EVENT_BUS, configure_logging, get_logger, log_event, timed
from app.tracing import configure_tracing, set_attributes, shutdown_tracing, span
from gateway.client import LLMGateway
from guardrails.engine import GuardrailEngine
from memory.cache import SemanticCache
from memory.ltm import LongTermMemory
from memory.stm import SessionMemory

logger = get_logger("agentforge.api")
STATIC_DIR = Path(__file__).parent / "static"


def resolve_stages(req: PipelineRequest, os_registry: AgentOS) -> tuple[Stage, ...]:
    """Explicit stages > command workflow > full pipeline, in canonical order."""
    if req.stages:
        wanted = set(req.stages)
        return tuple(s for s in STAGE_ORDER if s in wanted) or tuple(STAGE_ORDER)
    if req.command:
        cmd = os_registry.command(req.command)
        if cmd is None:
            raise HTTPException(
                status_code=404, detail=f"unknown command: {req.command}"
            )
        return tuple(cmd.ordered_stages())
    return tuple(STAGE_ORDER)


def get_graph(state, stages: tuple[Stage, ...]):
    """One compiled graph per stage subset; nodes share the app's deps."""
    cache: dict[tuple[Stage, ...], Any] = state.graphs
    if stages not in cache:
        cache[stages] = build_graph(
            AgentDeps(
                gateway=state.gateway,
                guardrails=state.guardrails,
                stm=state.stm,
                ltm=state.ltm,
                personas=state.os,
            ),
            stages=stages,
        )
    return cache[stages]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    # No-op unless OTEL_ENABLED and at least one backend credential are present.
    configure_tracing(service_name="agentforge-api")

    gateway = LLMGateway(settings)
    app.state.settings = settings
    app.state.gateway = gateway
    app.state.guardrails = GuardrailEngine(settings, gateway=gateway)
    app.state.stm = SessionMemory(settings)
    app.state.ltm = LongTermMemory(settings)
    app.state.cache = SemanticCache(settings)

    # OS layer: personas + kernel + commands, loaded from disk (hot-reloadable).
    app.state.os = AgentOS(Path(settings.os_dir))
    app.state.journal = Journal(Path(settings.os_dir)) if settings.os_journal else None
    app.state.graphs = {}
    app.state.graph = get_graph(app.state, tuple(STAGE_ORDER))

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
        shutdown_tracing()   # flush any spans still batched


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
    enabled = resolve_stages(req, state.os)
    full_run = enabled == tuple(STAGE_ORDER)

    # The semantic cache is keyed by topic alone, so only cacheable full runs
    # consult it: serving a cached partial report for a different stage subset
    # would lie about what ran.
    command = state.os.command(req.command) if req.command else None
    use_cache = full_run and not req.bypass_cache and not (
        command is not None and command.bypass_cache
    )

    if use_cache:
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

    EVENT_BUS.publish(
        run_id,
        {
            "type": "run_start",
            "run_id": run_id,
            "topic": req.topic,
            "stages": [s.value for s in enabled],
            **({"command": req.command} if req.command else {}),
        },
    )
    with span(
        "agentforge.pipeline.run",
        **{
            "agentforge.run_id": run_id,
            "agentforge.session_id": req.session_id,
            "gen_ai.system": "agentforge",
            "gen_ai.operation.name": "chain",
        },
    ) as run_span, timed() as t:
        graph = get_graph(request.app.state, enabled)
        final = await graph.ainvoke(initial_state(run_id, req.session_id, req.topic))
        report = to_report(final, expected=enabled)
        set_attributes(
            run_span,
            **{
                "agentforge.status": report.status,
                "agentforge.stages_completed": sum(
                    1 for s in (report.analysis, report.develop, report.test, report.deploy) if s
                ),
                "agentforge.guardrail_reports": len(report.guardrail_reports),
                "agentforge.errors": len(report.errors),
                "agentforge.latency_ms": report.total_latency_ms,
            },
        )

    if report.status == "completed":
        if full_run and not req.bypass_cache:
            await state.cache.set(req.topic, report.model_dump(mode="json"))
        if report.deploy is not None:
            await state.ltm.remember(
                session_id=req.session_id,
                run_id=run_id,
                topic=req.topic,
                stage="deploy",
                content=report.deploy.handoff.summary,
            )
        if (
            state.journal is not None
            and report.test is not None
            and report.test.verdict == "fail"
        ):
            state.journal.log_decision(
                "qa-fail-escalation",
                {"run_id": run_id, "topic": req.topic[:200]},
            )

    if state.journal is not None:
        state.journal.log_run(report.model_dump(mode="json"))

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


# --------------------------------------------------------------------------
# OS layer: registry inspection, kernel routing, command dispatch, reload.
# --------------------------------------------------------------------------


@app.get("/os/agents")
async def os_agents(request: Request) -> dict:
    """The persona registry as currently loaded."""
    return request.app.state.os.summary()


@app.get("/os/kernel")
async def os_kernel(request: Request) -> dict:
    state = request.app.state
    kernel = state.os.kernel
    return {
        "name": kernel.name,
        "identity": kernel.identity,
        "team_charter": kernel.team_charter,
        "routing": [r.model_dump(mode="json") for r in kernel.routing],
        "commands": [c.id for c in state.os.commands.values()],
        "sources": state.os.sources,
    }


class DispatchRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


@app.post("/os/dispatch")
async def os_dispatch(body: DispatchRequest, request: Request) -> dict:
    """Route a free-form task through the kernel and say what would run.

    Read-only: it returns the routing decision (stages/command) so a client
    can pass it to /pipeline/run; it never executes the pipeline itself.
    """
    decision = request.app.state.os.route(body.task)
    payload = decision.model_dump(mode="json")
    payload["stages"] = [s.value for s in decision.stages]
    return payload


@app.post("/os/reload")
async def os_reload(request: Request) -> dict:
    """Re-read kernel/personas/commands from disk into the running process."""
    state = request.app.state
    state.os.reload()
    # Persona charters may have changed: compiled graphs capture prompts only
    # at call time, but drop cached graphs anyway so subsets rebuild cleanly.
    state.graphs.clear()
    log_event(logger, "os.reloaded", agents=len(state.os.personas))
    return {"reloaded": True, "agents": len(state.os.personas), "commands": len(state.os.commands)}
