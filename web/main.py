"""The admin console: pages plus the backend-for-frontend it needs.

Everything here lives in web/. The three endpoint groups that have no upstream
equivalent -- agent configs, run history, knowledge ingestion -- are implemented
in this app rather than bolted onto Phase 1, so no existing module changes.

    python -m web.main
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.observability import configure_logging, get_logger, log_event
from web.auth import (
    SESSIONS,
    Session,
    clear_cookie,
    current_session,
    issue_cookie,
    require_admin,
)
from web.backend import (
    BackendClient,
    BackendError,
    ingest_documents,
    redteam_failures,
    redteam_summary,
)
from web.config import get_web_settings
from web.schemas import (
    STAGE_NAMES,
    AgentConfig,
    AgentConfigInput,
    IngestRequest,
    IngestResult,
    RunRecord,
    RunRequest,
    _now,
    summarize_report,
)
from web.store import build_store

logger = get_logger("agentforge.web")
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(HERE / "templates"))


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(max_length=200)
    password: str = Field(max_length=400)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_web_settings()
    configure_logging(get_settings().log_level)

    app.state.settings = settings
    app.state.store = await build_store(get_settings())
    app.state.backend = BackendClient(settings)

    if not settings.auth_configured:
        log_event(
            logger,
            "web.startup_warning",
            warning="WEB_ADMIN_PASSWORD is unset; every login will be refused",
        )
    log_event(
        logger,
        "web.startup",
        store=getattr(app.state.store, "name", "unknown"),
        backend=settings.backend_url,
    )
    try:
        yield
    finally:
        await app.state.backend.aclose()


app = FastAPI(
    title="AgentForge Console",
    version="0.1.0",
    description="Admin UI for agent configuration, pipeline runs, and red-team results.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root(request: Request) -> RedirectResponse:
    return RedirectResponse("/app" if current_session(request) else "/login")


@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    if current_session(request):
        return RedirectResponse("/app")
    return templates.TemplateResponse(
        request, "login.html", {"configured": get_web_settings().auth_configured}
    )


@app.get("/app", include_in_schema=False)
async def console(request: Request):
    session = current_session(request)
    if session is None:
        return RedirectResponse("/login")
    return templates.TemplateResponse(
        request,
        "app.html",
        {
            "user": session.user,
            "stages": STAGE_NAMES,
            "redteam_url": get_web_settings().redteam_url,
        },
    )


@app.get("/static/{name}", include_in_schema=False)
async def static_file(name: str) -> FileResponse:
    # Explicit allow-list rather than a mount: keeps path traversal impossible.
    if name not in {"app.css", "app.js"}:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(HERE / "static" / name)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


@app.post("/api/login")
async def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
    settings = get_web_settings()
    if not settings.auth_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin auth is not configured; set WEB_ADMIN_PASSWORD",
        )

    session = SESSIONS.authenticate(payload.username, payload.password)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials"
        )

    issue_cookie(response, session, settings)
    return {"user": session.user}


@app.post("/api/logout")
async def logout(request: Request, response: Response) -> dict[str, Any]:
    settings = get_web_settings()
    SESSIONS.revoke(request.cookies.get(settings.cookie_name))
    clear_cookie(response, settings)
    return {"ok": True}


@app.get("/api/health")
async def health(request: Request, _: Session = Depends(require_admin)) -> dict[str, Any]:
    state = request.app.state
    return {
        "console": "ok",
        "store": getattr(state.store, "name", "unknown"),
        "pipeline": await state.backend.health(),
    }


# --------------------------------------------------------------------------
# Agent configuration  [GAP 1: nothing upstream persists or serves these]
# --------------------------------------------------------------------------


@app.get("/api/configs")
async def list_configs(request: Request, _: Session = Depends(require_admin)) -> dict[str, Any]:
    configs = await request.app.state.store.list_configs()
    return {"configs": [c.model_dump(mode="json") for c in configs]}


@app.post("/api/configs", status_code=status.HTTP_201_CREATED)
async def create_config(
    payload: AgentConfigInput, request: Request, _: Session = Depends(require_admin)
) -> dict[str, Any]:
    config = AgentConfig(**payload.model_dump())
    if not config.stages:
        # A config with no stages is unusable; seed one entry per pipeline stage.
        config = AgentConfig.blank(config.name).model_copy(
            update={"description": config.description}
        )
    await request.app.state.store.save_config(config)
    log_event(logger, "web.config_created", config_id=config.id, name=config.name)
    return config.model_dump(mode="json")


@app.get("/api/configs/{config_id}")
async def get_config(
    config_id: str, request: Request, _: Session = Depends(require_admin)
) -> dict[str, Any]:
    config = await request.app.state.store.get_config(config_id)
    if config is None:
        raise HTTPException(status_code=404, detail="no such agent config")
    return config.model_dump(mode="json")


@app.put("/api/configs/{config_id}")
async def update_config(
    config_id: str,
    payload: AgentConfigInput,
    request: Request,
    _: Session = Depends(require_admin),
) -> dict[str, Any]:
    existing = await request.app.state.store.get_config(config_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="no such agent config")

    # Rebuilt through the constructor rather than model_copy(update=...): the
    # payload's nested items are plain dicts, and only the constructor coerces
    # them back into StagePrompt/KnowledgeRef/ActionRef models.
    updated = AgentConfig(
        **payload.model_dump(),
        # Identity and creation time belong to the server, not the client.
        id=existing.id,
        created_at=existing.created_at,
        version=existing.version + 1,
        updated_at=_now(),
    )
    await request.app.state.store.save_config(updated)
    log_event(logger, "web.config_updated", config_id=config_id, version=updated.version)
    return updated.model_dump(mode="json")


# --------------------------------------------------------------------------
# Runs  [GAP 2: upstream keeps no listable run history]
# --------------------------------------------------------------------------


@app.post("/api/runs")
async def trigger_run(
    payload: RunRequest, request: Request, _: Session = Depends(require_admin)
) -> dict[str, Any]:
    state = request.app.state

    config_name = ""
    if payload.agent_config_id:
        config = await state.store.get_config(payload.agent_config_id)
        if config is None:
            raise HTTPException(status_code=404, detail="no such agent config")
        config_name = config.name

    try:
        report = await state.backend.run_pipeline(payload.topic, payload.run_id)
    except BackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    record = summarize_report(
        report, agent_config_id=payload.agent_config_id, agent_config_name=config_name
    )
    await state.store.save_run(record)
    return record.model_dump(mode="json")


@app.get("/api/runs")
async def list_runs(
    request: Request, limit: int = 50, _: Session = Depends(require_admin)
) -> dict[str, Any]:
    return {"runs": await request.app.state.store.list_runs(min(limit, 200))}


@app.get("/api/runs/{run_id}")
async def get_run(
    run_id: str, request: Request, _: Session = Depends(require_admin)
) -> RunRecord:
    record = await request.app.state.store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no such run")
    return record


@app.get("/api/runs/{run_id}/stream")
async def stream_run(
    run_id: str, request: Request, _: Session = Depends(require_admin)
) -> StreamingResponse:
    """Relay Phase 1's SSE progress so the console is the only origin in play."""
    return StreamingResponse(
        request.app.state.backend.stream(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Knowledge  [GAP 3: RAG ingestion has no REST surface upstream]
# --------------------------------------------------------------------------


@app.post("/api/knowledge/ingest")
async def ingest(payload: IngestRequest, _: Session = Depends(require_admin)) -> IngestResult:
    try:
        return await ingest_documents(payload.kb_id, payload.documents)
    except Exception as exc:  # noqa: BLE001 -- surface the reason, do not 500 blankly
        log_event(logger, "web.ingest_failed", kb_id=payload.kb_id, error=str(exc)[:300])
        raise HTTPException(
            status_code=502, detail=f"ingestion failed: {exc}"[:300]
        ) from exc


# --------------------------------------------------------------------------
# Red team (Phase 3 data, read-only)
# --------------------------------------------------------------------------


@app.get("/api/redteam")
async def redteam(_: Session = Depends(require_admin)) -> dict[str, Any]:
    return await redteam_summary()


@app.get("/api/redteam/{run_id}/failures")
async def redteam_run_failures(
    run_id: str, _: Session = Depends(require_admin)
) -> dict[str, Any]:
    return {"run_id": run_id, "failures": await redteam_failures(run_id)}


# Phase 5 inspection endpoints (/trajectories, /audit-log). Mounted here rather
# than served separately so they inherit this app's session auth.
from instrumentation.api import router as instrumentation_router  # noqa: E402

app.include_router(instrumentation_router)


@app.exception_handler(HTTPException)
async def _json_errors(request: Request, exc: HTTPException) -> Response:
    """Unauthenticated page loads bounce to login; API calls get JSON.

    The prefixes are listed explicitly because "not under /api/" is not the same
    question as "is this a JSON API": the Phase 5 inspection endpoints sit at
    /trajectories and /audit-log, and redirecting those to a login page would
    hand an API client an HTML redirect instead of a 401.
    """
    json_api = ("/api/", "/trajectories", "/audit-log")
    unauthenticated_page = (
        exc.status_code == status.HTTP_401_UNAUTHORIZED
        and not request.url.path.startswith(json_api)
    )
    if unauthenticated_page:
        return RedirectResponse("/login")
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


def main() -> None:
    import uvicorn

    settings = get_web_settings()
    configure_logging(get_settings().log_level)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
