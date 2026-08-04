"""The console's backend-for-frontend: auth gate, configs, runs, knowledge, red team."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from web.auth import SESSIONS
from web.config import WebSettings
from web.main import app
from web.schemas import summarize_report
from web.store import InMemoryWebStore

PASSWORD = "correct-horse-battery-staple"

REPORT: dict[str, Any] = {
    "run_id": "run-1",
    "topic": "Design a rate limiter",
    "status": "completed",
    "created_at": "2026-08-04T15:05:00Z",
    "total_latency_ms": 1234.5,
    "analysis": {
        "agent": "Ana (Analyst)",
        "confidence": 0.8,
        "handoff": {"summary": "Framed the problem for Dev."},
    },
    "develop": {
        "agent": "Dev (Solution Architect)",
        "confidence": 0.7,
        "handoff": {"summary": "Token bucket design."},
    },
    "traces": [
        {"stage": "analysis", "provider": "anthropic", "latency_ms": 400.0},
        {"stage": "develop", "provider": "anthropic", "latency_ms": 800.0},
    ],
    "guardrail_reports": [
        {"action": "allow", "direction": "input"},
        {"action": "redact", "direction": "input", "risk_score": 0.5},
    ],
    "errors": [],
}


class StubPipeline:
    """Stands in for the Phase 1 API."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = False

    async def run_pipeline(
        self, topic: str, run_id: str, session_id: str = ""
    ) -> dict[str, Any]:
        from web.backend import BackendError

        if self.fail:
            raise BackendError("pipeline unreachable: connection refused")
        self.calls.append({"topic": topic, "run_id": run_id})
        return {**REPORT, "run_id": run_id, "topic": topic}

    async def health(self) -> dict[str, Any]:
        return {"reachable": True, "status": "ok"}

    async def aclose(self) -> None:
        return None


@pytest.fixture
def web_settings(monkeypatch) -> WebSettings:
    monkeypatch.setenv("WEB_ADMIN_USER", "admin")
    monkeypatch.setenv("WEB_ADMIN_PASSWORD", PASSWORD)
    settings = WebSettings()
    # SESSIONS captured its settings at import time; point it at this test's.
    SESSIONS.settings = settings
    SESSIONS._sessions.clear()
    return settings


@pytest.fixture
async def client(web_settings):
    async with app.router.lifespan_context(app):
        app.state.store = InMemoryWebStore()
        app.state.backend = StubPipeline()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://console") as c:
            yield c


async def login(client: httpx.AsyncClient, password: str = PASSWORD) -> httpx.Response:
    return await client.post("/api/login", json={"username": "admin", "password": password})


# --- auth ------------------------------------------------------------------


async def test_protected_endpoints_reject_anonymous_callers(client):
    for path in ("/api/configs", "/api/runs", "/api/health", "/api/redteam"):
        resp = await client.get(path)
        assert resp.status_code == 401, path


async def test_login_with_wrong_password_is_refused(client):
    assert (await login(client, "wrong")).status_code == 401
    assert (await client.get("/api/configs")).status_code == 401


async def test_login_issues_an_httponly_session_cookie(client, web_settings):
    resp = await login(client)

    assert resp.status_code == 200
    assert resp.json()["user"] == "admin"
    cookie = resp.headers.get("set-cookie", "")
    assert web_settings.cookie_name in cookie
    assert "httponly" in cookie.lower(), "session cookie must not be readable from JS"
    assert "samesite=lax" in cookie.lower()
    assert (await client.get("/api/configs")).status_code == 200


async def test_logout_revokes_the_session_immediately(client):
    await login(client)
    assert (await client.get("/api/configs")).status_code == 200

    await client.post("/api/logout")

    assert (await client.get("/api/configs")).status_code == 401


async def test_login_is_refused_when_no_password_is_configured(client, monkeypatch):
    monkeypatch.setenv("WEB_ADMIN_PASSWORD", "")
    SESSIONS.settings = WebSettings()

    resp = await login(client)

    assert resp.status_code == 503
    assert "WEB_ADMIN_PASSWORD" in resp.json()["detail"]


async def test_unauthenticated_page_load_redirects_to_login(client):
    resp = await client.get("/app", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/login"


# --- agent configs [GAP 1] -------------------------------------------------


async def test_create_config_seeds_one_entry_per_pipeline_stage(client):
    await login(client)

    resp = await client.post("/api/configs", json={"name": "Support agent"})

    assert resp.status_code == 201
    config = resp.json()
    assert config["name"] == "Support agent"
    assert [s["stage"] for s in config["stages"]] == ["analysis", "develop", "test", "deploy"]
    assert config["version"] == 1


async def test_config_round_trips_instructions_knowledge_and_actions(client):
    await login(client)
    payload = {
        "name": "Ops agent",
        "description": "handles runbooks",
        "stages": [
            {"stage": "analysis", "instructions": "Be terse.", "notes": "v2"},
            {"stage": "develop", "instructions": "Prefer boring designs."},
        ],
        "knowledge": [{"kb_id": "handbook", "label": "Handbook"}],
        "actions": [
            {"name": "create_ticket", "method": "POST", "endpoint": "https://api.example.test/t"}
        ],
    }

    created = (await client.post("/api/configs", json=payload)).json()
    fetched = (await client.get(f"/api/configs/{created['id']}")).json()

    assert fetched["stages"][0]["instructions"] == "Be terse."
    assert fetched["knowledge"][0]["kb_id"] == "handbook"
    assert fetched["actions"][0]["name"] == "create_ticket"
    assert (await client.get("/api/configs")).json()["configs"][0]["id"] == created["id"]


async def test_update_bumps_version_and_keeps_server_owned_fields(client):
    await login(client)
    created = (await client.post("/api/configs", json={"name": "First"})).json()

    updated = (
        await client.put(
            f"/api/configs/{created['id']}",
            json={"name": "Renamed", "stages": [{"stage": "analysis", "instructions": "New."}]},
        )
    ).json()

    assert updated["name"] == "Renamed"
    assert updated["version"] == 2
    assert updated["id"] == created["id"], "client must not be able to change the id"
    assert updated["created_at"] == created["created_at"]


async def test_unknown_config_is_a_404(client):
    await login(client)
    assert (await client.get("/api/configs/nope")).status_code == 404
    assert (await client.put("/api/configs/nope", json={"name": "x"})).status_code == 404


# --- runs [GAP 2] ----------------------------------------------------------


async def test_triggering_a_run_proxies_upstream_and_records_it(client):
    await login(client)

    record = (await client.post("/api/runs", json={"topic": "Design a rate limiter"})).json()

    assert record["status"] == "completed"
    assert app.state.backend.calls[0]["topic"] == "Design a rate limiter"
    # Four stage slots always, two of them reached by this report.
    assert len(record["stages"]) == 4
    assert [s["stage"] for s in record["stages"] if s["reached"]] == ["analysis", "develop"]
    assert record["stages"][0]["handoff"] == "Framed the problem for Dev."
    # Only non-allow guardrail reports count as interventions.
    assert len(record["guardrail_interventions"]) == 1


async def test_run_appears_in_history_and_detail(client):
    await login(client)
    record = (await client.post("/api/runs", json={"topic": "Cache design"})).json()

    listing = (await client.get("/api/runs")).json()["runs"]
    detail = (await client.get(f"/api/runs/{record['run_id']}")).json()

    assert [r["run_id"] for r in listing] == [record["run_id"]]
    assert listing[0]["stages_reached"] == 2
    assert listing[0]["interventions"] == 1
    assert detail["topic"] == "Cache design"
    assert detail["report"]["status"] == "completed", "full report kept for the detail view"


async def test_run_against_an_unknown_config_is_a_404(client):
    await login(client)
    resp = await client.post("/api/runs", json={"topic": "x", "agent_config_id": "nope"})
    assert resp.status_code == 404


async def test_upstream_failure_surfaces_as_502_not_500(client):
    await login(client)
    app.state.backend.fail = True

    resp = await client.post("/api/runs", json={"topic": "x"})

    assert resp.status_code == 502
    assert "unreachable" in resp.json()["detail"]


async def test_unknown_run_is_a_404(client):
    await login(client)
    assert (await client.get("/api/runs/nope")).status_code == 404


# --- knowledge [GAP 3] -----------------------------------------------------


async def test_ingest_calls_the_rag_pipeline_and_reports_chunks(client, monkeypatch):
    await login(client)
    seen: dict[str, Any] = {}

    async def fake_ingest(kb_id, documents):
        from web.schemas import IngestResult

        seen["kb_id"] = kb_id
        seen["documents"] = documents
        return IngestResult(kb_id=kb_id, documents=len(documents), chunks=7)

    monkeypatch.setattr("web.main.ingest_documents", fake_ingest)

    resp = await client.post(
        "/api/knowledge/ingest",
        json={"kb_id": "handbook", "documents": [{"text": "Rate limits protect an API."}]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert (body["kb_id"], body["documents"], body["chunks"]) == ("handbook", 1, 7)
    assert seen["kb_id"] == "handbook"


async def test_ingest_failure_surfaces_as_502(client, monkeypatch):
    await login(client)

    async def boom(kb_id, documents):
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr("web.main.ingest_documents", boom)

    resp = await client.post(
        "/api/knowledge/ingest", json={"kb_id": "kb", "documents": [{"text": "hello"}]}
    )

    assert resp.status_code == 502
    assert "qdrant unreachable" in resp.json()["detail"]


async def test_ingest_rejects_an_empty_document_set(client):
    await login(client)
    resp = await client.post("/api/knowledge/ingest", json={"kb_id": "kb", "documents": []})
    assert resp.status_code == 422


# --- red team --------------------------------------------------------------


async def test_redteam_panel_reports_when_there_are_no_runs(client, monkeypatch):
    await login(client)

    async def empty(limit: int = 10):
        return {"available": False, "runs": [], "latest": None, "trend": []}

    monkeypatch.setattr("web.main.redteam_summary", empty)

    assert (await client.get("/api/redteam")).json()["available"] is False


async def test_redteam_panel_surfaces_categories_and_trend(client, monkeypatch):
    await login(client)

    async def summary(limit: int = 10):
        return {
            "available": True,
            "latest": {
                "run_id": "rt-2",
                "categories": [{"category": "jailbreak", "block_rate": 0.8, "blocked": 4}],
            },
            "trend": [{"category": "jailbreak", "current": 0.8, "previous": 1.0, "delta": -0.2}],
            "runs": [],
        }

    monkeypatch.setattr("web.main.redteam_summary", summary)

    body = (await client.get("/api/redteam")).json()

    assert body["latest"]["categories"][0]["block_rate"] == 0.8
    assert body["trend"][0]["delta"] == -0.2


# --- store -----------------------------------------------------------------


async def test_in_memory_store_orders_runs_newest_first():
    store = InMemoryWebStore()
    await store.save_run(summarize_report({**REPORT, "run_id": "a", "topic": "first"}))
    await store.save_run(summarize_report({**REPORT, "run_id": "b", "topic": "second"}))

    assert [r["run_id"] for r in await store.list_runs()] == ["b", "a"]
