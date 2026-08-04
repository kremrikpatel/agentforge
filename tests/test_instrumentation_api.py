"""The inspection endpoints, mounted into the Phase 4 console and gated by its auth."""

from __future__ import annotations

import httpx
import pytest

from instrumentation.api import Components, set_components
from instrumentation.audit import AuditLog
from instrumentation.recorder import TrajectoryRecorder
from instrumentation.store import InMemoryInstrumentationStore
from instrumentation.versioning import VersionStore
from tests.test_instrumentation_scrubbing import EMAIL, REPORT_WITH_PII
from tests.test_web_api import PASSWORD, StubPipeline
from web.auth import SESSIONS
from web.config import WebSettings
from web.main import app
from web.store import InMemoryWebStore


@pytest.fixture
def components():
    store = InMemoryInstrumentationStore()
    bundle = Components(
        store=store,
        recorder=TrajectoryRecorder(store),
        audit=AuditLog(store),
        versions=VersionStore(store),
    )
    set_components(bundle)
    yield bundle
    set_components(None)


@pytest.fixture
async def client(monkeypatch, components):
    monkeypatch.setenv("WEB_ADMIN_USER", "admin")
    monkeypatch.setenv("WEB_ADMIN_PASSWORD", PASSWORD)
    SESSIONS.settings = WebSettings()
    SESSIONS._sessions.clear()

    async with app.router.lifespan_context(app):
        app.state.store = InMemoryWebStore()
        app.state.backend = StubPipeline()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://console") as c:
            yield c


async def login(client: httpx.AsyncClient) -> None:
    await client.post("/api/login", json={"username": "admin", "password": PASSWORD})


# --- auth gating -----------------------------------------------------------


async def test_inspection_endpoints_reject_anonymous_callers(client):
    for path in ("/trajectories", "/trajectories/run-pii", "/audit-log"):
        assert (await client.get(path)).status_code == 401, path


async def test_logout_closes_access_to_trajectories(client, components):
    await login(client)
    await components.recorder.record(REPORT_WITH_PII)
    assert (await client.get("/trajectories/run-pii")).status_code == 200

    await client.post("/api/logout")

    assert (await client.get("/trajectories/run-pii")).status_code == 401


# --- trajectories ----------------------------------------------------------


async def test_get_trajectory_returns_the_full_scrubbed_record(client, components):
    """Acceptance: GET /trajectories/{run_id} returns correct, auth-gated data."""
    await login(client)
    await components.recorder.record(REPORT_WITH_PII)

    resp = await client.get("/trajectories/run-pii")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run-pii"
    assert body["scrubbed"] is True
    assert body["steps"], "steps must be returned, not just the header"
    assert EMAIL not in resp.text, "the endpoint must never serve raw PII"
    assert body["reward"]["resolution"] == "completed"


async def test_unknown_run_is_a_404(client):
    await login(client)
    assert (await client.get("/trajectories/nope")).status_code == 404


async def test_trajectory_listing_summarises_runs(client, components):
    await login(client)
    await components.recorder.record(REPORT_WITH_PII)

    rows = (await client.get("/trajectories")).json()["trajectories"]

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-pii"
    assert rows[0]["resolution"] == "completed"
    assert rows[0]["step_count"] > 0


async def test_export_endpoint_returns_the_training_shape(client, components):
    await login(client)
    await components.recorder.record(REPORT_WITH_PII)

    body = (await client.get("/trajectories/run-pii/export")).json()

    assert body["schema_version"] == "1.0"
    assert body["pii_scrubbed"] is True
    assert body["turns"] and body["reward"] >= 0.0
    assert EMAIL not in str(body)


async def test_feedback_endpoint_records_an_explicit_signal(client, components):
    await login(client)
    await components.recorder.record(REPORT_WITH_PII)

    resp = await client.post(
        "/trajectories/run-pii/feedback",
        json={"feedback": "negative", "note": f"wrong account for {EMAIL}"},
    )

    assert resp.status_code == 200
    reward = resp.json()["reward"]
    assert reward["user_feedback"] == "negative"
    assert EMAIL not in resp.text
    assert reward["feedback_at"]


async def test_feedback_rejects_an_unknown_label(client, components):
    await login(client)
    await components.recorder.record(REPORT_WITH_PII)

    resp = await client.post("/trajectories/run-pii/feedback", json={"feedback": "wonderful"})

    assert resp.status_code == 422


# --- audit log -------------------------------------------------------------


async def test_audit_log_returns_entries_with_diffs(client, components):
    """Acceptance: GET /audit-log returns correct, auth-gated data."""
    await login(client)
    await components.audit.record(
        actor="admin",
        action="agent_config.update",
        entity_type="agent_config",
        entity_id="cfg-1",
        before={"name": "Ops"},
        after={"name": "Ops v2"},
    )

    entries = (await client.get("/audit-log")).json()["entries"]

    assert len(entries) == 1
    assert entries[0]["actor"] == "admin"
    assert entries[0]["action"] == "agent_config.update"
    assert entries[0]["changes"][0]["path"] == "name"
    assert entries[0]["changes"][0]["after"] == "Ops v2"


async def test_audit_log_filters_narrow_the_result(client, components):
    await login(client)
    await components.audit.record(
        actor="alice", action="action.register", entity_type="action", entity_id="a"
    )
    await components.audit.record(
        actor="bob", action="agent_config.update", entity_type="agent_config", entity_id="b"
    )

    all_entries = (await client.get("/audit-log")).json()["entries"]
    filtered = (await client.get("/audit-log?entity_type=action")).json()["entries"]

    assert len(all_entries) == 2
    assert len(filtered) == 1 and filtered[0]["actor"] == "alice"


async def test_audit_log_limit_is_bounded(client):
    await login(client)
    assert (await client.get("/audit-log?limit=0")).status_code == 422
    assert (await client.get("/audit-log?limit=9999")).status_code == 422
