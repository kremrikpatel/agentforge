"""The alert endpoints, mounted into the Phase 4 console and gated by its auth.

Drives the real router over ASGI with in-memory backends, so nothing here
needs Redis or Postgres.
"""

from __future__ import annotations

import httpx
import pytest

from monitoring.api import Components, set_components
from monitoring.config import get_monitoring_settings
from monitoring.dedup import InMemoryAlertState
from monitoring.schemas import Alert, Severity
from monitoring.store import InMemoryAlertStore
from tests.test_web_api import PASSWORD, StubPipeline
from web.auth import SESSIONS
from web.config import WebSettings
from web.main import app
from web.store import InMemoryWebStore


@pytest.fixture
def components():
    bundle = Components(
        state=InMemoryAlertState(dedup_window_s=900.0),
        store=InMemoryAlertStore(),
        settings=get_monitoring_settings(),
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
    resp = await client.post("/api/login", json={"username": "admin", "password": PASSWORD})
    assert resp.status_code == 200


async def seed(bundle: Components, fingerprint: str = "escalation_rate") -> None:
    await bundle.store.record(
        Alert(
            rule="escalation_rate",
            severity=Severity.WARNING,
            summary="Escalation rate 80% exceeds 25%",
            fingerprint=fingerprint,
        ),
        ["slack"],
    )


async def test_alerts_require_authentication(client, components):
    """The endpoints inherit the console's session auth, not a second model."""
    await seed(components)

    resp = await client.get("/api/alerts")

    assert resp.status_code in (401, 403)


async def test_list_alerts_returns_history_with_suppression_state(client, components):
    await login(client)
    await seed(components)

    body = (await client.get("/api/alerts")).json()

    assert len(body["alerts"]) == 1
    row = body["alerts"][0]
    assert row["rule"] == "escalation_rate"
    assert row["channels"] == ["slack"]
    assert row["suppression"] == {"silenced": False, "acknowledged": False}


async def test_silence_endpoint_mutes_the_fingerprint(client, components):
    await login(client)
    await seed(components)

    resp = await client.post(
        "/api/alerts/escalation_rate/silence", json={"duration_s": 600}
    )

    assert resp.status_code == 200
    assert resp.json()["suppression"]["silenced"] is True
    assert await components.state.should_notify("escalation_rate") is False


async def test_acknowledge_endpoint_mutes_until_cleared(client, components):
    await login(client)
    await seed(components)

    resp = await client.post("/api/alerts/escalation_rate/acknowledge")

    assert resp.status_code == 200
    assert resp.json()["suppression"]["acknowledged"] is True
    assert await components.state.should_notify("escalation_rate") is False


async def test_clearing_suppression_lets_the_alert_fire_again(client, components):
    await login(client)
    await seed(components)
    await client.post("/api/alerts/escalation_rate/acknowledge")

    resp = await client.request("DELETE", "/api/alerts/escalation_rate/suppression")

    assert resp.status_code == 200
    assert resp.json()["suppression"] == {"silenced": False, "acknowledged": False}
    assert await components.state.should_notify("escalation_rate") is True


async def test_a_per_action_fingerprint_survives_the_url_round_trip(client, components):
    """circuit_breaker_trips fingerprints carry a colon; silencing one action
    must not silence the whole rule."""
    await login(client)
    await seed(components, fingerprint="circuit_breaker_trips:create_ticket")

    resp = await client.post(
        "/api/alerts/circuit_breaker_trips%3Acreate_ticket/acknowledge"
    )

    assert resp.status_code == 200
    assert resp.json()["fingerprint"] == "circuit_breaker_trips:create_ticket"
    assert await components.state.should_notify("circuit_breaker_trips:create_ticket") is False
    assert await components.state.should_notify("circuit_breaker_trips:other") is True


async def test_silence_duration_is_bounded(client, components):
    """An unbounded silence is an acknowledgement, and that has its own route."""
    await login(client)
    await seed(components)

    resp = await client.post(
        "/api/alerts/escalation_rate/silence", json={"duration_s": 999_999_999}
    )

    assert resp.status_code == 422
