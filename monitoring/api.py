"""Alert endpoints for the admin console.

Mounted into the Phase 4 console app and gated by its `require_admin`, the
same way `instrumentation/api.py` is -- one session model, not a second one
invented here.

Components are lazily built and injectable, which is what lets tests drive
the router against in-memory backends with no Redis and no Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from app.observability import get_logger
from monitoring.config import MonitoringSettings, get_monitoring_settings
from monitoring.dedup import AlertState, build_state
from monitoring.store import AlertStore, build_alert_store
from web.auth import Session, require_admin

logger = get_logger("agentforge.monitoring.api")

router = APIRouter(tags=["monitoring"])


@dataclass
class Components:
    state: AlertState
    store: AlertStore
    settings: MonitoringSettings


_components: Components | None = None


async def get_components() -> Components:
    global _components
    if _components is None:
        settings = get_monitoring_settings()
        _components = Components(
            state=await build_state(settings.dedup_window_s, settings.state_backend),
            store=await build_alert_store(),
            settings=settings,
        )
    return _components


def set_components(components: Components | None) -> None:
    """Inject (or reset) the component set. Used by tests and by app startup."""
    global _components
    _components = components


class SilenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Bounded: an unbounded silence is an acknowledgement, and that has its
    # own endpoint. 7 days is the ceiling so a muted alert cannot be forgotten
    # forever by accident.
    duration_s: float | None = Field(default=None, gt=0, le=604_800)


@router.get("/api/alerts")
async def list_alerts(
    limit: int = Query(50, ge=1, le=200), _: Session = Depends(require_admin)
) -> dict[str, Any]:
    """Recent fired alerts, each overlaid with its current suppression state.

    The history comes from Postgres and the flags from Redis, so a row shows
    both what fired and whether it is currently muted.
    """
    components = await get_components()
    rows = await components.store.recent(limit)
    for row in rows:
        row["suppression"] = await components.state.status(row["fingerprint"])
    return {"alerts": rows, "silence_default_s": components.settings.silence_default_s}


@router.post("/api/alerts/{fingerprint}/silence")
async def silence_alert(
    fingerprint: str,
    payload: SilenceRequest | None = None,
    _: Session = Depends(require_admin),
) -> dict[str, Any]:
    """Mute a fingerprint for a bounded window."""
    components = await get_components()
    duration = (
        payload.duration_s
        if payload and payload.duration_s
        else components.settings.silence_default_s
    )
    await components.state.silence(fingerprint, duration)
    return {
        "fingerprint": fingerprint,
        "duration_s": duration,
        "suppression": await components.state.status(fingerprint),
    }


@router.post("/api/alerts/{fingerprint}/acknowledge")
async def acknowledge_alert(
    fingerprint: str, _: Session = Depends(require_admin)
) -> dict[str, Any]:
    """Mute a fingerprint until it is explicitly cleared."""
    components = await get_components()
    await components.state.acknowledge(fingerprint)
    return {
        "fingerprint": fingerprint,
        "suppression": await components.state.status(fingerprint),
    }


@router.delete("/api/alerts/{fingerprint}/suppression")
async def clear_suppression(
    fingerprint: str, _: Session = Depends(require_admin)
) -> dict[str, Any]:
    """Drop silence and acknowledgement so the alert can fire again."""
    components = await get_components()
    await components.state.clear(fingerprint)
    return {
        "fingerprint": fingerprint,
        "suppression": await components.state.status(fingerprint),
    }
