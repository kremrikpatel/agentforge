"""Inspection endpoints for trajectories and the audit log.

Mounted into the Phase 4 console app and gated by its `require_admin`, so there
is one session model rather than a second one invented here.

Components are lazily built and replaceable, which is what lets tests drive the
router against an in-memory store without a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.observability import get_logger
from instrumentation.audit import AuditLog
from instrumentation.recorder import TrajectoryRecorder
from instrumentation.schemas import Trajectory, TrainingExample, to_training_example
from instrumentation.store import InstrumentationStore, build_store
from instrumentation.versioning import VersionStore
from web.auth import Session, require_admin

logger = get_logger("agentforge.instrumentation.api")

router = APIRouter(tags=["instrumentation"])


@dataclass
class Components:
    store: InstrumentationStore
    recorder: TrajectoryRecorder
    audit: AuditLog
    versions: VersionStore


_components: Components | None = None


async def get_components() -> Components:
    global _components
    if _components is None:
        store = await build_store()
        _components = Components(
            store=store,
            recorder=TrajectoryRecorder(store),
            audit=AuditLog(store),
            versions=VersionStore(store),
        )
    return _components


def set_components(components: Components | None) -> None:
    """Inject (or reset) the component set. Used by tests and by app startup."""
    global _components
    _components = components


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback: Literal["positive", "negative", "neutral"]
    note: str = Field(default="", max_length=4000)


@router.get("/trajectories")
async def list_trajectories(
    limit: int = Query(50, ge=1, le=200), _: Session = Depends(require_admin)
) -> dict[str, Any]:
    components = await get_components()
    return {"trajectories": await components.store.list_trajectories(limit)}


@router.get("/trajectories/{run_id}")
async def get_trajectory(run_id: str, _: Session = Depends(require_admin)) -> Trajectory:
    components = await get_components()
    trajectory = await components.store.get_trajectory(run_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail="no trajectory for that run")
    return trajectory


@router.get("/trajectories/{run_id}/export")
async def export_trajectory(
    run_id: str, _: Session = Depends(require_admin)
) -> TrainingExample:
    """The training-export projection of a stored trajectory."""
    components = await get_components()
    trajectory = await components.store.get_trajectory(run_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail="no trajectory for that run")
    try:
        return to_training_example(trajectory)
    except ValueError as exc:
        # Only reachable if something wrote around the store's scrubbing.
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/trajectories/{run_id}/feedback")
async def record_feedback(
    run_id: str, payload: FeedbackRequest, _: Session = Depends(require_admin)
) -> Trajectory:
    components = await get_components()
    updated = await components.recorder.record_feedback(
        run_id, payload.feedback, payload.note
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="no trajectory for that run")
    return updated


@router.get("/audit-log")
async def audit_log(
    limit: int = Query(100, ge=1, le=500),
    entity_type: str = "",
    entity_id: str = "",
    actor: str = "",
    action: str = "",
    _: Session = Depends(require_admin),
) -> dict[str, Any]:
    components = await get_components()
    entries = await components.audit.list(
        limit, entity_type=entity_type, entity_id=entity_id, actor=actor, action=action
    )
    return {"entries": entries}
