"""Read-only red-team dashboard.

FastAPI + a static page rather than Streamlit, for three reasons: Phase 1
already serves a static page from FastAPI so this reuses an established pattern;
it adds no dependency (Streamlit would be a second web framework and a second
server model); and the page is read-only aggregates plus transcripts, which
needs nothing Streamlit provides.

Served on its own port, separate from the Phase 1 API, because attack
transcripts are sensitive and should not ride on the public app.

    python -m redteam.dashboard
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.config import get_settings
from app.observability import configure_logging
from redteam.config import get_redteam_settings
from redteam.store import build_store, rates_from_rows

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="AgentForge Red Team",
    version="0.1.0",
    description="Attack results, block rates, and failing transcripts.",
)


async def _store():
    return await build_store(get_settings(), get_redteam_settings().persist)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "dashboard.html")


@app.get("/api/runs")
async def runs(limit: int = 20) -> dict:
    store = await _store()
    return {"runs": await store.recent_runs(limit)}


@app.get("/api/runs/{run_id}")
async def run_detail(run_id: str) -> dict:
    store = await _store()
    rows = await store.recent_runs(50)
    current = next((r for r in rows if r["run_id"] == run_id), None)
    if current is None:
        raise HTTPException(status_code=404, detail="no such run")

    attempts = await store.attempts_for(run_id)
    # The run immediately before this one, against the same target.
    order = [r for r in rows if r["target"] == current["target"]]
    idx = next((i for i, r in enumerate(order) if r["run_id"] == run_id), 0)
    previous = order[idx + 1] if idx + 1 < len(order) else None

    trend = []
    if previous is not None:
        prev_rates = rates_from_rows(previous["categories"])
        for row in current["categories"]:
            before = prev_rates.get(row["category"])
            trend.append(
                {
                    "category": row["category"],
                    "current": row["block_rate"],
                    "previous": before,
                    "delta": None if before is None else round(row["block_rate"] - before, 4),
                }
            )

    return {
        "run": current,
        "previous_run_id": previous["run_id"] if previous else None,
        "trend": trend,
        "attempts": attempts,
        "failures": [a for a in attempts if a.get("outcome") == "leaked"],
    }


def main() -> None:
    import uvicorn

    settings = get_redteam_settings()
    configure_logging(get_settings().log_level)
    uvicorn.run(
        app,
        host=settings.dashboard_host,  # loopback by default, not 0.0.0.0
        port=settings.dashboard_port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
