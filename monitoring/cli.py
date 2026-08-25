"""Operable entrypoint for the trajectory-window rules.

There is no scheduler in app/ to attach `evaluate_trajectories` to, and
wiring one into the Helm chart (a CronJob, like the existing red-team one) is
a Helm change -- out of scope for this phase unless asked for. Run this on a
cron until that lands:

    python -m monitoring.cli sweep [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from app.observability import configure_logging
from instrumentation.store import build_store
from monitoring.engine import MonitoringEngine


async def _load_trajectories(limit: int):
    store = await build_store()
    rows = await store.list_trajectories(limit=limit)
    trajectories = [await store.get_trajectory(row["run_id"]) for row in rows]
    return [t for t in trajectories if t is not None]


async def _sweep(limit: int) -> int:
    trajectories = await _load_trajectories(limit)
    engine = await MonitoringEngine.build()
    alerts = await engine.evaluate_trajectories(trajectories)
    for alert in alerts:
        print(f"[{alert.severity.value}] {alert.rule}: {alert.summary}")
    print(f"{len(trajectories)} trajectories evaluated, {len(alerts)} alert(s) fired.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="monitoring.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="Evaluate the trajectory-window rules once.")
    sweep.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)

    configure_logging(get_settings().log_level)
    if args.command == "sweep":
        return asyncio.run(_sweep(args.limit))
    return 1


if __name__ == "__main__":
    sys.exit(main())
