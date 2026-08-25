"""Append-only filesystem journal: the OS's persistent memory of runs.

No external database -- one JSONL file per record type under
``<root>/data/journal/``. Lines are only ever appended; history stays readable
by ``jq`` or plain ``Get-Content`` and schema evolution follows the
add-fields-don't-rename rule. Journal failures degrade to log warnings: the
pipeline must never refuse to run because its diary is full.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.observability import get_logger

logger = get_logger("agentforge.os.journal")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Journal:
    def __init__(self, root: Path) -> None:
        self.dir = Path(root) / "data" / "journal"
        self.runs_path = self.dir / "runs.jsonl"
        self.decisions_path = self.dir / "decisions.jsonl"

    def _append(self, path: Path, record: dict) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            return True
        except OSError as exc:
            logger.warning("journal append failed: %s", exc)
            return False

    def log_run(self, report: dict) -> bool:
        """One line per pipeline run: status, stages, latency, errors."""
        return self._append(
            self.runs_path,
            {
                "at": _now(),
                "run_id": report.get("run_id"),
                "session_id": report.get("session_id"),
                "topic": report.get("topic"),
                "status": report.get("status"),
                "cached": report.get("cached", False),
                "stages": [
                    s
                    for s in ("analysis", "develop", "test", "deploy")
                    if report.get(s)
                ],
                "total_latency_ms": report.get("total_latency_ms"),
                "errors": report.get("errors") or [],
            },
        )

    def log_decision(self, kind: str, payload: dict) -> bool:
        """Decision records: routing choices, go/no-go calls, escalations."""
        return self._append(
            self.decisions_path,
            {"at": _now(), "kind": kind, **payload},
        )
