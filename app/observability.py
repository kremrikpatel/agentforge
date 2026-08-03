"""Structured JSON logging + an in-process event bus the coordination UI subscribes to.

Every agent node emits a NodeTrace (reasoning summary, tool calls, latency,
outcome). Phase 2 ships these to LangSmith/Arize; today they land in stdout as
JSON and in the run's report, which is the shape a trainer needs later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Iterator

_SECRET_HINTS = ("api_key", "apikey", "token", "secret", "password", "authorization")


def _scrub(value: Any) -> Any:
    """Never let a credential reach a log line."""
    if isinstance(value, dict):
        return {
            k: ("***" if any(h in k.lower() for h in _SECRET_HINTS) else _scrub(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload.update(_scrub(extra))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, /, **context: Any) -> None:
    logger.info(msg, extra={"context": context})


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """`with timed() as t: ...` -> t["ms"] holds elapsed wall time."""
    slot: dict[str, float] = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield slot
    finally:
        slot["ms"] = round((time.perf_counter() - start) * 1000, 2)


class EventBus:
    """Fan-out of run events to SSE subscribers.

    ponytail: in-process only, so it breaks across replicas. Swap the deque for
    Redis pub/sub when the API runs more than one pod.
    """

    def __init__(self, history: int = 200) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict]]] = defaultdict(list)
        self._history: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=history))

    def publish(self, run_id: str, event: dict) -> None:
        self._history[run_id].append(event)
        for queue in list(self._subscribers[run_id]):
            queue.put_nowait(event)

    def replay(self, run_id: str) -> list[dict]:
        return list(self._history[run_id])

    def subscribe(self, run_id: str) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        self._subscribers[run_id].append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[dict]) -> None:
        if queue in self._subscribers[run_id]:
            self._subscribers[run_id].remove(queue)


EVENT_BUS = EventBus()
