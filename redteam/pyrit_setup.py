"""Initialise PyRIT's central memory.

PyRIT refuses to construct a target until CentralMemory is bound, and
PromptTarget.__init__ is synchronous, so this has to be callable from sync code.

The binding is an ephemeral in-memory SQLite. Deliberate: attack transcripts are
sensitive, and PyRIT's default would drop them in a file beside the repo. Our
Postgres store is the system of record; PyRIT's memory is scratch space that
disappears with the process.
"""

from __future__ import annotations

from app.observability import get_logger, log_event

logger = get_logger("agentforge.redteam.pyrit")

_ready = False


def ensure_pyrit_memory() -> None:
    """Idempotent, synchronous, safe to call from anywhere."""
    global _ready
    if _ready:
        return

    from pyrit.memory import CentralMemory, SQLiteMemory

    try:
        CentralMemory.get_memory_instance()
    except ValueError:
        CentralMemory.set_memory_instance(SQLiteMemory(db_path=":memory:", silent=True))
        log_event(logger, "redteam.pyrit_memory", backend="sqlite::memory:")
    _ready = True
