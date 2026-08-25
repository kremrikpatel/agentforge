"""The OS in action: persona-driven prompts, stage-subset runs, the journal."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agents.contracts import STAGE_ORDER, Stage
from agents.graph import build_graph, initial_state, to_report
from agentos.journal import Journal
from agentos.loader import AgentOS
from gateway.providers import StubProvider
from tests.conftest import contract_json
from tests.test_graph import TOPIC, make_deps

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_OS = REPO_ROOT / "agentos"


def make_os_deps(settings, providers=None):
    deps = make_deps(settings, providers)
    return dataclasses.replace(deps, personas=AgentOS(SHIPPED_OS))


async def run(deps, stages=STAGE_ORDER, topic=TOPIC):
    graph = build_graph(deps, stages=tuple(stages))
    return await graph.ainvoke(initial_state("run-os", "sess-os", topic))


# --- persona wiring -----------------------------------------------------------


async def test_nodes_use_persona_charter_from_files(settings):
    seen_systems: list[str] = []

    class RecordingStub(StubProvider):
        async def complete(self, req, client) -> str:
            seen_systems.append(req.system)
            return await super().complete(req, client)

    cfg = dataclasses.replace(settings, allow_stub_provider=True)
    await run(make_os_deps(settings, [RecordingStub(cfg)]))

    assert len(seen_systems) == 4
    # Charter preamble first, then each persona's own identity text.
    for system, marker in zip(seen_systems, ("Ana", "Dev", "Tess", "Dep")):
        assert system.startswith("You are one member of a four-agent delivery team")
        assert f"You are {marker}" in system


# --- partial runs ---------------------------------------------------------------


async def test_analysis_only_run_completes_with_one_contract(settings):
    state = await run(make_os_deps(settings), stages=[Stage.ANALYSIS])

    assert set(state["stages"]) == {"analysis"}
    report = to_report(state, expected=(Stage.ANALYSIS,))
    assert report.status == "completed"
    assert not report.errors


async def test_non_contiguous_subset_skips_disabled_stages(settings):
    state = await run(make_os_deps(settings), stages=[Stage.ANALYSIS, Stage.TEST])

    assert set(state["stages"]) == {"analysis", "test"}
    report = to_report(state, expected=(Stage.ANALYSIS, Stage.TEST))
    assert report.status == "completed"


def test_to_report_judges_completion_against_the_expected_set():
    analysis = json.loads(contract_json("analysis"))
    state = {
        "run_id": "r",
        "session_id": "s",
        "topic": "t",
        "stages": {"analysis": analysis},
        "traces": [],
        "errors": [],
    }
    # Intentionally partial: completed for what was asked, halted for the default.
    assert to_report(state, expected=(Stage.ANALYSIS,)).status == "completed"
    assert to_report(state).status == "halted"


# --- journal ---------------------------------------------------------------------


def test_journal_appends_valid_jsonl(tmp_path: Path):
    journal = Journal(tmp_path)
    record = {
        "run_id": "r1",
        "session_id": "s1",
        "topic": "topic",
        "status": "completed",
        "cached": False,
        "analysis": {"stage": "analysis"},
        "total_latency_ms": 12.5,
        "errors": [],
    }
    assert journal.log_run(record)
    assert journal.log_decision("qa-fail-escalation", {"run_id": "r1"})
    assert journal.log_run(dict(record, run_id="r2"))

    lines = (tmp_path / "data" / "journal" / "runs.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["r1", "r2"]

    decisions = (tmp_path / "data" / "journal" / "decisions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert json.loads(decisions[0])["kind"] == "qa-fail-escalation"


def test_journal_failure_degrades_to_false(tmp_path: Path):
    journal = Journal(tmp_path)
    # A directory where the log file belongs makes every append an OSError.
    (tmp_path / "data" / "journal" / "runs.jsonl").mkdir(parents=True)
    assert journal.log_run({"run_id": "r", "status": "completed"}) is False
