"""LangGraph wiring: state shape, routing, and report assembly.

Sequential by necessity -- each stage's input is the previous stage's contract,
so there is nothing to fan out. Coordination happens through the shared
blackboard every node reads and appends to.

After each node the router asks one question: did anything halt the run? A
guardrail block, an exhausted provider chain, an unparseable contract, or an
agent raising a blocking open question all short-circuit to END, so a broken
handoff never reaches the next agent.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from agents.contracts import STAGE_ORDER, PipelineReport, Stage
from agents.nodes import AgentDeps, make_node, next_stage


def _merge_stages(left: dict, right: dict) -> dict:
    return {**left, **right}


class PipelineState(TypedDict, total=False):
    run_id: str
    session_id: str
    topic: str
    # Reducers: without these, a node returning {"traces": [x]} would overwrite
    # rather than append, and the audit trail would only hold the last node.
    stages: Annotated[dict, _merge_stages]
    blackboard: Annotated[list, operator.add]
    traces: Annotated[list, operator.add]
    guardrails: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    halted: bool


def initial_state(run_id: str, session_id: str, topic: str) -> PipelineState:
    return {
        "run_id": run_id,
        "session_id": session_id,
        "topic": topic,
        "stages": {},
        "blackboard": [],
        "traces": [],
        "guardrails": [],
        "errors": [],
        "halted": False,
    }


def route_after(stage: Stage):
    """Continue to the next agent, unless the run has halted."""
    nxt = next_stage(stage)

    def _route(state: PipelineState) -> str:
        if state.get("halted"):
            return END
        return nxt.value if nxt else END

    _route.__name__ = f"route_after_{stage.value}"
    return _route


def build_graph(deps: AgentDeps):
    graph = StateGraph(PipelineState)

    for stage in STAGE_ORDER:
        graph.add_node(stage.value, make_node(stage, deps))

    graph.set_entry_point(STAGE_ORDER[0].value)
    for stage in STAGE_ORDER:
        nxt = next_stage(stage)
        targets = {END: END} if nxt is None else {nxt.value: nxt.value, END: END}
        graph.add_conditional_edges(stage.value, route_after(stage), targets)

    return graph.compile()


def to_report(state: PipelineState) -> PipelineReport:
    stages = state.get("stages", {})
    errors = state.get("errors", [])
    completed = all(s.value in stages for s in STAGE_ORDER)

    if completed and not errors:
        status = "completed"
    elif stages:
        status = "halted"
    else:
        status = "failed"

    return PipelineReport(
        run_id=state["run_id"],
        session_id=state["session_id"],
        topic=state["topic"],
        status=status,  # type: ignore[arg-type]
        analysis=stages.get(Stage.ANALYSIS.value),
        develop=stages.get(Stage.DEVELOP.value),
        test=stages.get(Stage.TEST.value),
        deploy=stages.get(Stage.DEPLOY.value),
        team_messages=state.get("blackboard", []),
        traces=state.get("traces", []),
        guardrail_reports=state.get("guardrails", []),
        errors=errors,
        total_latency_ms=round(
            sum(t.get("latency_ms", 0.0) for t in state.get("traces", [])), 2
        ),
    )
