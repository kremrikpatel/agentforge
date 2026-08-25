"""The OS layer: frontmatter parsing, registry loading, kernel routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.contracts import STAGE_ORDER, Stage
from agentos.loader import AgentOS, parse_frontmatter
from agentos.specs import KernelSpec, RoutingRule

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_OS = REPO_ROOT / "agentos"


# --- frontmatter parser ------------------------------------------------------


def test_parses_toml_frontmatter_from_body():
    raw, body = parse_frontmatter('+++\nname = "Ana"\nstage = "analysis"\n+++\n\nCharter text.')
    assert raw == {"name": "Ana", "stage": "analysis"}
    assert body == "Charter text."


def test_missing_fences_returns_none_and_whole_text():
    raw, body = parse_frontmatter("no fences here")
    assert raw is None
    assert body == "no fences here"


def test_invalid_toml_degrades_to_no_frontmatter():
    raw, body = parse_frontmatter("+++\nnot toml at all [[[\n+++\nbody")
    assert raw is None


# --- shipped files load ------------------------------------------------------


@pytest.fixture
def os_registry() -> AgentOS:
    return AgentOS(SHIPPED_OS)


def test_all_four_personas_load_from_shipped_files(os_registry: AgentOS):
    assert set(os_registry.personas) == set(STAGE_ORDER)
    ana = os_registry.personas[Stage.ANALYSIS]
    assert ana.name == "Ana"
    assert "You are Ana" in ana.charter


def test_kernel_routing_and_charter_load(os_registry: AgentOS):
    assert os_registry.kernel.name == "AgentForge OS"
    assert "four-agent delivery team" in os_registry.kernel.team_charter
    assert len(os_registry.kernel.routing) >= 1


def test_commands_load_with_stage_subsets(os_registry: AgentOS):
    cmd = os_registry.command("analysis-only")
    assert cmd is not None
    assert cmd.ordered_stages() == [Stage.ANALYSIS]
    assert os_registry.command("through-test").ordered_stages() == [
        Stage.ANALYSIS,
        Stage.DEVELOP,
        Stage.TEST,
    ]


def test_system_prompt_composes_charter_then_persona(os_registry: AgentOS):
    prompt = os_registry.system_prompt(Stage.ANALYSIS)
    assert prompt.startswith("You are one member of a four-agent delivery team")
    assert "You are Ana, the Analyst." in prompt


def test_persona_model_policy_overrides_defaults(os_registry: AgentOS):
    tess = os_registry.personas[Stage.TEST]
    assert tess.model.temperature == 0.1


def test_summary_shape_for_api(os_registry: AgentOS):
    summary = os_registry.summary()
    ids = {a["id"] for a in summary["agents"]}
    assert ids == {"ana-analyst", "dev-architect", "tess-qa", "dep-release"}
    assert any(c["id"] == "full-delivery" for c in summary["commands"])


# --- fallbacks ----------------------------------------------------------------


def test_missing_directory_falls_back_to_builtin_prompts(tmp_path: Path):
    os_reg = AgentOS(tmp_path / "does-not-exist")
    from agents.nodes import SYSTEM_PROMPTS

    for stage in STAGE_ORDER:
        assert os_reg.system_prompt(stage) == SYSTEM_PROMPTS[stage]
        assert os_reg.policy(stage) is None


def test_broken_persona_file_is_skipped_not_fatal(tmp_path: Path):
    personas = tmp_path / "personas"
    personas.mkdir()
    (personas / "ana-analyst.md").write_text("no frontmatter at all", encoding="utf-8")
    (personas / "dev-architect.md").write_text(
        '+++\nname = "Dev"\nrole = "Solution Architect"\nstage = "develop"\n+++\n\nDev body.',
        encoding="utf-8",
    )
    os_reg = AgentOS(tmp_path)
    assert set(os_reg.personas) == {Stage.DEVELOP}
    # Ana falls through to the built-in prompt rather than vanishing.
    assert "You are Ana" in os_reg.system_prompt(Stage.ANALYSIS)


def test_broken_kernel_file_keeps_defaults(tmp_path: Path):
    (tmp_path / "kernel.md").write_text("+++\nbroken [[[\n+++\nignored", encoding="utf-8")
    os_reg = AgentOS(tmp_path)
    assert "four-agent delivery team" in os_reg.kernel.team_charter
    assert os_reg.kernel.routing == []


# --- routing ------------------------------------------------------------------


def test_first_matching_rule_wins():
    kernel = KernelSpec(
        routing=[
            RoutingRule(intent="quick", match=["quick analysis"], command="analysis-only"),
            RoutingRule(intent="analyze", match=["analyze"], stage=Stage.ANALYSIS),
        ]
    )
    rule, rationale = kernel.route("please do a quick analysis of this")
    assert rule is not None and rule.intent == "quick"


def test_route_to_command_resolves_stages(os_registry: AgentOS):
    decision = os_registry.route("give me a quick analysis of caching")
    assert decision.target_kind == "command"
    assert decision.command == "analysis-only"
    assert decision.stages == [Stage.ANALYSIS]


def test_route_to_single_stage(os_registry: AgentOS):
    decision = os_registry.route("just analyze the requirements")
    assert decision.target_kind == "pipeline"
    assert decision.stages == [Stage.ANALYSIS]


def test_unmatched_task_falls_back_to_full_pipeline(os_registry: AgentOS):
    decision = os_registry.route("design a rate limiter for a public REST API")
    assert decision.target_kind == "fallback"
    assert decision.stages == list(STAGE_ORDER)


def test_rule_without_target_is_dropped(tmp_path: Path):
    (tmp_path / "kernel.md").write_text(
        '+++\nname = "K"\n[[routing]]\nintent = "empty"\nmatch = ["x"]\n+++',
        encoding="utf-8",
    )
    os_reg = AgentOS(tmp_path)
    assert os_reg.kernel.routing == []


# --- reload -------------------------------------------------------------------


def test_reload_picks_up_edits(tmp_path: Path):
    personas = tmp_path / "personas"
    personas.mkdir()
    path = personas / "tess-qa.md"
    path.write_text(
        '+++\nname = "Tess"\nrole = "QA"\nstage = "test"\nversion = 1'
        "\n[model]\ntemperature = 0.1\n+++\n\nv1 body.",
        encoding="utf-8",
    )
    os_reg = AgentOS(tmp_path)
    assert os_reg.personas[Stage.TEST].model.temperature == 0.1

    path.write_text(
        '+++\nname = "Tess"\nrole = "QA"\nstage = "test"\nversion = 2'
        "\n[model]\ntemperature = 0.7\n+++\n\nv2 body.",
        encoding="utf-8",
    )
    os_reg.reload()
    assert os_reg.personas[Stage.TEST].model.temperature == 0.7
