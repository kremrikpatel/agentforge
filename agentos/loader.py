"""Load and serve the OS layer: kernel, personas, commands.

The directory layout is the database::

    agentos/
    +-- kernel.md          TOML frontmatter (routing table) + identity body
    +-- personas/*.md      one specialist agent per file
    +-- commands/*.md      named stage-subset workflows
    +-- data/journal/      append-only run log (written by journal.Journal)

Files are read fresh on :meth:`AgentOS.reload`; the pipeline holds only this
registry object, so personas change without code changes or restarts. When the
directory is missing or empty the registry falls back to the built-in prompts
in ``agents.nodes`` -- the shipped persona files mirror those exactly.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import ValidationError

from agents.contracts import Stage, STAGE_ORDER
from agentos.specs import CommandSpec, KernelSpec, ModelPolicy, PersonaSpec, RouteDecision
from app.observability import get_logger

logger = get_logger("agentforge.os.loader")

PERSONA_STAGES: dict[Stage, str] = {
    Stage.ANALYSIS: "ana-analyst",
    Stage.DEVELOP: "dev-architect",
    Stage.TEST: "tess-qa",
    Stage.DEPLOY: "dep-release",
}


def parse_frontmatter(text: str) -> tuple[dict | None, str]:
    """Split ``+++``-fenced TOML frontmatter from the markdown body."""
    if not text.startswith("+++"):
        return None, text.strip()
    try:
        end = text.index("\n+++")
    except ValueError:
        return None, text.strip()
    head = text[3:end]
    body = text[end + 4 :]
    try:
        return tomllib.loads(head), body.strip()
    except tomllib.TOMLDecodeError:
        return None, body.strip()


def _load_persona(path: Path) -> PersonaSpec:
    """PersonaSpec from one markdown file; grouped tables flatten to fields."""
    raw, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"{path.name}: missing or invalid TOML frontmatter")
    nested = {
        "tools": (raw.pop("tools", None) or {}).get("allowed"),
        "memory_scope": (raw.pop("memory", None) or {}).get("scope"),
        "triggers": (raw.pop("routing", None) or {}).get("triggers"),
    }
    raw.update({k: v for k, v in nested.items() if v is not None})
    raw.setdefault("id", path.stem)
    return PersonaSpec.model_validate({**raw, "charter": body})


def _load_command(path: Path) -> CommandSpec:
    raw, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"{path.name}: missing or invalid TOML frontmatter")
    raw.setdefault("id", path.stem)
    return CommandSpec.model_validate(raw)


def _try_load(loader, path: Path):
    """Load one spec file; a broken file is skipped rather than fatal."""
    try:
        return loader(path)
    except (ValueError, ValidationError) as exc:
        logger.warning("skipping invalid OS file %s: %s", path.name, exc)
        return None


class AgentOS:
    """Registry of personas + kernel routing + commands. One per process."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else None
        self.kernel = KernelSpec(team_charter=_default_team_charter())
        self.personas: dict[Stage, PersonaSpec] = {}
        self.commands: dict[str, CommandSpec] = {}
        self.sources: list[str] = []
        self.reload()

    # --- loading ---------------------------------------------------------

    def reload(self) -> None:
        """Re-read every file under root. Missing dir keeps built-in defaults."""
        self.kernel = KernelSpec(team_charter=_default_team_charter())
        self.personas.clear()
        self.commands.clear()
        self.sources = []
        if self.root is None or not self.root.is_dir():
            return
        kernel_path = self.root / "kernel.md"
        if kernel_path.exists():
            self._merge_kernel(kernel_path)
            self.sources.append(str(kernel_path))
        personas_dir = self.root / "personas"
        if personas_dir.is_dir():
            for path in sorted(personas_dir.glob("*.md")):
                persona = _try_load(_load_persona, path)
                if persona is not None:
                    self.personas[persona.stage] = persona
                    self.sources.append(str(path))
        commands_dir = self.root / "commands"
        if commands_dir.is_dir():
            for path in sorted(commands_dir.glob("*.md")):
                command = _try_load(_load_command, path)
                if command is not None:
                    self.commands[command.id] = command
                    self.sources.append(str(path))

    def _merge_kernel(self, path: Path) -> None:
        raw, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        if not raw:
            return  # a broken kernel file must not take the API down
        from agentos.specs import RoutingRule

        rules = []
        for entry in raw.get("routing", []):
            try:
                rule = RoutingRule.model_validate(entry)
            except ValidationError:
                continue
            if rule.stage or rule.command:
                rules.append(rule)
        self.kernel = KernelSpec(
            name=str(raw.get("name") or self.kernel.name),
            team_charter=str(raw.get("team_charter") or "") or self.kernel.team_charter,
            identity=body,
            routing=rules,
        )

    # --- accessors used by the pipeline -----------------------------------

    def has_persona(self, stage: Stage) -> bool:
        return stage in self.personas

    def system_prompt(self, stage: Stage) -> str:
        """Shared team charter first, then the persona's own charter."""
        persona = self.personas.get(stage)
        if persona is None:
            return _builtin_prompt(stage)
        parts = [p for p in (self.kernel.team_charter, persona.charter) if p]
        return "\n\n".join(parts)

    def policy(self, stage: Stage) -> ModelPolicy | None:
        persona = self.personas.get(stage)
        return persona.model if persona else None

    def command(self, name: str) -> CommandSpec | None:
        return self.commands.get(name)

    # --- routing ----------------------------------------------------------

    def route(self, task: str) -> RouteDecision:
        rule, rationale = self.kernel.route(task)
        if rule is None:
            return RouteDecision(
                task=task, stages=list(STAGE_ORDER), rationale=rationale
            )
        if rule.command and (cmd := self.command(rule.command)):
            return RouteDecision(
                task=task,
                intent=rule.intent,
                target_kind="command",
                stages=cmd.ordered_stages(),
                command=cmd.id,
                rationale=f"{rationale}; command '{cmd.id}' runs {[s.value for s in cmd.ordered_stages()]}",
            )
        if rule.stage:
            return RouteDecision(
                task=task,
                intent=rule.intent,
                target_kind="pipeline",
                stages=[rule.stage],
                rationale=f"{rationale}; single-agent run of {rule.stage.value}",
            )
        return RouteDecision(task=task, stages=list(STAGE_ORDER), rationale=rationale)

    def summary(self) -> dict:
        return {
            "name": self.kernel.name,
            "root": str(self.root) if self.root else "",
            "agents": [p.summary() for stage, p in sorted(self.personas.items())],
            "commands": [c.model_dump(mode="json") for c in self.commands.values()],
            "routing": [r.model_dump(mode="json") for r in self.kernel.routing],
        }


# --- built-in fallbacks -----------------------------------------------------


def _default_team_charter() -> str:
    from agents.nodes import TEAM_CHARTER

    return TEAM_CHARTER


def _builtin_prompt(stage: Stage) -> str:
    from agents.nodes import SYSTEM_PROMPTS

    return SYSTEM_PROMPTS[stage]
