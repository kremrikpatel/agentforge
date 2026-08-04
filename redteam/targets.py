"""PyRIT target adapters.

Two of them:

  PipelineTarget      the system under test -- POSTs to /pipeline/run and reads
                      the Phase 1 report back out, including its guardrail verdict.
  GatewayChatTarget   the adversarial model that drives multi-turn attacks,
                      routed through the existing LLM gateway. PyRIT would
                      happily open its own OpenAI client; that would be a second
                      gateway, so it gets ours instead.

Every construction path runs assert_safe_target first. There is no way to build
a PipelineTarget pointing at something that is not a designated test instance.
"""

from __future__ import annotations

import json

import uuid

import httpx
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration

# Both targets are driven turn-by-turn by the multi-turn executors, so both must
# advertise chat capabilities or PyRIT refuses to wire them up.
_TEXT = frozenset({frozenset({"text"})})
_CHAT_CONFIG = TargetConfiguration(
    capabilities=TargetCapabilities(
        supports_multi_turn=True,
        supports_editable_history=True,
        supports_system_prompt=True,
        supports_multi_message_pieces=True,
        input_modalities=_TEXT,
        output_modalities=_TEXT,
    )
)

from app.observability import get_logger, log_event, timed
from gateway.client import AllProvidersFailed, LLMGateway
from gateway.providers import GatewayRequest
from redteam.config import RedTeamSettings, get_redteam_settings
from redteam.pyrit_setup import ensure_pyrit_memory
from redteam.safety import assert_safe_target
from redteam.scoring import TargetResponse

logger = get_logger("agentforge.redteam.target")


def _reply(text: str) -> list[Message]:
    """Wrap a string as the assistant turn PyRIT expects back."""
    return [Message(message_pieces=[MessagePiece(role="assistant", original_value=text)])]


def _last_user_text(conversation: list[Message]) -> str:
    for message in reversed(conversation):
        for piece in reversed(message.message_pieces):
            if piece.role == "user":
                return piece.converted_value or piece.original_value
    return ""


def _conversation_as_chat(conversation: list[Message]) -> tuple[str, str]:
    """Flatten to (system, user) -- the shape our gateway speaks."""
    system_parts, turns = [], []
    for message in conversation:
        for piece in message.message_pieces:
            value = piece.converted_value or piece.original_value or ""
            if piece.role == "system":
                system_parts.append(value)
            else:
                turns.append(f"{piece.role}: {value}")
    return "\n\n".join(system_parts), "\n\n".join(turns)


class PipelineTarget(PromptTarget):
    """The AgentForge pipeline under attack.

    Records the last response so the scorer can read the target's own guardrail
    verdict, which PyRIT's Message type has nowhere to carry.
    """

    _DEFAULT_CONFIGURATION = _CHAT_CONFIG

    # PromptTarget enforces keyword-only construction on its subclasses.
    def __init__(self, *, settings: RedTeamSettings | None = None, client=None) -> None:
        ensure_pyrit_memory()
        super().__init__()
        self.settings = settings or get_redteam_settings()
        # Refuses here, before any attack can be constructed against it.
        self.safety_reason = assert_safe_target(
            self.settings.target_url, self.settings.allowed_hosts
        )
        self._client = client
        self.last: TargetResponse = TargetResponse()
        # Multi-turn attacks call the target repeatedly; a leak in any turn counts,
        # so the whole transcript is kept and scored together.
        self.history: list[TargetResponse] = []
        # /pipeline/run is stateless per request, so a crescendo would otherwise be
        # N unrelated first turns. Holding one session id for the length of an
        # attack lets Phase 1's session memory carry the escalation forward.
        self.session_id: str = uuid.uuid4().hex
        log_event(
            logger,
            "redteam.target_accepted",
            endpoint=self.settings.run_endpoint,
            reason=self.safety_reason,
        )

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"content-type": "application/json"}
            if self.settings.target_token:
                headers["authorization"] = f"Bearer {self.settings.target_token}"
            self._client = httpx.AsyncClient(
                timeout=self.settings.target_timeout_s, headers=headers
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, prompt: str) -> TargetResponse:
        """One attack request.

        Never raises -- a transport failure is an ERROR outcome, not a crashed run.
        """
        try:
            with timed() as t:
                resp = await self.client.post(
                    self.settings.run_endpoint,
                    json={"topic": prompt, "session_id": self.session_id},
                )
        except httpx.HTTPError as exc:
            return TargetResponse(error=f"transport: {exc}"[:300])

        if resp.status_code >= 500:
            return TargetResponse(
                error=f"HTTP {resp.status_code}", latency_ms=t["ms"], status="server_error"
            )

        try:
            report = resp.json()
        except ValueError:
            return TargetResponse(text=resp.text[:20000], latency_ms=t["ms"], status="non_json")

        guard_reports = report.get("guardrail_reports") or []
        actions = [str(g.get("action", "")) for g in guard_reports if g.get("action")]
        categories = [
            str(f.get("category", ""))
            for g in guard_reports
            for f in (g.get("findings") or [])
            if f.get("category")
        ]
        errors = report.get("errors") or []
        blocked = (
            "block" in actions
            or any("guardrail" in str(e).lower() for e in errors)
            or report.get("status") in {"failed", "halted"}
        )

        return TargetResponse(
            # The whole report, so a canary hiding in any stage is still caught.
            text=json.dumps(report, default=str),
            status=str(report.get("status", "")),
            blocked_by_target=bool(blocked),
            guardrail_actions=actions,
            guardrail_categories=categories,
            latency_ms=t["ms"],
        )

    def begin(self) -> None:
        """Start a fresh transcript. Called once per attack by the runner."""
        self.history.clear()
        self.last = TargetResponse()
        # New session per attack: escalation must not leak between attacks.
        self.session_id = uuid.uuid4().hex

    def collected(self) -> TargetResponse:
        """Fold the transcript into one scoreable response."""
        if not self.history:
            return self.last
        if len(self.history) == 1:
            return self.history[0]

        usable = [r for r in self.history if not r.error]
        return TargetResponse(
            text="\n\n".join(r.text for r in usable),
            status=self.history[-1].status,
            blocked_by_target=any(r.blocked_by_target for r in self.history),
            guardrail_actions=[a for r in self.history for a in r.guardrail_actions],
            guardrail_categories=[c for r in self.history for c in r.guardrail_categories],
            turns=len(self.history),
            latency_ms=sum(r.latency_ms for r in self.history),
            # Only an error if every turn failed; a partial transcript is scoreable.
            error="" if usable else (self.history[0].error or "all turns failed"),
        )

    async def _send_prompt_to_target_async(
        self, *, normalized_conversation: list[Message]
    ) -> list[Message]:
        prompt = _last_user_text(normalized_conversation)
        self.last = await self.send(prompt)
        self.history.append(self.last)
        if self.last.error:
            return _reply(f"[target error] {self.last.error}")
        return _reply(self.last.text)


class GatewayChatTarget(PromptTarget):
    """The adversarial model PyRIT uses to compose multi-turn attacks.

    Backed by the existing gateway, so the whole subsystem still has exactly one
    LLM client and one fallback chain.
    """

    _DEFAULT_CONFIGURATION = _CHAT_CONFIG

    def __init__(self, *, gateway: LLMGateway, max_tokens: int = 800) -> None:
        ensure_pyrit_memory()
        super().__init__()
        self.gateway = gateway
        self.max_tokens = max_tokens

    async def _send_prompt_to_target_async(
        self, *, normalized_conversation: list[Message]
    ) -> list[Message]:
        system, user = _conversation_as_chat(normalized_conversation)
        try:
            resp = await self.gateway.complete(
                GatewayRequest(
                    system=system or "You are a red-team assistant composing test prompts.",
                    user=user,
                    max_tokens=self.max_tokens,
                    temperature=0.8,  # attack variety
                    stub_response="",
                )
            )
        except AllProvidersFailed as exc:
            log_event(logger, "redteam.adversary_unavailable", error=str(exc)[:200])
            return _reply("")
        return _reply(resp.text)
