"""Wires rules + suppression + routing + notifiers + history.

Async because its state and history backends are (Redis and Postgres), and
because all three call sites already run inside an event loop:
`redteam/runner.py`'s `main_async`, the sweep CLI, and the console's FastAPI
handlers. The notifiers stay synchronous -- `httpx.Client` and `smtplib` are
blocking libraries -- so they are pushed to a worker thread with
`asyncio.to_thread`, the same escape hatch `guardrails/engine.py` uses.

Order matters in `_dispatch`: notify first, record second. History is a
side effect of alerting, never a precondition, so a dead Postgres costs the
audit trail and not the page.
"""

from __future__ import annotations

import asyncio

from app.observability import get_logger, log_event
from instrumentation.schemas import Trajectory
from monitoring import rules
from monitoring.config import MonitoringSettings, get_monitoring_settings
from monitoring.dedup import AlertState, InMemoryAlertState, build_state
from monitoring.notifiers import Notifier, build_notifiers
from monitoring.router import route
from monitoring.schemas import Alert
from monitoring.store import AlertStore, InMemoryAlertStore, build_alert_store
from redteam.thresholds import ThresholdResult

logger = get_logger("agentforge.monitoring.engine")

# Each name must match the MonitoringSettings attribute holding that rule's RuleConfig.
_TRAJECTORY_RULES = (
    ("guardrail_intervention_rate", rules.guardrail_intervention_rate),
    ("escalation_rate", rules.escalation_rate),
)


class MonitoringEngine:
    def __init__(
        self,
        settings: MonitoringSettings | None = None,
        notifiers: dict[str, Notifier] | None = None,
        state: AlertState | None = None,
        store: AlertStore | None = None,
    ) -> None:
        self.settings = settings or get_monitoring_settings()
        self.notifiers = notifiers or build_notifiers(self.settings)
        self.state = state or InMemoryAlertState(self.settings.dedup_window_s)
        self.store = store or InMemoryAlertStore()

    @classmethod
    async def build(cls, settings: MonitoringSettings | None = None) -> "MonitoringEngine":
        """Resolve the configured backends, falling back when either is down."""
        settings = settings or get_monitoring_settings()
        return cls(
            settings=settings,
            state=await build_state(settings.dedup_window_s, settings.state_backend),
            store=await build_alert_store(),
        )

    async def _dispatch(self, alert: Alert) -> Alert | None:
        """Suppress, route, then record. Returns the alert iff it was sent."""
        if not await self.state.should_notify(alert.fingerprint):
            return None

        channels = await asyncio.to_thread(route, alert, self.notifiers)
        log_event(
            logger,
            "monitoring.alert_fired",
            rule=alert.rule,
            severity=alert.severity.value,
            fingerprint=alert.fingerprint,
            channels=channels,
        )
        await self.store.record(alert, channels)
        return alert

    async def evaluate_trajectories(self, trajectories: list[Trajectory]) -> list[Alert]:
        """Run every trajectory-window rule once over the given window."""
        fired: list[Alert] = []
        for name, evaluator in _TRAJECTORY_RULES:
            alert = evaluator(trajectories, getattr(self.settings, name))
            if alert and await self._dispatch(alert):
                fired.append(alert)

        for alert in rules.circuit_breaker_trips(
            trajectories, self.settings.circuit_breaker_trips
        ):
            if await self._dispatch(alert):
                fired.append(alert)

        return fired

    async def notify_redteam_result(self, result: ThresholdResult) -> Alert | None:
        """Called from redteam/runner.py after check_thresholds, so a failed
        gate sends a real notification instead of only a non-zero exit code."""
        alert = rules.redteam_block_rate(result, self.settings.redteam_severity)
        return await self._dispatch(alert) if alert else None


async def notify_redteam_result(result: ThresholdResult) -> Alert | None:
    """One lazy import gives redteam/runner.py a working notifier without it
    owning an engine lifecycle. Built per call: the red-team suite is a batch
    job that exits, so there is no process to cache an engine in."""
    engine = await MonitoringEngine.build()
    return await engine.notify_redteam_result(result)
