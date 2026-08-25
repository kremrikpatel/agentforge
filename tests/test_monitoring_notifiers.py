"""Notifier delivery, severity-based routing, and the full engine dispatch
path -- httpx and smtplib are faked out, so nothing here touches a network."""

from __future__ import annotations

import dataclasses

from instrumentation.schemas import RewardSignal, Trajectory
from monitoring.config import get_monitoring_settings
from monitoring.dedup import InMemoryAlertState
from monitoring.engine import MonitoringEngine
from monitoring.notifiers import EmailNotifier, PagerDutyNotifier, SlackNotifier
from monitoring.router import route
from monitoring.schemas import Alert, Severity
from monitoring.store import InMemoryAlertStore
from redteam.thresholds import check_thresholds

from tests.test_redteam_thresholds import summary_with


def _alert(severity: Severity) -> Alert:
    return Alert(rule="test_rule", severity=severity, summary="summary", detail="detail")


def _trajectory(run_id: str, **reward_kwargs) -> Trajectory:
    return Trajectory(run_id=run_id, reward=RewardSignal(**reward_kwargs))


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None


class _FakeHttpxClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict):
        self.calls.append((url, json))
        return _FakeResponse()


class _FakeSmtp:
    instances: list["_FakeSmtp"] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.started_tls = False
        self.logged_in = None
        self.sent: list = []
        _FakeSmtp.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.logged_in = (username, password)

    def send_message(self, message):
        self.sent.append(message)


# --- individual notifiers ---------------------------------------------------


def test_slack_notifier_posts_to_webhook():
    client = _FakeHttpxClient()
    notifier = SlackNotifier("https://hooks.slack.test/x", client=client)

    assert notifier.send(_alert(Severity.WARNING)) is True
    assert client.calls[0][0] == "https://hooks.slack.test/x"
    assert "summary" in client.calls[0][1]["text"]


def test_slack_notifier_noop_without_webhook_url():
    notifier = SlackNotifier("", client=_FakeHttpxClient())
    assert notifier.send(_alert(Severity.WARNING)) is False


def test_email_notifier_sends_via_smtp():
    _FakeSmtp.instances.clear()
    notifier = EmailNotifier(
        "smtp.test", 587, "user", "pass", "alerts@agentforge.test", "oncall@agentforge.test",
        use_tls=True, smtp_cls=_FakeSmtp,
    )

    assert notifier.send(_alert(Severity.CRITICAL)) is True
    sent = _FakeSmtp.instances[0]
    assert sent.started_tls is True
    assert sent.logged_in == ("user", "pass")
    assert len(sent.sent) == 1


def test_email_notifier_noop_without_recipient():
    notifier = EmailNotifier(
        "smtp.test", 587, "", "", "alerts@agentforge.test", "", smtp_cls=_FakeSmtp
    )
    assert notifier.send(_alert(Severity.CRITICAL)) is False


def test_pagerduty_stub_never_delivers():
    assert PagerDutyNotifier("routing-key").send(_alert(Severity.CRITICAL)) is False


# --- severity-based routing --------------------------------------------------


def _notifiers(slack_client: _FakeHttpxClient) -> dict:
    return {
        "slack": SlackNotifier("https://hooks.slack.test/x", client=slack_client),
        "email": EmailNotifier(
            "smtp.test", 587, "", "", "alerts@agentforge.test", "oncall@agentforge.test",
            smtp_cls=_FakeSmtp,
        ),
        "pagerduty": PagerDutyNotifier(""),
    }


def test_warning_routes_to_slack_only():
    slack_client = _FakeHttpxClient()
    _FakeSmtp.instances.clear()

    delivered = route(_alert(Severity.WARNING), _notifiers(slack_client))

    assert delivered == ["slack"]
    assert len(slack_client.calls) == 1
    assert _FakeSmtp.instances == []


def test_critical_routes_to_slack_and_email():
    slack_client = _FakeHttpxClient()
    _FakeSmtp.instances.clear()

    delivered = route(_alert(Severity.CRITICAL), _notifiers(slack_client))

    assert delivered == ["slack", "email"]
    assert len(slack_client.calls) == 1
    assert len(_FakeSmtp.instances) == 1


# --- full engine dispatch (rules -> dedup -> router -> notifiers) ----------


async def test_engine_dispatches_warning_to_slack_only_and_dedupes_repeat_fires():
    slack_client = _FakeHttpxClient()
    _FakeSmtp.instances.clear()
    engine = MonitoringEngine(
        notifiers=_notifiers(slack_client), state=InMemoryAlertState(dedup_window_s=900.0)
    )
    trajectories = [
        _trajectory(f"r{i}", guardrail_interventions=1 if i < 4 else 0) for i in range(5)
    ]

    fired_once = await engine.evaluate_trajectories(trajectories)
    fired_twice = await engine.evaluate_trajectories(trajectories)

    assert len(fired_once) == 1
    assert fired_once[0].rule == "guardrail_intervention_rate"
    assert len(slack_client.calls) == 1
    assert _FakeSmtp.instances == []  # warning severity never reaches email
    assert fired_twice == []  # deduped: same fingerprint, inside the window


async def test_engine_dispatches_critical_redteam_alert_to_slack_and_email():
    slack_client = _FakeHttpxClient()
    _FakeSmtp.instances.clear()
    settings = dataclasses.replace(get_monitoring_settings(), redteam_severity=Severity.CRITICAL)
    engine = MonitoringEngine(
        settings=settings,
        notifiers=_notifiers(slack_client),
        state=InMemoryAlertState(dedup_window_s=900.0),
    )
    result = check_thresholds(summary_with(jailbreak=1.0, xpia=0.4), minimum=0.9)

    fired = await engine.notify_redteam_result(result)

    assert fired is not None
    assert fired.severity is Severity.CRITICAL
    assert len(slack_client.calls) == 1
    assert len(_FakeSmtp.instances) == 1


async def test_engine_records_fired_alerts_to_the_store():
    """History is written after delivery, with the channels that succeeded."""
    slack_client = _FakeHttpxClient()
    _FakeSmtp.instances.clear()
    store = InMemoryAlertStore()
    engine = MonitoringEngine(
        notifiers=_notifiers(slack_client),
        state=InMemoryAlertState(dedup_window_s=900.0),
        store=store,
    )
    trajectories = [
        _trajectory(f"r{i}", guardrail_interventions=1 if i < 4 else 0) for i in range(5)
    ]

    await engine.evaluate_trajectories(trajectories)
    rows = await store.recent()

    assert len(rows) == 1
    assert rows[0]["rule"] == "guardrail_intervention_rate"
    assert rows[0]["channels"] == ["slack"]


async def test_a_silenced_alert_sends_nothing_and_records_nothing():
    """Acceptance: silenced conditions do not re-fire notifications."""
    slack_client = _FakeHttpxClient()
    store = InMemoryAlertStore()
    state = InMemoryAlertState(dedup_window_s=0.0)  # dedup off, so silence is what is tested
    engine = MonitoringEngine(
        notifiers=_notifiers(slack_client), state=state, store=store
    )
    trajectories = [
        _trajectory(f"r{i}", guardrail_interventions=1 if i < 4 else 0) for i in range(5)
    ]
    await state.silence("guardrail_intervention_rate", duration_s=3600)

    fired = await engine.evaluate_trajectories(trajectories)

    assert fired == []
    assert slack_client.calls == []
    assert await store.recent() == []
