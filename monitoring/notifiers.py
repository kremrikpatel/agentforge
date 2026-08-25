"""Pluggable notification channels behind one `Notifier` interface.

Slack uses httpx -- already a core dependency (every provider adapter uses
it). Email uses smtplib -- stdlib. That covers both channels the brief asks
for with zero new dependencies. PagerDuty is a stub behind the same
interface so a real Events API v2 integration is a new class, not a new
call site, if it's ever wanted.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Protocol

import httpx

from app.observability import get_logger, log_event
from monitoring.config import MonitoringSettings
from monitoring.schemas import Alert

logger = get_logger("agentforge.monitoring.notifiers")


class Notifier(Protocol):
    def send(self, alert: Alert) -> bool:
        """Deliver one alert. Returns whether delivery succeeded."""


class SlackNotifier:
    def __init__(self, webhook_url: str, client: httpx.Client | None = None) -> None:
        self.webhook_url = webhook_url
        self._client = client

    def send(self, alert: Alert) -> bool:
        if not self.webhook_url:
            return False
        payload = {
            "text": f"[{alert.severity.value.upper()}] {alert.summary}\n{alert.detail}".strip()
        }
        client = self._client or httpx.Client(timeout=10.0)
        try:
            resp = client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            log_event(logger, "monitoring.slack_failed", rule=alert.rule, error=str(exc))
            return False
        finally:
            if self._client is None:
                client.close()


class EmailNotifier:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        sender: str,
        recipient: str,
        use_tls: bool = True,
        smtp_cls: type[smtplib.SMTP] = smtplib.SMTP,
    ) -> None:
        self.host, self.port = host, port
        self.username, self.password = username, password
        self.sender, self.recipient = sender, recipient
        self.use_tls = use_tls
        self._smtp_cls = smtp_cls

    def send(self, alert: Alert) -> bool:
        if not (self.host and self.sender and self.recipient):
            return False
        message = EmailMessage()
        message["Subject"] = f"[AgentForge/{alert.severity.value}] {alert.summary}"
        message["From"] = self.sender
        message["To"] = self.recipient
        message.set_content(alert.detail or alert.summary)
        try:
            with self._smtp_cls(self.host, self.port, timeout=10.0) as smtp:
                if self.use_tls:
                    smtp.starttls()
                if self.username:
                    smtp.login(self.username, self.password)
                smtp.send_message(message)
            return True
        except (smtplib.SMTPException, OSError) as exc:
            log_event(logger, "monitoring.email_failed", rule=alert.rule, error=str(exc))
            return False


class PagerDutyNotifier:
    """Stub behind the `Notifier` interface. No real integration -- wire the
    Events API v2 here if PagerDuty becomes a real requirement."""

    def __init__(self, routing_key: str) -> None:
        self.routing_key = routing_key

    def send(self, alert: Alert) -> bool:
        if not self.routing_key:
            return False
        log_event(logger, "monitoring.pagerduty_not_implemented", rule=alert.rule)
        return False


def build_notifiers(settings: MonitoringSettings) -> dict[str, Notifier]:
    return {
        "slack": SlackNotifier(settings.slack_webhook_url),
        "email": EmailNotifier(
            settings.smtp_host,
            settings.smtp_port,
            settings.smtp_username,
            settings.smtp_password,
            settings.alert_email_from,
            settings.alert_email_to,
            settings.smtp_use_tls,
        ),
        "pagerduty": PagerDutyNotifier(settings.pagerduty_routing_key),
    }
