"""Severity -> channel routing.

warning fires Slack only; critical fires Slack plus email and the PagerDuty
stub. A silent PagerDuty (no routing key configured, or the stub's
not-implemented no-op) simply contributes nothing to `delivered` -- it is
listed as a route because the brief calls it an optional *additional*
channel for critical, not because it does anything today.
"""

from __future__ import annotations

from monitoring.notifiers import Notifier
from monitoring.schemas import Alert, Severity

ROUTES: dict[Severity, tuple[str, ...]] = {
    Severity.WARNING: ("slack",),
    Severity.CRITICAL: ("slack", "email", "pagerduty"),
}


def route(alert: Alert, notifiers: dict[str, Notifier]) -> list[str]:
    """Send to every channel this severity routes to. Returns the channels
    that actually delivered -- callers use this to log what happened, not to
    decide whether the alert should be considered handled."""
    delivered = []
    for channel in ROUTES.get(alert.severity, ()):
        notifier = notifiers.get(channel)
        if notifier and notifier.send(alert):
            delivered.append(channel)
    return delivered
