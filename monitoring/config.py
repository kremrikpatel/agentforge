"""Alert rule thresholds, routing, and notification-channel settings.

Rule definitions are data (`RuleConfig` instances), not branches in `rules.py`
-- moving a rule's threshold or severity is a config change here, not a code
change there. Mirrors the env-driven `dataclass` pattern `redteam/config.py`
already uses, so a new operator reads one shape of settings module instead of
learning a second one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import _env, _env_bool, _env_float, _env_int
from monitoring.schemas import Severity


@dataclass(frozen=True)
class RuleConfig:
    """Threshold and identity for one trajectory-window rule.

    `min_samples` keeps a rule quiet until the window has enough runs to mean
    something -- one failed run out of one is not a 25% escalation rate worth
    paging anyone. `breaker_failures` is only read by `circuit_breaker_trips`.
    """

    name: str
    severity: Severity
    threshold: float
    min_samples: int = 5
    breaker_failures: int = 3


@dataclass(frozen=True)
class MonitoringSettings:
    # --- Slack -----------------------------------------------------------
    slack_webhook_url: str = field(default_factory=lambda: _env("SLACK_WEBHOOK_URL"))

    # --- Email (SMTP) ------------------------------------------------------
    smtp_host: str = field(default_factory=lambda: _env("SMTP_HOST"))
    smtp_port: int = field(default_factory=lambda: _env_int("SMTP_PORT", 587))
    smtp_username: str = field(default_factory=lambda: _env("SMTP_USERNAME"))
    smtp_password: str = field(default_factory=lambda: _env("SMTP_PASSWORD"))
    smtp_use_tls: bool = field(default_factory=lambda: _env_bool("SMTP_USE_TLS", True))
    alert_email_from: str = field(default_factory=lambda: _env("ALERT_EMAIL_FROM"))
    alert_email_to: str = field(default_factory=lambda: _env("ALERT_EMAIL_TO"))

    # --- PagerDuty (stub) ----------------------------------------------------
    pagerduty_routing_key: str = field(
        default_factory=lambda: _env("PAGERDUTY_ROUTING_KEY")
    )

    # --- Alert hygiene -------------------------------------------------------
    # A fingerprint that keeps firing inside this window sends one notification,
    # not one per evaluation -- the flapping-alert case.
    dedup_window_s: float = field(
        default_factory=lambda: _env_float("ALERT_DEDUP_WINDOW_S", 900.0)
    )
    # "redis" shares suppression across the sweep CronJob and the console;
    # "memory" is single-process and only correct for local runs and tests.
    state_backend: str = field(
        default_factory=lambda: _env("ALERT_STATE_BACKEND", "redis").lower()
    )
    # How long the console's "Silence" button mutes a fingerprint for.
    silence_default_s: float = field(
        default_factory=lambda: _env_float("ALERT_SILENCE_DEFAULT_S", 3600.0)
    )

    # --- Trajectory-window rules (Phase 6 data) -------------------------------
    guardrail_intervention_rate: RuleConfig = field(
        default_factory=lambda: RuleConfig(
            "guardrail_intervention_rate",
            Severity.WARNING,
            threshold=_env_float("ALERT_GUARDRAIL_RATE_MAX", 0.3),
            min_samples=_env_int("ALERT_GUARDRAIL_MIN_SAMPLES", 5),
        )
    )
    escalation_rate: RuleConfig = field(
        default_factory=lambda: RuleConfig(
            "escalation_rate",
            Severity.WARNING,
            threshold=_env_float("ALERT_ESCALATION_RATE_MAX", 0.25),
            min_samples=_env_int("ALERT_ESCALATION_MIN_SAMPLES", 5),
        )
    )
    circuit_breaker_trips: RuleConfig = field(
        default_factory=lambda: RuleConfig(
            "circuit_breaker_trips",
            Severity.CRITICAL,
            threshold=1.0,
            breaker_failures=_env_int("ALERT_BREAKER_FAILURES", 3),
        )
    )

    # --- Red-team gate (Phase 4 data, own threshold lives in RedTeamSettings) --
    redteam_severity: Severity = field(
        default_factory=lambda: Severity(_env("ALERT_REDTEAM_SEVERITY", "critical"))
    )


def get_monitoring_settings() -> MonitoringSettings:
    """Read settings fresh, same rationale as `app.config.get_settings`."""
    return MonitoringSettings()
