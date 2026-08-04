"""Admin UI settings.

The admin password has no default. An admin console that ships with a known
credential is worse than one that refuses to start, so if WEB_ADMIN_PASSWORD is
unset the login route rejects every attempt and says why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import _env, _env_bool, _env_int


@dataclass(frozen=True)
class WebSettings:
    # --- auth --------------------------------------------------------------
    admin_user: str = field(default_factory=lambda: _env("WEB_ADMIN_USER", "admin"))
    # No default on purpose -- see the module docstring.
    admin_password: str = field(default_factory=lambda: _env("WEB_ADMIN_PASSWORD"))
    session_ttl_s: int = field(default_factory=lambda: _env_int("WEB_SESSION_TTL_S", 28800))
    cookie_name: str = field(default_factory=lambda: _env("WEB_COOKIE_NAME", "af_admin"))
    # Set true when served over HTTPS; left off so local http still works.
    cookie_secure: bool = field(default_factory=lambda: _env_bool("WEB_COOKIE_SECURE", False))

    # --- upstream backend --------------------------------------------------
    backend_url: str = field(
        default_factory=lambda: _env("WEB_BACKEND_URL", "http://localhost:8000")
    )
    backend_timeout_s: int = field(
        default_factory=lambda: _env_int("WEB_BACKEND_TIMEOUT_S", 180)
    )

    # --- serving -----------------------------------------------------------
    host: str = field(default_factory=lambda: _env("WEB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("WEB_PORT", 8200))
    redteam_url: str = field(
        default_factory=lambda: _env("WEB_REDTEAM_URL", "http://127.0.0.1:8100")
    )

    @property
    def auth_configured(self) -> bool:
        return bool(self.admin_password)

    @property
    def run_endpoint(self) -> str:
        return f"{self.backend_url.rstrip('/')}/pipeline/run"

    @property
    def stream_endpoint(self) -> str:
        return f"{self.backend_url.rstrip('/')}/pipeline/stream"

    @property
    def health_endpoint(self) -> str:
        return f"{self.backend_url.rstrip('/')}/health"


def get_web_settings() -> WebSettings:
    return WebSettings()
