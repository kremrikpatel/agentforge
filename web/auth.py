"""Session auth for the admin console.

Deliberately small: an opaque random cookie against a server-side session table.
That avoids signed-cookie machinery (itsdangerous) entirely, which keeps this
phase at zero new dependencies, and it means logout actually revokes -- a signed
cookie stays valid until it expires whatever the server thinks.

Two limits worth knowing, both fine for this phase and both flagged in the README:
  - sessions live in this process, so they drop on restart and do not work
    across replicas. Move to Redis when the console runs more than once.
  - one shared admin credential, no user accounts, no SSO. Enterprise auth is a
    later phase; this is the gate, not the identity system.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Response, status

from app.observability import get_logger, log_event
from web.config import WebSettings, get_web_settings

logger = get_logger("agentforge.web.auth")


@dataclass
class Session:
    token: str
    user: str
    created_at: float
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at


@dataclass
class SessionStore:
    settings: WebSettings = field(default_factory=get_web_settings)
    _sessions: dict[str, Session] = field(default_factory=dict)

    def _sweep(self) -> None:
        for token in [t for t, s in self._sessions.items() if s.expired]:
            self._sessions.pop(token, None)

    def authenticate(self, username: str, password: str) -> Session | None:
        """Constant-time credential check. Refuses everything when unconfigured."""
        if not self.settings.auth_configured:
            log_event(logger, "web.login_refused", reason="WEB_ADMIN_PASSWORD is not set")
            return None

        # Compare both halves regardless, so a wrong username and a wrong
        # password take the same time.
        user_ok = secrets.compare_digest(username or "", self.settings.admin_user)
        password_ok = secrets.compare_digest(password or "", self.settings.admin_password)
        if not (user_ok and password_ok):
            log_event(logger, "web.login_failed", user=username[:40])
            return None

        self._sweep()
        now = time.time()
        session = Session(
            token=secrets.token_urlsafe(32),
            user=username,
            created_at=now,
            expires_at=now + self.settings.session_ttl_s,
        )
        self._sessions[session.token] = session
        log_event(logger, "web.login", user=username)
        return session

    def get(self, token: str | None) -> Session | None:
        if not token:
            return None
        session = self._sessions.get(token)
        if session is None:
            return None
        if session.expired:
            self._sessions.pop(token, None)
            return None
        return session

    def revoke(self, token: str | None) -> None:
        if token:
            self._sessions.pop(token, None)

    @property
    def active(self) -> int:
        self._sweep()
        return len(self._sessions)


SESSIONS = SessionStore()


def issue_cookie(response: Response, session: Session, settings: WebSettings) -> None:
    response.set_cookie(
        settings.cookie_name,
        session.token,
        max_age=settings.session_ttl_s,
        httponly=True,            # not reachable from JS, so XSS cannot lift it
        samesite="lax",           # blocks cross-site form posts
        secure=settings.cookie_secure,
        path="/",
    )


def clear_cookie(response: Response, settings: WebSettings) -> None:
    response.delete_cookie(settings.cookie_name, path="/")


def current_session(request: Request) -> Session | None:
    settings = get_web_settings()
    return SESSIONS.get(request.cookies.get(settings.cookie_name))


def require_admin(request: Request) -> Session:
    """Dependency for every admin route and API call."""
    session = current_session(request)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )
    return session
