"""Refuses to attack anything that is not a designated test/staging target.

This is the guard the whole subsystem hangs off. A red-team run sends
deliberately hostile traffic; pointing it at production would be an incident, so
the default is deny and every path to "allowed" is explicit.

Three independent checks, all of which must pass:
  1. a deny-list on the hostname, which no allow-list can override
  2. a public-IP check, so a literal address cannot slip through
  3. an allow-list: built-in local/staging shapes, or an exact opt-in host
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

# Substrings that disqualify a host no matter what else permits it. Checked
# first and never overridable -- "staging.prod.example.com" is still refused.
DENY_SUBSTRINGS: tuple[str, ...] = (
    "prod",
    "production",
    "live",
    "www.",
    "public",
    "customer",
)

# Hosts that are always safe: this machine.
LOCAL_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}
)

# Hostname shapes that designate a non-production environment.
STAGING_SUFFIXES: tuple[str, ...] = (
    ".test",
    ".local",
    ".localhost",
    ".internal",
    ".invalid",
    ".staging",
    ".svc.cluster.local",
)
STAGING_PREFIXES: tuple[str, ...] = ("staging.", "stage.", "test.", "qa.", "dev.")


class UnsafeTargetError(RuntimeError):
    """The configured target is not a designated test/staging endpoint."""


def _is_public_ip(host: str) -> bool:
    """True only for a literal address that is globally routable."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # a name, not an address
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved)


def classify_host(host: str, allowed_hosts: frozenset[str] = frozenset()) -> tuple[bool, str]:
    """Return (safe, reason). Reason explains the verdict either way."""
    if not host:
        return False, "target URL has no hostname"

    lowered = host.lower()

    banned = [s for s in DENY_SUBSTRINGS if s in lowered]
    if banned:
        return False, f"hostname contains a production marker: {', '.join(banned)}"

    if lowered in LOCAL_HOSTS:
        return True, "local host"

    # An exact opt-in, never a pattern -- a wildcard here would defeat the guard.
    if lowered in {h.lower() for h in allowed_hosts}:
        return True, "explicitly allow-listed via REDTEAM_ALLOWED_HOSTS"

    if _is_public_ip(lowered):
        return False, "refuses to target a public IP address"

    if lowered.endswith(STAGING_SUFFIXES) or lowered.startswith(STAGING_PREFIXES):
        return True, "hostname designates a test/staging environment"

    return False, (
        "host is neither local, staging-shaped, nor explicitly allow-listed; "
        "add it to REDTEAM_ALLOWED_HOSTS if it really is a test instance"
    )


def _refuse(reason: str, url: str) -> str:
    raise UnsafeTargetError(
        f"refusing to run attacks against {url!r}: {reason}. "
        "The red-team suite only runs against a designated test/staging instance."
    )


def assert_safe_target(url: str, allowed_hosts: frozenset[str] = frozenset()) -> str:
    """Raise UnsafeTargetError unless `url` is a designated test/staging target."""
    if not url:
        return _refuse("no target configured; set REDTEAM_TARGET_URL", url)

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return _refuse(f"unsupported scheme {parsed.scheme!r}", url)

    safe, reason = classify_host(parsed.hostname or "", allowed_hosts)
    if not safe:
        return _refuse(reason, url)
    return reason
