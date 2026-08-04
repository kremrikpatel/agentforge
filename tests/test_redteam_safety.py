"""The target guard. Default deny; every route to 'allowed' is explicit."""

from __future__ import annotations

import pytest

from redteam.safety import UnsafeTargetError, assert_safe_target, classify_host


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000",
        "http://127.0.0.1:8000/pipeline/run",
        "https://staging.acme.internal",
        "http://qa.acme.test",
        "http://dev.example.local",
        "http://agentforge.svc.cluster.local:8000",
    ],
)
def test_designated_test_targets_are_allowed(url):
    assert assert_safe_target(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.production.example.com",
        "https://live.acme.io",
        "https://www.acme.com",
        "https://customer-portal.acme.io",
        "https://example.com",            # ordinary public host
        "https://8.8.8.8/pipeline/run",   # public IP literal
        "ftp://localhost",                # not HTTP
        "",                               # nothing configured
    ],
)
def test_everything_else_is_refused(url):
    with pytest.raises(UnsafeTargetError):
        assert_safe_target(url)


def test_production_marker_beats_a_staging_shaped_name():
    """The deny-list runs first so 'staging' cannot launder a prod host."""
    with pytest.raises(UnsafeTargetError, match="production marker"):
        assert_safe_target("https://prod-staging.example.com")

    safe, reason = classify_host("staging.prod.internal")
    assert not safe and "production marker" in reason


def test_allow_list_is_exact_and_cannot_be_wildcarded():
    host = "redteam-box.example.com"
    assert assert_safe_target(f"https://{host}", frozenset({host}))

    # A neighbouring host is not covered by allow-listing its sibling.
    with pytest.raises(UnsafeTargetError):
        assert_safe_target("https://other.example.com", frozenset({host}))

    # And an allow-list entry still cannot override the deny-list.
    with pytest.raises(UnsafeTargetError, match="production marker"):
        assert_safe_target("https://prod.example.com", frozenset({"prod.example.com"}))


def test_private_ip_is_allowed_only_when_explicitly_listed():
    with pytest.raises(UnsafeTargetError):
        assert_safe_target("http://10.1.2.3:8000")
    assert assert_safe_target("http://10.1.2.3:8000", frozenset({"10.1.2.3"}))


def test_refusal_names_the_target_and_the_reason():
    with pytest.raises(UnsafeTargetError) as exc:
        assert_safe_target("https://api.production.example.com")
    message = str(exc.value)
    assert "api.production.example.com" in message
    assert "test/staging" in message
