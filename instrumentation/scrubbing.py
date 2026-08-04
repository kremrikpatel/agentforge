"""Deep PII scrubbing for anything about to be persisted.

Detection is entirely Phase 1's: `guardrails.patterns` already knows what an
email, SSN, or Luhn-valid card looks like, and a second implementation would
only drift from the first. This module adds the part guardrails does not do --
walking a nested structure and scrubbing every string inside it.

Distinct from the inline request-time guardrails in intent, too. Those decide
whether a *request* may proceed. This decides what may be *written down*, and it
never blocks: it redacts and records what it found.

Known boundary, stated rather than hidden: only strings are scanned. Numeric
fields are left alone because scrubbing them would destroy latencies,
confidences and counts to catch a case that does not occur in these structures
-- Phase 1 contracts carry their free text as strings. If a future producer puts
PII in an int, this will not catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guardrails import patterns

# Deep enough for any contract we produce; a guard against pathological nesting
# rather than an expected limit.
MAX_DEPTH = 16


@dataclass
class ScrubReport:
    value: Any = None
    findings: int = 0
    categories: set[str] = field(default_factory=set)

    @property
    def clean(self) -> bool:
        return self.findings == 0

    @property
    def sorted_categories(self) -> list[str]:
        return sorted(self.categories)


def scrub_text(text: str) -> tuple[str, list[str], int]:
    """Redact PII in one string. Returns (scrubbed, categories, count)."""
    if not text:
        return text, [], 0

    matches = [m for m in patterns.scan(text) if patterns.is_pii(m.category)]
    if not matches:
        return text, [], 0

    return (
        patterns.redact(text, matches),
        sorted({m.category for m in matches}),
        len(matches),
    )


def scrub(value: Any, _depth: int = 0) -> ScrubReport:
    """Recursively scrub every string in a structure.

    Dictionary keys are scrubbed too -- an address used as a key is still an
    address, and a payload keyed by email address is a realistic shape.
    """
    report = ScrubReport(value=value)

    if _depth > MAX_DEPTH:
        # Do not silently keep unscanned data: past the limit, drop it.
        report.value = "[TRUNCATED:DEPTH]"
        return report

    if isinstance(value, str):
        cleaned, categories, count = scrub_text(value)
        report.value = cleaned
        report.findings = count
        report.categories.update(categories)
        return report

    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                new_key, key_categories, key_count = scrub_text(key)
                report.findings += key_count
                report.categories.update(key_categories)
            else:
                new_key = key
            child = scrub(item, _depth + 1)
            report.findings += child.findings
            report.categories.update(child.categories)
            out[new_key] = child.value
        report.value = out
        return report

    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            child = scrub(item, _depth + 1)
            report.findings += child.findings
            report.categories.update(child.categories)
            items.append(child.value)
        report.value = items if isinstance(value, list) else tuple(items)
        return report

    # int / float / bool / None / anything else: see the module docstring.
    return report


def scrub_model(model, /) -> tuple[dict[str, Any], ScrubReport]:
    """Scrub a Pydantic model by value, returning the cleaned dict.

    Returns a dict rather than a rebuilt model on purpose: rebuilding would
    re-validate and could raise on a redacted value that no longer matches a
    constrained field, which would turn a compliance step into an outage.
    """
    report = scrub(model.model_dump(mode="json"))
    return report.value, report


def assert_clean(value: Any, context: str = "payload") -> None:
    """Fail loudly if PII survives. Used at test and export boundaries.

    Not used on the write path -- there, scrubbing is applied rather than
    asserted, because dropping a trajectory would lose the audit trail we are
    trying to keep.
    """
    report = scrub(value)
    if not report.clean:
        raise AssertionError(
            f"{context} still contains PII after scrubbing: "
            f"{report.findings} finding(s) in {report.sorted_categories}"
        )
