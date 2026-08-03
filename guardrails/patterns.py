"""Tier 1: deterministic pattern matching for PII and known injection shapes.

Fast, explainable, zero-dependency. Catches the concrete stuff; the classifier
and LLM tiers handle the paraphrased attacks regex cannot see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.contracts import GuardrailFinding


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    category: str
    severity: float
    text: str


@dataclass(frozen=True)
class Rule:
    category: str
    severity: float
    pattern: re.Pattern[str]
    redact: bool
    detail: str


def _c(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


# --- PII -------------------------------------------------------------------
PII_RULES: tuple[Rule, ...] = (
    Rule(
        "email",
        0.5,
        _c(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
        True,
        "email address",
    ),
    Rule(
        "ssn",
        0.9,
        _c(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        True,
        "US social security number",
    ),
    Rule(
        "phone",
        0.45,
        _c(r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
        True,
        "phone number",
    ),
    Rule(
        "credit_card",
        0.95,
        _c(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
        True,
        "payment card number",
    ),
    Rule(
        "ip_address",
        0.3,
        _c(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        True,
        "IP address",
    ),
    Rule(
        "api_key",
        0.95,
        _c(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}"
            r"|AKIA[0-9A-Z]{16}"
            r"|gh[pousr]_[A-Za-z0-9]{20,}"
            r"|AIza[0-9A-Za-z_-]{30,}"
            r"|xox[baprs]-[A-Za-z0-9-]{10,})\b"
        ),
        True,
        "credential or API key",
    ),
)

# --- Prompt injection / jailbreak ------------------------------------------
INJECTION_RULES: tuple[Rule, ...] = (
    Rule(
        "instruction_override",
        0.9,
        _c(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|earlier|above|all|any|your)\b[^.\n]{0,25}"
            r"\b(?:instruction|prompt|rule|direction|guideline|constraint)s?\b"
        ),
        False,
        "attempts to override prior instructions",
    ),
    Rule(
        "system_prompt_exfil",
        0.85,
        _c(
            r"\b(?:reveal|show|print|repeat|output|disclose|dump|leak)\b[^.\n]{0,30}"
            r"\b(?:system prompt|initial prompt|your instructions|prompt above"
            r"|hidden (?:rules|prompt)|verbatim)\b"
        ),
        False,
        "attempts to exfiltrate the system prompt",
    ),
    Rule(
        "role_hijack",
        0.8,
        _c(
            r"\b(?:you are now|from now on you|act as|pretend to be|roleplay as|"
            r"simulate being)\b[^.\n]{0,40}"
            r"\b(?:DAN|developer mode|unrestricted|jailbroken|no restrictions|"
            r"without (?:any )?(?:filter|limit|guardrail|restriction))\b"
        ),
        False,
        "attempts to reassign the model's role",
    ),
    Rule(
        "chat_template_injection",
        0.9,
        _c(r"(?:<\|im_(?:start|end)\|>|\[/?INST\]|<<SYS>>|###\s*(?:system|assistant)\s*:)"),
        False,
        "raw chat-template control tokens in user content",
    ),
    Rule(
        "policy_pressure",
        0.6,
        _c(
            r"\b(?:this is (?:a )?(?:test|drill)|for (?:educational|research) purposes only|"
            r"i am (?:the|your) (?:developer|admin|administrator|owner)|"
            r"you have (?:my )?permission|i authorize you)\b"
        ),
        False,
        "social-engineering pressure to relax policy",
    ),
    Rule(
        "encoded_payload",
        0.5,
        _c(r"\b(?:base64|rot13|hex)\s*(?:decode|encoded?)\b|(?:[A-Za-z0-9+/]{60,}={0,2})"),
        False,
        "encoded payload that may hide instructions",
    ),
)

ALL_RULES: tuple[Rule, ...] = PII_RULES + INJECTION_RULES


def _luhn_ok(digits: str) -> bool:
    """Reject the ocean of 16-digit numbers that are not payment cards."""
    nums = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total, parity = 0, len(nums) % 2
    for idx, digit in enumerate(nums):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def scan(text: str) -> list[Match]:
    matches: list[Match] = []
    for rule in ALL_RULES:
        for m in rule.pattern.finditer(text):
            span = m.group(0)
            if rule.category == "credit_card" and not _luhn_ok(span):
                continue
            matches.append(Match(m.start(), m.end(), rule.category, rule.severity, span))
    return sorted(matches, key=lambda m: m.start)


_REDACTABLE = {r.category for r in ALL_RULES if r.redact}
_DETAILS = {r.category: r.detail for r in ALL_RULES}


def redact(text: str, matches: list[Match]) -> str:
    """Replace PII spans in place. Walk backwards so offsets stay valid."""
    out = text
    for m in sorted(matches, key=lambda m: m.start, reverse=True):
        if m.category in _REDACTABLE:
            out = f"{out[: m.start]}[REDACTED:{m.category.upper()}]{out[m.end :]}"
    return out


def to_findings(matches: list[Match]) -> list[GuardrailFinding]:
    return [
        GuardrailFinding(
            tier="regex",
            category=m.category,
            detail=_DETAILS.get(m.category, m.category),
            severity=m.severity,
            # Never echo the raw secret back into a report or log line.
            span=(m.text[:4] + "..." if m.category in _REDACTABLE else m.text[:120]),
        )
        for m in matches
    ]


def is_pii(category: str) -> bool:
    return category in _REDACTABLE
