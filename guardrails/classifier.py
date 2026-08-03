"""Tier 2: a scored classifier for attacks that are paraphrased past the regexes.

Honest description: this is a logistic model over hand-engineered features whose
weights are priors, not fitted coefficients -- a greenfield Phase 1 has no
labelled corpus to fit against. The feature extraction and scoring head are real.

ponytail: hand-set weights, decent on the obvious attacks and blind to subtle
ones. Upgrade path: keep `features()`, fit WEIGHTS on a labelled set, or drop in
a trained sequence classifier behind the same `classify()` signature.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from agents.contracts import GuardrailFinding

_TOKEN = re.compile(r"[a-z0-9']+")

# Lexicons stand in for learned embeddings. Small on purpose -- they are priors.
_OVERRIDE_TERMS = frozenset(
    """ignore disregard forget override bypass circumvent supersede nullify void
    revoke suspend disable unlock unrestricted jailbreak jailbroken""".split()
)
_INSTRUCTION_TERMS = frozenset(
    """instruction instructions prompt prompts rule rules policy policies guideline
    guidelines constraint constraints restriction restrictions filter guardrail
    guardrails directive directives system""".split()
)
_PERSONA_TERMS = frozenset(
    """pretend roleplay simulate impersonate persona character mode alter act
    become behave""".split()
)
_SECRECY_TERMS = frozenset(
    """secretly covertly quietly silently hidden confidential nobody anyone
    without telling undetected unnoticed""".split()
)
_URGENCY_TERMS = frozenset(
    """urgent immediately now critical emergency must required mandatory authorized
    permission developer administrator admin owner override""".split()
)
# Deliberately mild: threat/harassment intent markers, not a slur list.
_TOXICITY_TERMS = frozenset(
    """hate kill destroy attack harm hurt threaten stupid idiot worthless useless
    pathetic disgusting revenge humiliate ruin punish""".split()
)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _density(tokens: list[str], lexicon: frozenset[str]) -> float:
    """Length-normalised hit rate, squashed so long benign text is not penalised."""
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in lexicon)
    return min(1.0, hits / math.sqrt(len(tokens)))


def _entropy(text: str) -> float:
    """Shannon entropy per char; encoded blobs sit far above prose (~4.0 vs ~2.9)."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _homoglyph_ratio(text: str) -> float:
    """Non-ASCII letters used to smuggle look-alike words past ASCII matchers."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    odd = sum(1 for c in letters if ord(c) > 127 and "LATIN" not in unicodedata.name(c, ""))
    return odd / len(letters)


def _imperative_ratio(text: str) -> float:
    """Fraction of clauses that open with a command verb -- attacks are bossy."""
    clauses = [c.strip() for c in re.split(r"[.!?\n;]", text) if c.strip()]
    if not clauses:
        return 0.0
    commanding = 0
    for clause in clauses:
        first = _tokens(clause)[:1]
        if first and first[0] in (_OVERRIDE_TERMS | _PERSONA_TERMS | {"you", "now", "instead"}):
            commanding += 1
    return commanding / len(clauses)


def features(text: str) -> dict[str, float]:
    tokens = _tokens(text)
    override = _density(tokens, _OVERRIDE_TERMS)
    instruction = _density(tokens, _INSTRUCTION_TERMS)
    return {
        "override": override,
        "instruction": instruction,
        # Co-occurrence is the strongest single signal: "ignore" near "rules".
        "override_x_instruction": override * instruction * 4.0,
        "persona": _density(tokens, _PERSONA_TERMS),
        "secrecy": _density(tokens, _SECRECY_TERMS),
        "urgency": _density(tokens, _URGENCY_TERMS),
        "toxic": _density(tokens, _TOXICITY_TERMS),
        "imperative": _imperative_ratio(text),
        "homoglyph": _homoglyph_ratio(text),
        "entropy": max(0.0, (_entropy(text) - 3.6) / 2.0),
    }


_INJECTION_WEIGHTS = {
    "override": 3.0,
    "instruction": 1.4,
    "override_x_instruction": 4.5,
    "persona": 2.2,
    "secrecy": 2.0,
    "urgency": 1.2,
    "imperative": 1.5,
    "homoglyph": 3.0,
    "entropy": 1.8,
}
_INJECTION_BIAS = -3.2

_TOXICITY_WEIGHTS = {"toxic": 6.0, "imperative": 0.6}
_TOXICITY_BIAS = -3.4


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _score(feats: dict[str, float], weights: dict[str, float], bias: float) -> float:
    return _sigmoid(bias + sum(w * feats.get(k, 0.0) for k, w in weights.items()))


def scores(text: str) -> dict[str, float]:
    feats = features(text)
    return {
        "injection": _score(feats, _INJECTION_WEIGHTS, _INJECTION_BIAS),
        "toxicity": _score(feats, _TOXICITY_WEIGHTS, _TOXICITY_BIAS),
    }


# Below this, the signal is indistinguishable from ordinary assertive prose.
REPORT_FLOOR = 0.35


def classify(text: str) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    for category, score in scores(text).items():
        if score >= REPORT_FLOOR:
            findings.append(
                GuardrailFinding(
                    tier="classifier",
                    category=f"{category}_semantic",
                    detail=f"heuristic {category} score {score:.2f}",
                    severity=round(score, 3),
                )
            )
    return findings
