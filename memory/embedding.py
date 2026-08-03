"""Deterministic local embeddings via the hashing trick.

No network, no key, no extra dependency -- so the semantic cache and its tests
work offline. Similarity is lexical rather than truly semantic, which is what
"identical or near-identical request" actually needs.

ponytail: lexical only, so "car" and "automobile" look unrelated. Upgrade path:
swap `embed()` for a real embedding endpoint; the vector column and cosine math
below do not change.
"""

from __future__ import annotations

import hashlib
import math
import re

DEFAULT_DIM = 384
_TOKEN = re.compile(r"[a-z0-9]+")
# Drop-in stopwords: they add cosine mass without carrying meaning.
_STOP = frozenset(
    "a an the and or of to in for on with is are be was were it this that as at by from".split()
)


def _bucket(token: str, dim: int) -> tuple[int, float]:
    """Map a token to (index, sign). The sign halves collision bias."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def embed(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    vec = [0.0] * dim
    tokens = tokenize(text)

    for token in tokens:
        idx, sign = _bucket(token, dim)
        vec[idx] += sign
        # Character 4-grams keep typos and morphology near their base word.
        for i in range(len(token) - 3):
            gidx, gsign = _bucket(token[i : i + 4], dim)
            vec[gidx] += gsign * 0.3

    # Bigrams give the vector a little word-order sensitivity.
    for left, right in zip(tokens, tokens[1:]):
        idx, sign = _bucket(f"{left}_{right}", dim)
        vec[idx] += sign * 0.5

    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    """Both inputs come from embed(), i.e. already unit length."""
    if len(a) != len(b):
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def normalize_text(text: str) -> str:
    """Canonical form for exact-hit cache keys: casing/whitespace do not matter."""
    return " ".join(text.lower().split())


def cache_key(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
