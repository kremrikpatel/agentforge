"""Pull a JSON object out of an LLM completion.

Models wrap JSON in prose, ```json fences, or both, however firmly you ask them
not to. One tolerant extractor beats sprinkling try/except at every call site.
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict | None:
    if not text:
        return None

    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    candidates.extend(m.group(1).strip() for m in _FENCE.finditer(text))

    # Last resort: the outermost balanced {...}, tracking string state so a brace
    # inside a JSON string value cannot end the scan early.
    start = text.find("{")
    if start != -1:
        depth, in_string, escaped = 0, False, False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
