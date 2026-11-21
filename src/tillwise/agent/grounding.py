"""Number grounding: every sizeable number in an answer must trace back to a tool result.

Numbers are parsed with their suffixes (£9.14M, 1.2k, 3 million). A number is grounded when it is
within 1% of a number in the tool outputs, or of the sum or difference of two of them (so "total"
and "change" statements pass). Small numbers (< 100), percentages and years are not checked: they
are usually ranks, horizons, dates or shares the model computed from grounded values.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any

_NUM = re.compile(r"(?<![\w.])[£$]?\s?(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s?"
                  r"(million|thousand|bn|billion|[kKmM](?![a-zA-Z]))?(\s?%)?")
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """Return (value, is_percent) for every number in the text."""
    out = []
    for m in _NUM.finditer(text):
        raw, suffix, pct = m.group(1), m.group(2), m.group(3)
        value = float(raw.replace(",", ""))
        if suffix:
            value *= _MULT[suffix.lower()]
        out.append((value, bool(pct)))
    return out


def tool_numbers(payloads: list[Any]) -> list[float]:
    nums: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, int | float):
            nums.append(float(x))
        elif isinstance(x, str):
            nums.extend(v for v, _ in extract_numbers(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v)

    for p in payloads:
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except ValueError:
                pass
        walk(p)
    return nums


def _close(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1.0)


def ungrounded(answer: str, payloads: list[Any]) -> list[float]:
    known = tool_numbers(payloads)
    big = [abs(k) for k in known if abs(k) >= 100]
    derived = [a + b for a, b in combinations(big[:200], 2)] + [abs(a - b) for a, b in combinations(big[:200], 2)]
    bad = []
    for value, is_pct in extract_numbers(answer):
        v = abs(value)
        if is_pct or v < 100 or (1900 <= v <= 2100 and float(v).is_integer()):
            continue
        if any(_close(v, abs(k)) for k in known) or any(_close(v, d) for d in derived):
            continue
        bad.append(value)
    return bad
