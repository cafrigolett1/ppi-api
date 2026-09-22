"""Token estimation for input size limits.

**This is an estimate, not an exact count.** An exact count needs the
provider's own tokeniser, which either means an extra API call per request.

The estimator is deliberately conservative: it rounds up, so it never lets
through input that a real tokeniser would judge larger..
"""

from __future__ import annotations

import math
import re
from typing import Protocol

# Runs of non-alphanumeric characters (punctuation, separators) are counted
# separately because tokenisers rarely merge them with adjacent words.
_WORD = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Offline estimate, no model download and no API call.

    Counts word-like runs, then charges extra for long words, which subword
    tokenisers split. Compared against real BPE counts this tends to run
    slightly high on identifier-dense text, which is the direction we want.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0

        pieces = _WORD.findall(text)
        if not pieces:
            return math.ceil(len(text) / self.chars_per_token)

        total = 0
        for piece in pieces:
            # A short word is one token; longer ones get split by BPE.
            total += max(1, math.ceil(len(piece) / self.chars_per_token))
        return total


def default_counter() -> TokenCounter:
    return HeuristicTokenCounter()
