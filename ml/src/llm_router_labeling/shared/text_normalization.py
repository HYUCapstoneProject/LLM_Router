from __future__ import annotations

import re
import unicodedata


_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def normalize_for_duplicate_detection(text: str) -> str:
    """Return a stable word-only representation for exact/near deduplication."""
    tokens = _WORD_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())
    return " ".join(tokens)
