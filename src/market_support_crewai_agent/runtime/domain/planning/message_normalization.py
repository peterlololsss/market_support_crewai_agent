from __future__ import annotations

import re
import unicodedata

_TRAILING_PUNCTUATION = " ,，.。!！?？~～;；"


def normalize_compact_message(message: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(message or ""))
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.strip(_TRAILING_PUNCTUATION)
