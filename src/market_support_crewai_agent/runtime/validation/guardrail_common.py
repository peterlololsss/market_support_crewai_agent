from __future__ import annotations

import re
from typing import Final

_IMAGE_MARKER_RE: Final = re.compile(r"%%([\w\d_.-]+\.png)%%")


def image_marker_filenames(text: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for match in _IMAGE_MARKER_RE.finditer(text):
        filename = match.group(1)
        if filename in seen:
            continue
        seen.add(filename)
        output.append(filename)
    return output
