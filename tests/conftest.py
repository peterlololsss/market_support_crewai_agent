from __future__ import annotations

from pathlib import Path

import pytest


_CATEGORY_MARKERS = {
    "unit": pytest.mark.unit,
    "integration": pytest.mark.integration,
    "contract": pytest.mark.contract,
    "live": pytest.mark.live,
}


def pytest_collection_modifyitems(config, items):
    del config
    tests_root = Path(__file__).resolve().parent
    for item in items:
        try:
            relative = Path(str(item.fspath)).resolve().relative_to(tests_root)
        except ValueError:
            continue
        category = relative.parts[0] if relative.parts else ""
        marker = _CATEGORY_MARKERS.get(category)
        if marker is not None:
            item.add_marker(marker)
