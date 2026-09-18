from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    del config
    os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"
