from __future__ import annotations

from typing import NoReturn
from urllib.request import Request

import pytest

import market_support_crewai_agent.runtime.evidence.adapter_client as adapter_client_module
from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
    AdapterResolveClient,
)
from market_support_crewai_agent.settings import Settings


def test_adapter_timeout_is_translated_to_adapter_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(_request: Request, *, timeout: float) -> NoReturn:
        del timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(adapter_client_module, "urlopen", raise_timeout)
    client = AdapterResolveClient(
        Settings(adapter_base_url="http://adapter.invalid", adapter_timeout_seconds=0.01)
    )

    with pytest.raises(AdapterClientError, match="timed out"):
        _ = client.capabilities()
