from __future__ import annotations

import os
from typing import Final

import pytest

from market_support_crewai_agent.runtime.integrations.adapter.client import (
    AdapterResolveClient,
)
from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
)
from market_support_crewai_agent.settings_model import Settings

RAW_SERVER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "receiver",
        "group_id",
        "room_id",
        "conversation_id",
        "actual_user_id",
        "user_id",
        "to_user_id",
        "from_user_id",
        "at_list",
        "mention_ids",
        "url",
        "link",
        "file_path",
        "path",
    }
)


def live_adapter_base_url() -> str:
    return os.getenv(
        "MARKET_AGENT_LIVE_ADAPTER_BASE_URL",
        os.getenv("MARKET_AGENT_ADAPTER_BASE_URL", "http://127.0.0.1:8011"),
    ).rstrip("/")


def live_adapter_client(
    base_url: str,
    api_key: str | None,
) -> AdapterResolveClient:
    return AdapterResolveClient(
        Settings(
            llm_api_key="test-key",
            adapter_base_url=base_url,
            adapter_api_key=api_key,
            adapter_timeout_seconds=float(
                os.getenv("MARKET_AGENT_LIVE_ADAPTER_TIMEOUT_SECONDS", "3")
            ),
        )
    )


def skip_if_adapter_is_not_running(
    base_url: str,
    api_key: str | None,
) -> None:
    try:
        capabilities = live_adapter_client(base_url, api_key).capabilities()
    except AdapterClientError:
        pytest.skip("external adapter release blocker: live adapter is not reachable")
    if capabilities.service != "xiaoyan-wecom-market-agent-adapter":
        pytest.skip("external adapter release blocker: unexpected live adapter service")
