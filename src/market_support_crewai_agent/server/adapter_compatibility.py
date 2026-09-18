from __future__ import annotations

from functools import lru_cache

from market_support_crewai_agent.runtime.integrations.adapter.client import (
    AdapterResolveClient,
)
from market_support_crewai_agent.settings import get_settings


def new_compatibility_client() -> AdapterResolveClient:
    return AdapterResolveClient(get_settings())


@lru_cache(maxsize=1)
def compatibility_client() -> AdapterResolveClient:
    return new_compatibility_client()


def clear_compatibility_client_cache_for_testing() -> None:
    compatibility_client.cache_clear()
