from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
    AdapterResolveClient,
)
from market_support_crewai_agent.runtime.domain.canonicalization import (
    CanonicalContext,
)
from market_support_crewai_agent.runtime.domain.capabilities import (
    adapter_resolve_types,
    capability_by_resolve_type,
    ordered_resolve_types,
    resolve_type_accepts_strategy,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveRequest,
    AdapterResolveResult,
    AdapterResolveType,
    ReplyRequest,
)
from market_support_crewai_agent.settings import Settings


@dataclass(frozen=True)
class AdapterPreflightItem:
    resolve_type: AdapterResolveType
    result: AdapterResolveResult | None = None
    error: str = ""

    @property
    def status(self) -> str:
        if self.result is not None:
            return self.result.status
        return "adapter_unavailable"


@dataclass(frozen=True)
class AdapterPreflightSnapshot:
    items: list[AdapterPreflightItem]

    @property
    def available(self) -> bool:
        return all(not item.error for item in self.items)

    @classmethod
    def empty(cls) -> AdapterPreflightSnapshot:
        return cls(items=[])


class AdapterPreflightService:
    """Request-scoped adapter preflight orchestration.

    The service asks the WeCom adapter for safe, sanitized resolve facts before
    the LLM proposes outbound actions.
    """

    def __init__(
        self,
        adapter_client: AdapterResolveClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.adapter_client = adapter_client or AdapterResolveClient(settings)

    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext | None = None,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_strategies: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        resolve_requests = _build_resolve_requests(
            request,
            canonical_context,
            resolve_types,
            resolve_strategies,
        )
        if not resolve_requests:
            return AdapterPreflightSnapshot.empty()
        try:
            await self.adapter_client.assert_ready_async()
            results = await self.adapter_client.resolve_many_async(resolve_requests)
        except AdapterClientError as exc:
            return AdapterPreflightSnapshot(
                items=[
                    AdapterPreflightItem(
                        resolve_type=resolve_request.resolve_type,
                        error=str(exc),
                    )
                    for resolve_request in resolve_requests
                ]
            )
        results_by_type = {result.resolve_type: result for result in results}
        return AdapterPreflightSnapshot(
            items=[
                AdapterPreflightItem(
                    resolve_type=resolve_request.resolve_type,
                    result=results_by_type.get(resolve_request.resolve_type),
                    error=(
                        ""
                        if resolve_request.resolve_type in results_by_type
                        else "adapter batch result missing"
                    ),
                )
                for resolve_request in resolve_requests
            ]
        )


def _build_resolve_requests(
    request: ReplyRequest,
    canonical_context: CanonicalContext | None = None,
    resolve_types: list[AdapterResolveType] | None = None,
    resolve_strategies: dict[AdapterResolveType, str] | None = None,
) -> list[AdapterResolveRequest]:
    del canonical_context
    common = {
        "dist_name": request.dist_channel_name,
    }
    requested_types = _normalize_resolve_types(resolve_types)
    resolve_requests: list[AdapterResolveRequest] = []
    for resolve_type in requested_types:
        if capability_by_resolve_type(resolve_type) is None:
            raise ValueError(f"Unknown adapter resolve type: {resolve_type}")
        resolve_requests.append(
            AdapterResolveRequest(
                resolve_type=resolve_type,
                strategy=(
                    _strategy_for_resolve_type(resolve_type, resolve_strategies)
                    if resolve_type_accepts_strategy(resolve_type)
                    else None
                ),
                **common,
            )
        )
    return resolve_requests


def _normalize_resolve_types(
    resolve_types: list[AdapterResolveType] | None = None,
) -> tuple[AdapterResolveType, ...]:
    if resolve_types is None:
        return tuple(ordered_resolve_types(tuple(adapter_resolve_types())))
    for resolve_type in resolve_types:
        if capability_by_resolve_type(resolve_type) is None:
            raise ValueError(f"Unknown adapter resolve type: {resolve_type}")
    return tuple(ordered_resolve_types(resolve_types))


def _strategy_for_resolve_type(
    resolve_type: AdapterResolveType,
    resolve_strategies: dict[AdapterResolveType, str] | None = None,
) -> str | None:
    if resolve_strategies is not None:
        strategy = resolve_strategies.get(resolve_type)
        if strategy:
            return strategy
    return None
