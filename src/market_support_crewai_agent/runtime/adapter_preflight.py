from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.adapter_client import (
    AdapterClientError,
    AdapterResolveClient,
)
from market_support_crewai_agent.runtime.canonicalization import (
    CanonicalContext,
    canonicalize_request,
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
    the LLM proposes side-effect actions.
    """

    def __init__(
        self,
        adapter_client: AdapterResolveClient | None = None,
        settings: Settings | None = None,
        enabled: bool = True,
    ) -> None:
        self.adapter_client = adapter_client or AdapterResolveClient(settings)
        self.enabled = enabled

    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext | None = None,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_strategies: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        if not self.enabled:
            return AdapterPreflightSnapshot.empty()

        resolve_requests = _build_resolve_requests(
            request,
            canonical_context,
            resolve_types,
            resolve_strategies,
        )
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


class NoopAdapterPreflightService:
    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext | None = None,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_strategies: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, canonical_context, resolve_types, resolve_strategies
        return AdapterPreflightSnapshot.empty()


def _build_resolve_requests(
    request: ReplyRequest,
    canonical_context: CanonicalContext | None = None,
    resolve_types: list[AdapterResolveType] | None = None,
    resolve_strategies: dict[AdapterResolveType, str] | None = None,
) -> list[AdapterResolveRequest]:
    material_strategy = _strategy_for_resolve_type(
        "material_pack",
        request,
        canonical_context,
        resolve_strategies,
    )
    weekly_strategy = _strategy_for_resolve_type(
        "weekly_report",
        request,
        canonical_context,
        resolve_strategies,
    )
    monthly_strategy = _strategy_for_resolve_type(
        "monthly_report",
        request,
        canonical_context,
        resolve_strategies,
    )
    common = {
        "dist_name": request.dist_channel_name,
    }
    requested_types = _normalize_resolve_types(resolve_types)
    requests_by_type = {
        "material_pack": AdapterResolveRequest(
            resolve_type="material_pack",
            strategy=material_strategy,
            **common,
        ),
        "weekly_report": AdapterResolveRequest(
            resolve_type="weekly_report",
            strategy=weekly_strategy,
            **common,
        ),
        "monthly_report": AdapterResolveRequest(
            resolve_type="monthly_report",
            strategy=monthly_strategy,
            **common,
        ),
        "sales_mention": AdapterResolveRequest(resolve_type="sales_mention", **common),
    }
    return [
        requests_by_type[resolve_type]
        for resolve_type in requested_types
    ]


def _normalize_resolve_types(
    resolve_types: list[AdapterResolveType] | None = None,
) -> tuple[AdapterResolveType, ...]:
    default_order: tuple[AdapterResolveType, ...] = (
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    )
    if resolve_types is None:
        return default_order
    requested = set(resolve_types)
    return tuple(resolve_type for resolve_type in default_order if resolve_type in requested)


def _strategy_for_resolve_type(
    resolve_type: AdapterResolveType,
    request: ReplyRequest,
    canonical_context: CanonicalContext | None = None,
    resolve_strategies: dict[AdapterResolveType, str] | None = None,
) -> str | None:
    if resolve_strategies is not None:
        strategy = resolve_strategies.get(resolve_type)
        if strategy:
            return strategy
    return _material_strategy_for_request(request, canonical_context)


def _material_strategy_for_request(
    request: ReplyRequest,
    canonical_context: CanonicalContext | None = None,
) -> str | None:
    canonical_context = canonical_context or canonicalize_request(request)
    if canonical_context.strategy_status == "resolved":
        return canonical_context.selected_strategy
    return None
