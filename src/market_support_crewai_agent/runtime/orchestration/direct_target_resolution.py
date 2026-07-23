from __future__ import annotations

from typing import Literal, Protocol

from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
)
from market_support_crewai_agent.schemas import (
    OutboundTargetKind,
    OutboundTargetResolveResult,
)


class DirectTargetResolver(Protocol):
    async def resolve_outbound_target_async(
        self,
        target_kind: OutboundTargetKind,
        target_name: str,
    ) -> OutboundTargetResolveResult: ...


TargetResolutionStatus = Literal[
    "resolved",
    "ambiguous",
    "unavailable",
    "unresolved",
]


async def resolve_direct_target(
    resolver: DirectTargetResolver,
    target_kind: OutboundTargetKind | None,
    target_name: str,
) -> tuple[TargetResolutionStatus, OutboundTargetResolveResult | None]:
    candidate_kinds: tuple[OutboundTargetKind, OutboundTargetKind]
    if target_kind == "group":
        candidate_kinds = ("group", "channel")
    else:
        candidate_kinds = ("channel", "group")
    candidates: list[OutboundTargetResolveResult] = []
    recognized_channel: OutboundTargetResolveResult | None = None
    completed_lookups = 0
    for candidate_kind in candidate_kinds:
        try:
            result = await resolver.resolve_outbound_target_async(
                candidate_kind,
                target_name,
            )
        except AdapterClientError:
            continue
        completed_lookups += 1
        if result.status == "resolved":
            candidates.append(result)
            if target_kind is not None:
                break
        elif (
            recognized_channel is None
            and result.target_kind == "channel"
            and result.reason_code == "target_incomplete"
            and result.target_count > 0
        ):
            recognized_channel = result
    if len(candidates) == 2:
        return "ambiguous", None
    if candidates:
        return "resolved", candidates[0]
    if completed_lookups == 0:
        return "unavailable", None
    return "unresolved", recognized_channel
