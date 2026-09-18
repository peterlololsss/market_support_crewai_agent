from __future__ import annotations

from market_support_crewai_agent.runtime.policy.capabilities.runtime_projection import (
    CurrentRuntimeCapabilityProjection,
    capability_by_name,
)
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveType,
    MaterialType,
    OutboundActionType,
    ReadCapability,
)


def capability_by_resolve_type(
    resolve_type: AdapterResolveType | str,
) -> CurrentRuntimeCapabilityProjection | None:
    match resolve_type:
        case "material_pack":
            return capability_by_name("material_pack")
        case "weekly_report":
            return capability_by_name("weekly_report")
        case "monthly_report":
            return capability_by_name("monthly_report")
        case "sales_mention":
            return capability_by_name("sales_mention")
        case _:
            return None


def capability_by_action_type(
    action_type: OutboundActionType | str,
) -> CurrentRuntimeCapabilityProjection | None:
    match action_type:
        case "send_material_pack":
            return capability_by_name("material_pack")
        case "send_weekly_report":
            return capability_by_name("weekly_report")
        case "send_monthly_report":
            return capability_by_name("monthly_report")
        case _:
            return None


def capability_by_read_capability(
    read_capability: ReadCapability | str,
) -> CurrentRuntimeCapabilityProjection | None:
    match read_capability:
        case "resolve_material_pack":
            return capability_by_name("material_pack")
        case "resolve_weekly_report":
            return capability_by_name("weekly_report")
        case "resolve_monthly_report":
            return capability_by_name("monthly_report")
        case "resolve_sales_mention":
            return capability_by_name("sales_mention")
        case "query_internal_company_info":
            return capability_by_name("document_context")
        case _:
            return None


def resolvable_fact_type_for_resolve(
    resolve_type: AdapterResolveType | str,
) -> str | None:
    projection = capability_by_resolve_type(resolve_type)
    return projection.resolvable_fact_type if projection is not None else None


def resolve_type_for_action(
    action_type: OutboundActionType | str,
) -> AdapterResolveType | None:
    projection = capability_by_action_type(action_type)
    return projection.resolve_type if projection is not None else None


def action_type_for_resolve(
    resolve_type: AdapterResolveType | str,
) -> OutboundActionType | None:
    projection = capability_by_resolve_type(resolve_type)
    return projection.outbound_action_type if projection is not None else None


def read_capability_for_resolve(
    resolve_type: AdapterResolveType | str,
) -> ReadCapability | None:
    projection = capability_by_resolve_type(resolve_type)
    return projection.read_capability if projection is not None else None


def resolve_type_for_read_capability(
    read_capability: ReadCapability | str,
) -> AdapterResolveType | None:
    projection = capability_by_read_capability(read_capability)
    return projection.resolve_type if projection is not None else None


def material_action_type(material_type: MaterialType) -> OutboundActionType | None:
    if material_type != "material":
        return None
    projection = capability_by_name("material_pack")
    return projection.outbound_action_type if projection is not None else None


def ordered_resolve_types(
    resolve_types: list[AdapterResolveType]
    | set[AdapterResolveType]
    | tuple[AdapterResolveType, ...],
) -> list[AdapterResolveType]:
    requested = set(resolve_types)
    return [
        resolve_type
        for resolve_type in (
            "material_pack",
            "weekly_report",
            "monthly_report",
            "sales_mention",
        )
        if resolve_type in requested
    ]
