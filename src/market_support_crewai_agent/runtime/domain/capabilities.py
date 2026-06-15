from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.schemas import (
    AdapterResolveType,
    MaterialType,
    ReadCapability,
    SideEffectActionType,
)

CapabilityName = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "sales_mention",
    "document_context",
]
ArtifactKind = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "knowledge_answer",
    "human_support",
    "refusal",
    "unclear",
    "smalltalk",
]
ResponseMode = Literal[
    "action",
    "clarification",
    "handoff",
    "refusal",
    "unable",
    "knowledge_answer",
    "smalltalk",
    "no_reply",
]
ResolvableBusinessStateField = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "sales_mention",
]


@dataclass(frozen=True)
class CapabilitySpec:
    name: CapabilityName
    artifact_kind: ArtifactKind
    read_capability: ReadCapability | None
    resolve_type: AdapterResolveType | None
    side_effect_action_type: SideEffectActionType | None
    resolvable_fact_type: str | None
    business_state_field: str | None
    requires_strategy_for_bank_material: bool = False
    is_report: bool = False
    prompt_label: str = ""


CAPABILITY_REGISTRY: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        name="material_pack",
        artifact_kind="material_pack",
        read_capability="resolve_material_pack",
        resolve_type="material_pack",
        side_effect_action_type="send_material_pack",
        resolvable_fact_type="material_pack_resolvable",
        business_state_field="material_pack",
        requires_strategy_for_bank_material=True,
        prompt_label="材料包",
    ),
    CapabilitySpec(
        name="weekly_report",
        artifact_kind="weekly_report",
        read_capability="resolve_weekly_report",
        resolve_type="weekly_report",
        side_effect_action_type="send_weekly_report",
        resolvable_fact_type="weekly_report_resolvable",
        business_state_field="weekly_report",
        is_report=True,
        prompt_label="周报",
    ),
    CapabilitySpec(
        name="monthly_report",
        artifact_kind="monthly_report",
        read_capability="resolve_monthly_report",
        resolve_type="monthly_report",
        side_effect_action_type="send_monthly_report",
        resolvable_fact_type="monthly_report_resolvable",
        business_state_field="monthly_report",
        is_report=True,
        prompt_label="月报",
    ),
    CapabilitySpec(
        name="sales_mention",
        artifact_kind="human_support",
        read_capability="resolve_sales_mention",
        resolve_type="sales_mention",
        side_effect_action_type=None,
        resolvable_fact_type="sales_mention_resolvable",
        business_state_field="sales_mention",
        prompt_label="销售/支持同事",
    ),
    CapabilitySpec(
        name="document_context",
        artifact_kind="knowledge_answer",
        read_capability="query_internal_company_info",
        resolve_type=None,
        side_effect_action_type=None,
        resolvable_fact_type="document_context",
        business_state_field=None,
        prompt_label="文档证据",
    ),
)

_RESOLVE_ORDER: tuple[AdapterResolveType, ...] = (
    "material_pack",
    "weekly_report",
    "monthly_report",
    "sales_mention",
)


def capability_by_name(name: CapabilityName | str) -> CapabilitySpec | None:
    for capability in CAPABILITY_REGISTRY:
        if capability.name == name:
            return capability
    return None


def capability_by_resolve_type(
    resolve_type: AdapterResolveType | str,
) -> CapabilitySpec | None:
    for capability in CAPABILITY_REGISTRY:
        if capability.resolve_type == resolve_type:
            return capability
    return None


def capability_by_action_type(
    action_type: SideEffectActionType | str,
) -> CapabilitySpec | None:
    for capability in CAPABILITY_REGISTRY:
        if capability.side_effect_action_type == action_type:
            return capability
    return None


def capability_by_read_capability(read_capability: ReadCapability | str) -> CapabilitySpec | None:
    for capability in CAPABILITY_REGISTRY:
        if capability.read_capability == read_capability:
            return capability
    return None


def capabilities_for_artifact(artifact_kind: ArtifactKind | str) -> tuple[CapabilitySpec, ...]:
    return tuple(
        capability
        for capability in CAPABILITY_REGISTRY
        if capability.artifact_kind == artifact_kind
    )


def read_capabilities() -> frozenset[ReadCapability]:
    return frozenset(
        capability.read_capability
        for capability in CAPABILITY_REGISTRY
        if capability.read_capability is not None
    )


def adapter_resolve_types() -> frozenset[AdapterResolveType]:
    return frozenset(_RESOLVE_ORDER)


def side_effect_action_types() -> frozenset[SideEffectActionType]:
    return frozenset(
        capability.side_effect_action_type
        for capability in CAPABILITY_REGISTRY
        if capability.side_effect_action_type is not None
    )


def read_capabilities_for_artifact(artifact_kind: ArtifactKind | str) -> frozenset[ReadCapability]:
    return frozenset(
        capability.read_capability
        for capability in capabilities_for_artifact(artifact_kind)
        if capability.read_capability is not None
    )


def resolvable_business_state_fields() -> tuple[ResolvableBusinessStateField, ...]:
    return tuple(
        capability.business_state_field
        for capability in CAPABILITY_REGISTRY
        if capability.business_state_field is not None
    )  # type: ignore[return-value]


def capability_by_business_state_field(
    field_name: ResolvableBusinessStateField | str,
) -> CapabilitySpec | None:
    for capability in CAPABILITY_REGISTRY:
        if capability.business_state_field == field_name:
            return capability
    return None


def resolvable_fact_type_for_resolve(
    resolve_type: AdapterResolveType | str,
) -> str | None:
    capability = capability_by_resolve_type(resolve_type)
    return capability.resolvable_fact_type if capability is not None else None


def resolve_type_for_action(
    action_type: SideEffectActionType | str,
) -> AdapterResolveType | None:
    capability = capability_by_action_type(action_type)
    return capability.resolve_type if capability is not None else None


def action_type_for_resolve(
    resolve_type: AdapterResolveType | str,
) -> SideEffectActionType | None:
    capability = capability_by_resolve_type(resolve_type)
    return capability.side_effect_action_type if capability is not None else None


def read_capability_for_resolve(
    resolve_type: AdapterResolveType | str,
) -> ReadCapability | None:
    capability = capability_by_resolve_type(resolve_type)
    return capability.read_capability if capability is not None else None


def resolve_type_accepts_strategy(resolve_type: AdapterResolveType | str) -> bool:
    capability = capability_by_resolve_type(resolve_type)
    if capability is None:
        return False
    return capability.is_report or capability.requires_strategy_for_bank_material


def resolve_type_for_read_capability(
    read_capability: ReadCapability | str,
) -> AdapterResolveType | None:
    capability = capability_by_read_capability(read_capability)
    return capability.resolve_type if capability is not None else None


def material_action_type(material_type: MaterialType) -> SideEffectActionType | None:
    if material_type != "material":
        return None
    capability = capability_by_name("material_pack")
    return capability.side_effect_action_type if capability is not None else None


def report_action_types() -> frozenset[SideEffectActionType]:
    return frozenset(
        capability.side_effect_action_type
        for capability in CAPABILITY_REGISTRY
        if capability.is_report and capability.side_effect_action_type is not None
    )


def ordered_resolve_types(
    resolve_types: list[AdapterResolveType] | set[AdapterResolveType] | tuple[AdapterResolveType, ...],
) -> list[AdapterResolveType]:
    requested = set(resolve_types)
    return [resolve_type for resolve_type in _RESOLVE_ORDER if resolve_type in requested]


def capability_prompt_dict() -> list[dict]:
    return [
        {
            "name": capability.name,
            "artifact_kind": capability.artifact_kind,
            "read_capability": capability.read_capability,
            "resolve_type": capability.resolve_type,
            "side_effect_action_type": capability.side_effect_action_type,
            "resolvable_fact_type": capability.resolvable_fact_type,
            "business_state_field": capability.business_state_field,
            "requires_strategy_for_bank_material": capability.requires_strategy_for_bank_material,
            "is_report": capability.is_report,
            "prompt_label": capability.prompt_label,
        }
        for capability in CAPABILITY_REGISTRY
    ]


def capability_registry_hash() -> str:
    payload = json.dumps(
        capability_prompt_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
