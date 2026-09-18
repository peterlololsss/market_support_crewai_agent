from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    ArtifactKind,
    CapabilityManifest,
    CapabilityManifestIdV2,
    CapabilityName,
    ResolvableBusinessStateField,
)
from market_support_crewai_agent.runtime.policy.capabilities.queries import (
    capability_manifest_by_id,
    capability_manifests,
)
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveType,
    OutboundActionType,
    ReadCapability,
)


@dataclass(frozen=True, slots=True)
class CurrentRuntimeCapabilityProjection:
    manifest: CapabilityManifest
    name: CapabilityName
    artifact_kind: ArtifactKind
    read_capability: ReadCapability | None
    resolve_type: AdapterResolveType | None
    outbound_action_type: OutboundActionType | None
    resolvable_fact_type: str | None
    business_state_field: ResolvableBusinessStateField | None
    supports_material_pack_option: bool = False
    is_report: bool = False


def capability_by_name(
    name: CapabilityName | str,
) -> CurrentRuntimeCapabilityProjection | None:
    match name:
        case "material_pack":
            return _project(
                "material_pack.send",
                "material_pack",
                "material_pack",
                "resolve_material_pack",
                "material_pack",
                "send_material_pack",
                "material_pack_resolvable",
                "material_pack",
                supports_material_pack_option=True,
            )
        case "weekly_report":
            return _project(
                "weekly_report.send",
                "weekly_report",
                "weekly_report",
                "resolve_weekly_report",
                "weekly_report",
                "send_weekly_report",
                "weekly_report_resolvable",
                "weekly_report",
                is_report=True,
            )
        case "monthly_report":
            return _project(
                "monthly_report.send",
                "monthly_report",
                "monthly_report",
                "resolve_monthly_report",
                "monthly_report",
                "send_monthly_report",
                "monthly_report_resolvable",
                "monthly_report",
                is_report=True,
            )
        case "sales_mention":
            return _project(
                "sales.handoff",
                "sales_mention",
                "human_support",
                "resolve_sales_mention",
                "sales_mention",
                None,
                "sales_mention_resolvable",
                "sales_mention",
            )
        case "document_context":
            return _project(
                "answer_internal_company_knowledge",
                "document_context",
                "knowledge_answer",
                "query_internal_company_info",
                None,
                None,
                "document_context",
                None,
            )
        case _:
            return None


def capabilities_for_artifact(
    artifact_kind: ArtifactKind | str,
) -> tuple[CurrentRuntimeCapabilityProjection, ...]:
    return tuple(
        projection
        for projection in current_runtime_projections()
        if projection.artifact_kind == artifact_kind
    )


def read_capabilities() -> frozenset[ReadCapability]:
    return frozenset(
        projection.read_capability
        for projection in current_runtime_projections()
        if projection.read_capability is not None
    )


def adapter_resolve_types() -> frozenset[AdapterResolveType]:
    return frozenset(
        projection.resolve_type
        for projection in current_runtime_projections()
        if projection.resolve_type is not None
    )


def outbound_action_types() -> frozenset[OutboundActionType]:
    return frozenset(
        projection.outbound_action_type
        for projection in current_runtime_projections()
        if projection.outbound_action_type is not None
    )


def read_capabilities_for_artifact(
    artifact_kind: ArtifactKind | str,
) -> frozenset[ReadCapability]:
    return frozenset(
        projection.read_capability
        for projection in capabilities_for_artifact(artifact_kind)
        if projection.read_capability is not None
    )


def resolvable_business_state_fields() -> tuple[ResolvableBusinessStateField, ...]:
    return tuple(
        projection.business_state_field
        for projection in current_runtime_projections()
        if projection.business_state_field is not None
    )


def capability_by_business_state_field(
    field_name: ResolvableBusinessStateField | str,
) -> CurrentRuntimeCapabilityProjection | None:
    for projection in current_runtime_projections():
        if projection.business_state_field == field_name:
            return projection
    return None


def report_action_types() -> frozenset[OutboundActionType]:
    return frozenset(
        projection.outbound_action_type
        for projection in current_runtime_projections()
        if projection.is_report and projection.outbound_action_type is not None
    )


def capability_prompt_dict() -> list[dict[str, str | bool | None]]:
    return [
        {
            "manifest_id": projection.manifest.manifest_id,
            "manifest_version": projection.manifest.manifest_version,
            "runtime_capability": projection.name,
            "artifact_kind": projection.artifact_kind,
            "read_capability": projection.read_capability,
            "resolve_type": projection.resolve_type,
            "outbound_action_type": projection.outbound_action_type,
            "resolvable_fact_type": projection.resolvable_fact_type,
            "business_state_field": projection.business_state_field,
            "is_report": projection.is_report,
        }
        for projection in current_runtime_projections()
    ]


def capability_manifest_prompt_dict() -> list[dict[str, object]]:
    return [manifest.model_dump(mode="json") for manifest in capability_manifests()]


def capability_registry_hash() -> str:
    payload = json.dumps(
        capability_manifest_prompt_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_runtime_projections() -> tuple[CurrentRuntimeCapabilityProjection, ...]:
    return tuple(
        projection
        for name in (
            "material_pack",
            "weekly_report",
            "monthly_report",
            "sales_mention",
            "document_context",
        )
        if (projection := capability_by_name(name)) is not None
    )


def _project(
    manifest_id: CapabilityManifestIdV2,
    name: CapabilityName,
    artifact_kind: ArtifactKind,
    read_capability: ReadCapability | None,
    resolve_type: AdapterResolveType | None,
    outbound_action_type: OutboundActionType | None,
    resolvable_fact_type: str | None,
    business_state_field: ResolvableBusinessStateField | None,
    *,
    supports_material_pack_option: bool = False,
    is_report: bool = False,
) -> CurrentRuntimeCapabilityProjection:
    manifest = capability_manifest_by_id(manifest_id)
    if manifest is None:
        raise KeyError(manifest_id)
    return CurrentRuntimeCapabilityProjection(
        manifest=manifest,
        name=name,
        artifact_kind=artifact_kind,
        read_capability=read_capability,
        resolve_type=resolve_type,
        outbound_action_type=outbound_action_type,
        resolvable_fact_type=resolvable_fact_type,
        business_state_field=business_state_field,
        supports_material_pack_option=supports_material_pack_option,
        is_report=is_report,
    )
