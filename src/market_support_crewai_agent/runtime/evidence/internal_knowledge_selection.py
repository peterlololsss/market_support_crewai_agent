from __future__ import annotations

from pydantic import Field, ValidationError

from market_support_crewai_agent.runtime.planning.models import (
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.recall.approved_static_catalog import (
    APPROVED_IMAGE_ASSETS,
    APPROVED_KNOWLEDGE,
    ApprovedKnowledgeEntry,
)
from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    ApprovedKnowledgeSelection,
)
from market_support_crewai_agent.schemas.base import StrictModel


def knowledge_targets(
    plan: ExecutionPlanV2,
    policy: PolicyManifestV2,
) -> tuple[tuple[ExecutionPlanUnitV2, ManifestRefV1], ...]:
    targets: list[tuple[ExecutionPlanUnitV2, ManifestRefV1]] = []
    required_sources = {"document_mcp", "approved_static_knowledge"}
    for unit in plan.units:
        if unit.manifest_ref not in policy.eligible_capabilities:
            continue
        manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.manifest_ref.manifest_id)
        if (
            manifest is None
            or manifest.manifest_version != unit.manifest_ref.manifest_version
        ):
            continue
        contract = manifest.evidence_contract
        if (
            "document_context" in contract.allowed_fact_types
            and "document_context" in contract.allowed_artifact_types
            and required_sources <= set(contract.allowed_source_types)
        ):
            targets.append((unit, unit.manifest_ref))
    return tuple(targets)


def validate_selection_for_gateway(
    selection: ApprovedKnowledgeSelection,
    *,
    max_entries: int,
    max_images: int,
) -> ApprovedKnowledgeSelection:
    if selection.confidence == "none":
        return ApprovedKnowledgeSelection(
            confidence="none",
            rationale=selection.rationale,
        )

    selected_entry_ids = valid_unique_entry_ids(selection.selected_entry_ids)[
        :max_entries
    ]
    if not selected_entry_ids:
        return ApprovedKnowledgeSelection(
            confidence="none",
            rationale=selection.rationale,
        )

    selected_entry_asset_ids = {
        asset_id
        for entry_id in selected_entry_ids
        if (entry := approved_entry_by_id(entry_id)) is not None
        for asset_id in entry.image_asset_ids
    }
    selected_image_asset_ids = valid_unique_asset_ids(
        selection.selected_image_asset_ids,
        allowed_asset_ids=selected_entry_asset_ids,
    )[:max_images]
    return ApprovedKnowledgeSelection(
        selected_entry_ids=tuple(selected_entry_ids),
        selected_image_asset_ids=tuple(selected_image_asset_ids),
        confidence=selection.confidence,
        rationale=selection.rationale,
    )


def valid_unique_entry_ids(entry_ids: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for entry_id in entry_ids:
        if approved_entry_by_id(entry_id) is None or entry_id in seen:
            continue
        seen.add(entry_id)
        output.append(entry_id)
    return output


def valid_unique_asset_ids(
    asset_ids: tuple[str, ...],
    *,
    allowed_asset_ids: set[str],
) -> list[str]:
    valid_asset_ids = {asset.asset_id for asset in APPROVED_IMAGE_ASSETS}
    seen: set[str] = set()
    output: list[str] = []
    for asset_id in asset_ids:
        if asset_id not in allowed_asset_ids or asset_id in seen:
            continue
        if asset_id not in valid_asset_ids:
            continue
        seen.add(asset_id)
        output.append(asset_id)
    return output


def approved_entry_by_id(entry_id: str) -> ApprovedKnowledgeEntry | None:
    return next(
        (entry for entry in APPROVED_KNOWLEDGE if entry.entry_id == entry_id), None
    )


class StateKeyRefV1(StrictModel):
    state_key_ref: str = Field(pattern=r"^csk1:[0-9a-f]{64}$")


def validated_state_key_ref(state_key_ref: str) -> str | None:
    try:
        return StateKeyRefV1(state_key_ref=state_key_ref).state_key_ref
    except ValidationError:
        return None
