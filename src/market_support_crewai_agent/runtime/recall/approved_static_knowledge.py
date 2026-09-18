from __future__ import annotations

from market_support_crewai_agent.runtime.recall.approved_static_catalog import (
    APPROVED_IMAGE_ASSETS,
    APPROVED_KNOWLEDGE,
    APPROVED_STATIC_MANIFEST_REF,
    ApprovedImageAsset,
    ApprovedKnowledgeEntry,
    approved_image_asset_by_marker,
    approved_image_markers,
    entry_image_assets,
)
from market_support_crewai_agent.runtime.recall.approved_static_selector import (
    ApprovedImageAssetCandidate,
    ApprovedKnowledgeCandidate,
    ApprovedKnowledgeSelection,
    ApprovedKnowledgeSelector,
    DirectApprovedKnowledgeSelector,
    NoopApprovedKnowledgeSelector,
)

__all__ = [
    "APPROVED_IMAGE_ASSETS",
    "APPROVED_KNOWLEDGE",
    "APPROVED_STATIC_MANIFEST_REF",
    "ApprovedImageAsset",
    "ApprovedImageAssetCandidate",
    "ApprovedKnowledgeCandidate",
    "ApprovedKnowledgeEntry",
    "ApprovedKnowledgeSelection",
    "ApprovedKnowledgeSelector",
    "DirectApprovedKnowledgeSelector",
    "NoopApprovedKnowledgeSelector",
    "_validate_selection",
    "approved_image_asset_by_marker",
    "approved_image_markers",
    "approved_knowledge_manifest",
]

_APPROVED_KNOWLEDGE_BY_ID = {entry.entry_id: entry for entry in APPROVED_KNOWLEDGE}
_APPROVED_IMAGE_ASSETS_BY_ID = {
    asset.asset_id: asset for asset in APPROVED_IMAGE_ASSETS
}


def approved_knowledge_manifest() -> tuple[ApprovedKnowledgeCandidate, ...]:
    return tuple(
        ApprovedKnowledgeCandidate(
            entry_id=entry.entry_id,
            manifest_ref=entry.manifest_ref,
            title=entry.title,
            semantic_purpose=entry.semantic_purpose,
            user_request_examples=entry.user_request_examples,
            image_assets=tuple(
                ApprovedImageAssetCandidate(
                    asset_id=asset.asset_id,
                    title=asset.title,
                    semantic_purpose=asset.semantic_purpose,
                    usage_notes=asset.usage_notes,
                )
                for asset in entry_image_assets(entry)
            ),
        )
        for entry in APPROVED_KNOWLEDGE
    )


def _validate_selection(
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

    selected_entry_ids = _valid_unique_entry_ids(selection.selected_entry_ids)
    selected_entry_ids = selected_entry_ids[:max_entries]
    if not selected_entry_ids:
        return ApprovedKnowledgeSelection(
            confidence="none",
            rationale=selection.rationale,
        )

    selected_entry_asset_ids = {
        asset_id
        for entry_id in selected_entry_ids
        for asset_id in _APPROVED_KNOWLEDGE_BY_ID[entry_id].image_asset_ids
    }
    selected_image_asset_ids = _valid_unique_asset_ids(
        selection.selected_image_asset_ids,
        allowed_asset_ids=selected_entry_asset_ids,
    )[:max_images]
    return ApprovedKnowledgeSelection(
        selected_entry_ids=tuple(selected_entry_ids),
        selected_image_asset_ids=tuple(selected_image_asset_ids),
        confidence=selection.confidence,
        rationale=selection.rationale,
    )


def _valid_unique_entry_ids(entry_ids: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for entry_id in entry_ids:
        if entry_id not in _APPROVED_KNOWLEDGE_BY_ID or entry_id in seen:
            continue
        seen.add(entry_id)
        output.append(entry_id)
    return output


def _valid_unique_asset_ids(
    asset_ids: tuple[str, ...],
    *,
    allowed_asset_ids: set[str],
) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for asset_id in asset_ids:
        if asset_id not in allowed_asset_ids or asset_id in seen:
            continue
        if asset_id not in _APPROVED_IMAGE_ASSETS_BY_ID:
            continue
        seen.add(asset_id)
        output.append(asset_id)
    return output
