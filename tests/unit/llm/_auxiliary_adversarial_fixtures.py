from __future__ import annotations

from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    ApprovedImageAssetCandidateViewV1,
    ApprovedKnowledgeCandidateViewV1,
    ApprovedKnowledgeSelectorInputV1,
    DocumentProductSelectorInputV1,
    ProductCandidateViewV1,
)


def adversarial_selector_inputs(
    adversarial: str,
    manifest_ref: ManifestRefV1,
) -> tuple[ApprovedKnowledgeSelectorInputV1, DocumentProductSelectorInputV1]:
    label = adversarial[:120]
    approved = ApprovedKnowledgeSelectorInputV1(
        user_query=adversarial,
        evidence_query=adversarial[:200],
        candidates=(
            ApprovedKnowledgeCandidateViewV1(
                entry_id="adversarial_entry",
                question=label,
                title=label,
                semantic_purpose=label,
                manifest_ref=manifest_ref,
                fact_type="document_context",
                image_assets=(
                    ApprovedImageAssetCandidateViewV1(
                        asset_id="adversarial_asset",
                        semantic_label=label,
                    ),
                ),
            ),
        ),
        max_entries=1,
        max_images=0,
        selected_manifest_ref=manifest_ref,
    )
    document = DocumentProductSelectorInputV1(
        user_query=adversarial,
        evidence_query=adversarial[:200],
        candidates=(
            ProductCandidateViewV1(
                id="document:adversarial",
                name=adversarial[:160],
                title=adversarial[:240],
                category=adversarial[:80],
                keywords=(adversarial[:160],),
                summary=adversarial[:1200],
            ),
        ),
        max_documents=1,
    )
    return approved, document
