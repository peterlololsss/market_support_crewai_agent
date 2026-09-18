from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    ApprovedStaticKnowledgeGatewayAdapter,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayStaticContextV1,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    APPROVED_KNOWLEDGE,
    ApprovedKnowledgeEntry,
    ApprovedKnowledgeSelection,
    approved_knowledge_manifest,
)
from market_support_crewai_agent.runtime.recall.approved_static_selector import (
    ApprovedKnowledgeCandidate,
)
from tests.helpers.reply_contract_requests import make_v2_envelope


def _knowledge_ref() -> ManifestRefV1:
    return ManifestRefV1(
        manifest_id="answer_internal_company_knowledge",
        manifest_version="2026-07-18.1",
    )


class _ExactSelector:
    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection:
        del user_message, evidence_query, catalog_manifest, max_entries, max_images
        return ApprovedKnowledgeSelection(
            selected_entry_ids=("company_shareholders",),
            selected_image_asset_ids=("company_shareholders_chart",),
            confidence="high",
        )


def test_every_catalog_entry_has_the_exact_registered_knowledge_manifest_ref() -> None:
    # Given: the complete reviewed approved-static catalog and 13-manifest registry.
    expected_ref = _knowledge_ref()

    # When: catalog and selector-facing rows are projected.
    catalog_refs = tuple(entry.manifest_ref for entry in APPROVED_KNOWLEDGE)
    candidate_refs = tuple(
        candidate.manifest_ref for candidate in approved_knowledge_manifest()
    )

    # Then: every row is explicitly bound to the one registered knowledge manifest.
    assert catalog_refs == (expected_ref,) * len(APPROVED_KNOWLEDGE)
    assert candidate_refs == catalog_refs
    assert len(CAPABILITY_MANIFEST_REGISTRY.list()) == 13
    assert (
        CAPABILITY_MANIFEST_REGISTRY.get(expected_ref.manifest_id).manifest_version
        == expected_ref.manifest_version
    )


@pytest.mark.parametrize(
    "manifest_ref",
    (
        None,
        {
            "manifest_id": "general.smalltalk",
            "manifest_version": "2026-07-18.1",
        },
        {
            "manifest_id": "answer_internal_company_knowledge",
            "manifest_version": "2026-07-15.1",
        },
    ),
)
def test_catalog_entry_rejects_missing_wrong_or_invalid_manifest_ref(
    manifest_ref: dict[str, str] | None,
) -> None:
    # Given: a valid catalog row whose authority binding is removed or replaced.
    payload = APPROVED_KNOWLEDGE[0].model_dump(mode="json", exclude_none=False)
    if manifest_ref is None:
        del payload["manifest_ref"]
    else:
        payload["manifest_ref"] = manifest_ref

    # When/Then: the catalog boundary rejects it instead of inferring a topic ref.
    with pytest.raises(ValidationError):
        _ = ApprovedKnowledgeEntry.model_validate(payload)


@pytest.mark.anyio
async def test_valid_static_selection_carries_manifest_and_media_as_separate_bindings() -> (
    None
):
    # Given: the real adapter with one exact catalog entry and registered asset selected.
    adapter = ApprovedStaticKnowledgeGatewayAdapter(_ExactSelector())
    request = make_v2_envelope("股权结构是什么？").request

    # When: the post-plan static selector returns its bounded context.
    contexts = await adapter.collect(request=request, evidence_query="股权结构")

    # Then: text authority and selected media remain explicit, separate fields.
    assert len(contexts) == 1
    assert contexts[0].entry_id == "company_shareholders"
    assert contexts[0].manifest_ref == _knowledge_ref()
    assert contexts[0].selected_asset_ids == ("company_shareholders_chart",)
    assert "fact_type" not in GatewayStaticContextV1.model_fields
    assert "topic" not in GatewayStaticContextV1.model_fields
