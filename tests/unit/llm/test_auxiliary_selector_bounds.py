from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    ApprovedImageAssetCandidateViewV1,
    ApprovedKnowledgeCandidateViewV1,
    ApprovedKnowledgeSelectorInputV1,
    DocumentProductSelectorInputV1,
    ProductCandidateViewV1,
)


APPROVED_REF = ManifestRefV1(
    manifest_id="answer_internal_company_knowledge",
    manifest_version="2026-07-18.1",
)
OTHER_REF = ManifestRefV1(
    manifest_id="material_pack.send",
    manifest_version="2026-07-18.1",
)


def _approved_candidate() -> ApprovedKnowledgeCandidateViewV1:
    return ApprovedKnowledgeCandidateViewV1(
        entry_id="company_profile",
        question="公司简介",
        title="公司简介",
        semantic_purpose="回答公司基本情况",
        manifest_ref=APPROVED_REF,
        fact_type="document_context",
        image_assets=(
            ApprovedImageAssetCandidateViewV1(
                asset_id="company_profile_chart",
                semantic_label="公司简介图",
            ),
        ),
    )


def _product_candidate() -> ProductCandidateViewV1:
    return ProductCandidateViewV1(
        id="document:alpha",
        name="Alpha",
        title="Alpha 产品",
        category="常见问答",
        keywords=("Alpha",),
        summary="Alpha 产品摘要",
    )


def _canonical_size(model: BaseModel) -> int:
    payload = model.model_dump(mode="json", exclude_none=False)
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evidence_query", "q" * 201),
        ("max_entries", 0),
        ("max_entries", 6),
        ("max_images", -1),
        ("max_images", 9),
    ),
)
def test_approved_selector_rejects_field_bound_overflow(
    field: str,
    value: str | int,
) -> None:
    # Given: a valid bounded approved-knowledge selector payload.
    payload = ApprovedKnowledgeSelectorInputV1(
        user_query="介绍公司",
        evidence_query="公司简介",
        candidates=(_approved_candidate(),),
        max_entries=1,
        max_images=1,
        selected_manifest_ref=None,
    ).model_dump(mode="json")
    payload[field] = value

    # When/Then: one value outside its declared range fails closed.
    with pytest.raises(ValidationError):
        _ = ApprovedKnowledgeSelectorInputV1.model_validate(payload)


def test_approved_selector_enforces_candidate_and_direct_bounds() -> None:
    # Given: valid candidates at the list ceiling and one mismatched direct ref.
    candidate = _approved_candidate()
    candidates = tuple(
        candidate.model_copy(update={"entry_id": f"entry_{index}"})
        for index in range(20)
    )
    mismatched = candidate.model_copy(update={"manifest_ref": OTHER_REF})

    # When/Then: 20 candidates pass, while 21 and direct widening fail.
    _ = ApprovedKnowledgeSelectorInputV1(
        user_query="介绍公司",
        evidence_query="",
        candidates=candidates,
        max_entries=5,
        max_images=8,
        selected_manifest_ref=None,
    )
    with pytest.raises(ValidationError):
        _ = ApprovedKnowledgeSelectorInputV1(
            user_query="介绍公司",
            evidence_query="",
            candidates=(*candidates, candidate),
            max_entries=5,
            max_images=8,
            selected_manifest_ref=None,
        )
    with pytest.raises(ValidationError):
        _ = ApprovedKnowledgeSelectorInputV1(
            user_query="介绍公司",
            evidence_query="",
            candidates=(candidate, mismatched),
            max_entries=2,
            max_images=0,
            selected_manifest_ref=APPROVED_REF,
        )
    with pytest.raises(ValidationError):
        _ = ApprovedKnowledgeSelectorInputV1(
            user_query="介绍公司",
            evidence_query="",
            candidates=(candidate,),
            max_entries=1,
            max_images=1,
            selected_manifest_ref=APPROVED_REF,
        )


def test_approved_selector_enforces_exact_serialized_byte_ceiling() -> None:
    # Given: a small valid payload and its exact ASCII query capacity.
    seed = ApprovedKnowledgeSelectorInputV1(
        user_query="q",
        evidence_query="",
        candidates=(_approved_candidate(),),
        max_entries=1,
        max_images=0,
        selected_manifest_ref=None,
    )
    exact_query_length = 131_072 - _canonical_size(seed) + 1

    # When/Then: 128 KiB passes byte-for-byte and one byte more rejects.
    exact = seed.model_copy(update={"user_query": "q" * exact_query_length})
    exact = ApprovedKnowledgeSelectorInputV1.model_validate(exact.model_dump())
    assert _canonical_size(exact) == 131_072
    with pytest.raises(ValidationError):
        _ = ApprovedKnowledgeSelectorInputV1.model_validate(
            seed.model_copy(
                update={"user_query": "q" * (exact_query_length + 1)}
            ).model_dump()
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evidence_query", "q" * 201),
        ("max_documents", 0),
        ("max_documents", 51),
    ),
)
def test_document_selector_rejects_field_bound_overflow(
    field: str,
    value: str | int,
) -> None:
    # Given: a valid bounded document-product selector payload.
    payload = DocumentProductSelectorInputV1(
        user_query="介绍 Alpha",
        evidence_query="Alpha",
        candidates=(_product_candidate(),),
        max_documents=1,
    ).model_dump(mode="json")
    payload[field] = value

    # When/Then: one value outside its declared range fails closed.
    with pytest.raises(ValidationError):
        _ = DocumentProductSelectorInputV1.model_validate(payload)


def test_document_selector_enforces_candidate_and_byte_bounds() -> None:
    # Given: 100 unique candidates and a seed for exact byte sizing.
    candidate = _product_candidate()
    candidates = tuple(
        candidate.model_copy(update={"id": f"document:{index}"}) for index in range(100)
    )
    seed = DocumentProductSelectorInputV1(
        user_query="q",
        evidence_query="",
        candidates=(candidate,),
        max_documents=1,
    )
    exact_query_length = 262_144 - _canonical_size(seed) + 1

    # When/Then: candidate and canonical JSON ceilings reject only overflow.
    _ = DocumentProductSelectorInputV1(
        user_query="介绍产品",
        evidence_query="",
        candidates=candidates,
        max_documents=50,
    )
    with pytest.raises(ValidationError):
        _ = DocumentProductSelectorInputV1(
            user_query="介绍产品",
            evidence_query="",
            candidates=(*candidates, candidate),
            max_documents=50,
        )
    exact = seed.model_copy(update={"user_query": "q" * exact_query_length})
    exact = DocumentProductSelectorInputV1.model_validate(exact.model_dump())
    assert _canonical_size(exact) == 262_144
    with pytest.raises(ValidationError):
        _ = DocumentProductSelectorInputV1.model_validate(
            seed.model_copy(
                update={"user_query": "q" * (exact_query_length + 1)}
            ).model_dump()
        )
