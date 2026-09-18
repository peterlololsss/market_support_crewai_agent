from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.capabilities.prompt_projection import (
    CapabilityViewAuthorityV1,
    project_capability_view_set,
)


def _ref(manifest_id: str) -> ManifestRefV1:
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(manifest_id)
    assert manifest is not None
    return ManifestRefV1(
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.manifest_version,
    )


def _project(stage: str, manifest_id: str):
    ref = _ref(manifest_id)
    return project_capability_view_set(
        stage=stage,
        authority=CapabilityViewAuthorityV1(
            eligible_manifest_refs=(ref,),
            selected_manifest_refs=() if stage == "planner_intent" else (ref,),
        ),
    )


@pytest.mark.parametrize(
    "manifest_id",
    (
        "material_pack.send",
        "weekly_report.send",
        "monthly_report.send",
        "sales.handoff",
        "general.clarification",
        "general.abstention",
        "general.refusal",
        "general.smalltalk",
        "general.no_reply",
        "weekly_report.product_list",
        "monthly_report.product_list",
        "answer_internal_company_knowledge",
        "general.handoff",
    ),
)
def test_every_canonical_manifest_projects_required_allowed_and_forbidden_artifacts(
    manifest_id: str,
) -> None:
    # Given: each manifest in the sealed 13-capability registry.
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(manifest_id)
    assert manifest is not None
    contract = manifest.evidence_contract

    # When: each role view is built from that manifest's canonical contract.
    planner = _project("planner_intent", manifest_id).items[0]
    composer = _project("knowledge_composer", manifest_id).items[0]
    verifier = _project("alignment_verifier", manifest_id).items[0]

    # Then: each provider-neutral evidence field is preserved without source-provider fields.
    expected_required = contract.required_artifact_types
    expected_allowed = contract.allowed_artifact_types
    expected_forbidden = tuple(
        artifact
        for artifact in (
            "material_pack",
            "weekly_report",
            "monthly_report",
            "document_context",
            "adapter_context",
            "history",
            "user_upload",
            "unknown",
        )
        if artifact not in expected_allowed
    )
    for item in (planner, composer, verifier):
        assert item.required_artifacts == expected_required
        assert item.allowed_artifacts == expected_allowed
        assert item.forbidden_artifacts == expected_forbidden
        evidence = (
            item.evidence_contract if hasattr(item, "evidence_contract") else item
        )
        assert evidence.required_fact_types == contract.required_fact_types
        assert evidence.any_of_fact_types == contract.any_of_fact_types
        assert evidence.allowed_fact_types == contract.allowed_fact_types
        assert evidence.forbidden_fact_types == contract.forbidden_fact_types
        assert evidence.required_artifact_types == contract.required_artifact_types
        assert evidence.allowed_artifact_types == contract.allowed_artifact_types
        assert evidence.required_scope_match == contract.required_scope_match
        assert evidence.min_facts == contract.min_facts
        assert evidence.allow_history is contract.allow_history
        assert evidence.history_constraints == contract.history_constraints
        assert evidence.fallback_policy == contract.fallback_policy
        assert evidence.citation_required is contract.citation_required
        assert evidence.provenance_required is contract.provenance_required
        assert evidence.citation_requirements == contract.citation_requirements
        assert evidence.stale_data_policy == contract.stale_data_policy
        assert not hasattr(item, "allowed_source_types")
        assert not hasattr(item, "forbidden_source_types")


def test_internal_knowledge_role_view_never_exposes_provider_authority() -> None:
    # Given: the canonical internal-knowledge manifest, whose deterministic contract has two sources.
    view = _project("knowledge_composer", "answer_internal_company_knowledge")

    # When: the model-facing item is serialized.
    payload = view.items[0].model_dump(mode="json")

    # Then: only the unified requirement and admitted artifact survive projection.
    assert payload["internal_company_knowledge_required"] is True
    assert payload["required_artifacts"] == ["document_context"]
    assert payload["allowed_artifacts"] == ["document_context"]
    assert "allowed_source_types" not in payload
    assert "forbidden_source_types" not in payload
    assert "document_mcp" not in str(payload)
    assert "approved_static_knowledge" not in str(payload)
