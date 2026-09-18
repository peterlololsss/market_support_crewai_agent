from __future__ import annotations

import hashlib
import json

import pytest

from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.capabilities.prompt_projection import (
    CapabilityViewAuthorityV1,
    CapabilityViewProjectionError,
    project_capability_view_set,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    ComposerCapabilityViewSetV1,
    PlannerCapabilityViewSetV1,
    VerifierCapabilityViewSetV1,
    capability_view_hash,
)


CANONICAL_MANIFEST_IDS = tuple(
    manifest.manifest_id for manifest in CAPABILITY_MANIFEST_REGISTRY.list()
)


def _ref(manifest_id: str) -> ManifestRefV1:
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(manifest_id)
    assert manifest is not None
    return ManifestRefV1(
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.manifest_version,
    )


def _authority(
    *,
    eligible: tuple[ManifestRefV1, ...] = tuple(
        _ref(item) for item in CANONICAL_MANIFEST_IDS
    ),
    selected: tuple[ManifestRefV1, ...] = (),
) -> CapabilityViewAuthorityV1:
    return CapabilityViewAuthorityV1(
        eligible_manifest_refs=eligible,
        selected_manifest_refs=selected,
    )


def test_planner_view_is_ordered_and_contains_every_canonical_manifest() -> None:
    # Given: the complete sealed eligibility set in registry order.
    authority = _authority()

    # When: the planner view is projected from that authority.
    view = project_capability_view_set(stage="planner_intent", authority=authority)

    # Then: refs/items preserve the positional taxonomy and carry a stable view hash.
    assert isinstance(view, PlannerCapabilityViewSetV1)
    assert (
        tuple(ref.manifest_id for ref in view.manifest_refs) == CANONICAL_MANIFEST_IDS
    )
    assert (
        tuple(item.manifest_ref.manifest_id for item in view.items)
        == CANONICAL_MANIFEST_IDS
    )
    assert view.visible_items_hash.startswith("cvh1:")
    assert len(view.visible_items_hash) == 69
    assert all(
        item.contract_version == "planner-capability-view.v1" for item in view.items
    )


def test_view_hash_changes_when_a_visible_item_changes() -> None:
    # Given: a fully projected planner view and its canonical visible-item hash.
    view = project_capability_view_set(stage="planner_intent", authority=_authority())
    payload = view.model_dump(mode="json")
    payload["items"][0]["selection_contract"]["summary"] += " changed"

    # When/Then: re-parsing the changed visible content rejects the stale hash.
    with pytest.raises(ValueError, match="view_hash_mismatch"):
        PlannerCapabilityViewSetV1.model_validate(payload)


def test_view_hash_uses_capability_view_set_domain_prefix() -> None:
    # Given: one planner view whose canonical serialization is independently available.
    view = project_capability_view_set(stage="planner_intent", authority=_authority())
    canonical_payload = {
        "contract_version": view.contract_version,
        "stage": view.stage,
        "manifest_refs": [ref.model_dump(mode="json") for ref in view.manifest_refs],
        "items": [item.model_dump(mode="json") for item in view.items],
    }
    encoded = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    # When: the view hash is recomputed from its canonical fields.
    actual = capability_view_hash(
        contract_version=view.contract_version,
        stage=view.stage,
        manifest_refs=view.manifest_refs,
        items=view.items,
    )

    # Then: the domain prefix binds the complete capability-view-set contract.
    domain_prefix = b"capability-view-set.v1\0"
    expected = f"cvh1:{hashlib.sha256(domain_prefix + encoded).hexdigest()}"
    assert actual == expected


@pytest.mark.parametrize(
    ("stage", "view_type"),
    (
        ("knowledge_composer", ComposerCapabilityViewSetV1),
        ("smalltalk_composer", ComposerCapabilityViewSetV1),
        ("alignment_verifier", VerifierCapabilityViewSetV1),
    ),
)
def test_selected_views_dedupe_refs_without_deduplicating_unit_associations(
    stage: str,
    view_type: type[ComposerCapabilityViewSetV1] | type[VerifierCapabilityViewSetV1],
) -> None:
    # Given: two units select the same manifest while the policy admits all manifests.
    selected = (_ref("answer_internal_company_knowledge"),)

    # When: the role view is projected from the repeated selection.
    view = project_capability_view_set(
        stage=stage,
        authority=_authority(selected=selected),
    )

    # Then: one selected manifest is visible and its identity remains positional.
    assert isinstance(view, view_type)
    assert view.manifest_refs == selected
    assert len(view.items) == 1
    assert view.items[0].manifest_ref == selected[0]
    assert view.items[0].internal_company_knowledge_required is True


def test_role_view_defaults_are_typed_and_provider_neutral() -> None:
    # Given: one internal-knowledge manifest selected for each model-facing role.
    selected = (_ref("answer_internal_company_knowledge"),)

    # When: planner, composer, and verifier views are built.
    planner = project_capability_view_set(
        stage="planner_intent",
        authority=_authority(eligible=selected),
    )
    composer = project_capability_view_set(
        stage="knowledge_composer",
        authority=_authority(selected=selected),
    )
    verifier = project_capability_view_set(
        stage="alignment_verifier",
        authority=_authority(selected=selected),
    )

    # Then: role contracts expose only machine-consumed, provider-neutral fields.
    planner_item = planner.items[0]
    composer_item = composer.items[0]
    verifier_item = verifier.items[0]
    assert planner_item.required_artifacts == ("document_context",)
    assert composer_item.capability_type == "answer"
    assert composer_item.required_artifacts == ("document_context",)
    assert composer_item.prohibited_output_types == ("actions", "mentions", "media")
    assert verifier_item.output_schema_hash.startswith("osh1:")
    assert verifier_item.verifier_primitives == (
        "output_schema",
        "required_evidence_present",
        "evidence_artifact_type_allowed",
        "forbidden_source_not_used",
        "abstention_correctness",
    )
    assert all(
        provider_field
        not in json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        for item in (planner_item, composer_item, verifier_item)
        for provider_field in (
            "document_mcp",
            "approved_static_knowledge",
            "allowed_source_types",
        )
    )


@pytest.mark.parametrize(
    ("stage", "selected"),
    (
        ("planner_intent", (_ref("general.smalltalk"),)),
        ("knowledge_composer", ()),
        ("alignment_verifier", ()),
    ),
)
def test_projection_rejects_missing_selected_refs_for_non_planner_roles(
    stage: str,
    selected: tuple[ManifestRefV1, ...],
) -> None:
    # Given: a valid authority with no selected manifest for a role that needs one.
    authority = _authority(selected=selected)

    # When/Then: the projection fails closed before model-visible data is produced.
    expected_error = (
        "planner_selected_manifest_refs_forbidden"
        if stage == "planner_intent"
        else "selected_manifest_refs_required"
    )
    with pytest.raises(CapabilityViewProjectionError, match=expected_error):
        project_capability_view_set(stage=stage, authority=authority)


@pytest.mark.parametrize(
    "authority",
    (
        CapabilityViewAuthorityV1(
            eligible_manifest_refs=(_ref("general.smalltalk"),),
            selected_manifest_refs=(_ref("weekly_report.product_list"),),
        ),
        CapabilityViewAuthorityV1(
            eligible_manifest_refs=(_ref("answer_internal_company_knowledge"),) * 2,
            selected_manifest_refs=(),
        ),
    ),
)
def test_projection_rejects_unselected_or_duplicate_manifest_refs(
    authority: CapabilityViewAuthorityV1,
) -> None:
    # Given: an authority with either a selection outside eligibility or a duplicate.

    # When/Then: projection rejects the malformed reference set deterministically.
    with pytest.raises(CapabilityViewProjectionError):
        project_capability_view_set(stage="knowledge_composer", authority=authority)


def test_projection_rejects_manifest_version_drift() -> None:
    # Given: a known manifest ID paired with a version not sealed by the registry.
    invalid = ManifestRefV1.model_construct(
        manifest_id="answer_internal_company_knowledge",
        manifest_version="2099-01-01.1",
    )

    # When/Then: both eligible and selected authority are rejected before projection.
    with pytest.raises(
        CapabilityViewProjectionError,
        match="manifest_ref_version_mismatch",
    ):
        project_capability_view_set(
            stage="alignment_verifier",
            authority=_authority(eligible=(invalid,), selected=(invalid,)),
        )
