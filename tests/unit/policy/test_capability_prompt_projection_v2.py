from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    prompt_projection as projection,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    CapabilityViewStageV1,
    ComposerCapabilityViewSetV1,
    VerifierCapabilityViewSetV1,
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
    selected: tuple[ManifestRefV1, ...],
) -> projection.CapabilityViewAuthorityV1:
    return projection.CapabilityViewAuthorityV1(
        eligible_manifest_refs=(
            _ref("answer_internal_company_knowledge"),
            _ref("weekly_report.product_list"),
        ),
        selected_manifest_refs=selected,
    )


@pytest.mark.parametrize("stage", ["knowledge_composer", "smalltalk_composer"])
def test_composer_view_exposes_only_selected_provider_neutral_contract(
    stage: CapabilityViewStageV1,
) -> None:
    # Given: policy admits two manifests while the validated plan selects one.
    selected_ref = _ref("answer_internal_company_knowledge")

    # When: the canonical stage projection builds the composer view.
    view = projection.project_capability_view_set(
        stage=stage,
        authority=_authority(selected=(selected_ref,)),
    )

    # Then: only the selected provider-neutral composer contract is serialized.
    assert isinstance(view, ComposerCapabilityViewSetV1)
    assert view.manifest_refs == (selected_ref,)
    item = view.items[0]
    assert item.composer_constraints
    assert not hasattr(item, "allowed_source_types")
    assert not hasattr(item, "forbidden_source_types")
    assert not hasattr(item, "verifier_primitives")


def test_verifier_view_exposes_selected_primitives_and_structured_postconditions() -> (
    None
):
    # Given: one selected manifest inside a broader eligible policy set.
    selected_ref = _ref("answer_internal_company_knowledge")

    # When: the canonical stage projection builds the verifier view.
    view = projection.project_capability_view_set(
        stage="alignment_verifier",
        authority=_authority(selected=(selected_ref,)),
    )

    # Then: verifier primitives become required postconditions without provider data.
    assert isinstance(view, VerifierCapabilityViewSetV1)
    assert view.manifest_refs == (selected_ref,)
    item = view.items[0]
    assert tuple(postcondition.primitive for postcondition in item.postconditions) == (
        item.verifier_primitives
    )
    assert all(postcondition.required for postcondition in item.postconditions)
    assert not hasattr(item, "composer_constraints")
    assert not hasattr(item, "selection_contract")


def test_selected_view_rejects_ref_outside_policy_authority() -> None:
    # Given: a plan-selected ref that is not in policy eligibility.
    authority = projection.CapabilityViewAuthorityV1(
        eligible_manifest_refs=(_ref("general.smalltalk"),),
        selected_manifest_refs=(_ref("weekly_report.product_list"),),
    )

    # When/Then: projection fails before creating model-visible data.
    with pytest.raises(
        projection.CapabilityViewProjectionError,
        match="selected_manifest_ref_not_eligible",
    ):
        _ = projection.project_capability_view_set(
            stage="knowledge_composer",
            authority=authority,
        )


def test_selected_view_rejects_manifest_version_mismatch() -> None:
    # Given: a canonical ID paired with a non-canonical version at the trust boundary.
    invalid_ref = ManifestRefV1.model_construct(
        manifest_id="answer_internal_company_knowledge",
        manifest_version="2099-01-01.1",
    )
    authority = projection.CapabilityViewAuthorityV1(
        eligible_manifest_refs=(invalid_ref,),
        selected_manifest_refs=(invalid_ref,),
    )

    # When/Then: registry version drift is rejected before projection.
    with pytest.raises(
        projection.CapabilityViewProjectionError,
        match="manifest_ref_version_mismatch",
    ):
        _ = projection.project_capability_view_set(
            stage="alignment_verifier",
            authority=authority,
        )
