from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.capabilities.prompt_projection import (
    CapabilityViewAuthorityV1,
    CapabilityViewByteBudgetError,
    capability_view_serialized_bytes,
    project_capability_view_set,
    validate_capability_view_budget,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    PlannerCapabilityViewSetV1,
)


def _ref(manifest_id: str) -> ManifestRefV1:
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(manifest_id)
    assert manifest is not None
    return ManifestRefV1(
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.manifest_version,
    )


def _planner_view() -> PlannerCapabilityViewSetV1:
    refs = tuple(
        _ref(manifest.manifest_id) for manifest in CAPABILITY_MANIFEST_REGISTRY.list()
    )
    return project_capability_view_set(
        stage="planner_intent",
        authority=CapabilityViewAuthorityV1(
            eligible_manifest_refs=refs,
            selected_manifest_refs=(),
        ),
    )


def test_complete_serialized_view_is_bounded_without_clipping() -> None:
    # Given: the complete planner view with all 13 canonical items.
    view = _planner_view()
    serialized = capability_view_serialized_bytes(view)

    # When: the exact measured byte count is checked against the default stage ceiling.
    validate_capability_view_budget(view)

    # Then: every item remains complete and the measured JSON is valid UTF-8 bytes.
    assert serialized == view.model_dump_json(exclude_none=False).encode("utf-8")
    assert serialized.decode("utf-8").endswith("}")
    assert all(item.selection_contract.summary for item in view.items)


def test_budget_negative_boundary_is_exact_and_non_mutating() -> None:
    # Given: a valid planner view and its complete serialized byte count.
    view = _planner_view()
    measured = len(capability_view_serialized_bytes(view))

    # When/Then: a caller-owned ceiling at measured-1 rejects, while measured passes.
    validate_capability_view_budget(view, max_bytes=measured)
    with pytest.raises(
        CapabilityViewByteBudgetError, match="capability_view_byte_budget_exceeded"
    ):
        validate_capability_view_budget(view, max_bytes=measured - 1)
    assert len(view.items) == 13


def test_budget_checker_reports_oversize_without_clipping_semantic_values() -> None:
    # Given: a valid view and an intentionally tight caller-owned aggregate ceiling.
    view = _planner_view()
    baseline = capability_view_serialized_bytes(view)

    # When/Then: +1 over the measured bytes fails and no item is truncated or removed.
    with pytest.raises(CapabilityViewByteBudgetError) as error:
        validate_capability_view_budget(view, max_bytes=len(baseline) - 1)
    assert error.value.actual_bytes == len(baseline)
    assert error.value.max_bytes == len(baseline) - 1
    assert capability_view_serialized_bytes(view) == baseline
    assert all(item.selection_contract.summary for item in view.items)
