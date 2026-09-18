from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.context.stage_inputs import (
    build_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    PlannerCapabilityViewSetV1,
)
from tests.unit.llm._stage_input_fixtures import stage_sources


def test_planner_stage_input_exposes_hash_bound_capability_views() -> None:
    # Given: a valid planner source with a typed eligible-capability projection.
    planner_source, _, _, _ = stage_sources()

    # When: the strict planner input is constructed.
    planner_input = build_planner_prompt_input_v1(planner_source)
    capability_view = planner_input.eligible_capabilities

    # Then: its machine contract and content hash reject payload mutation.
    assert capability_view.stage == "planner_intent"
    assert (
        capability_view.manifest_refs
        == planner_input.effective_policy.eligible_capabilities
    )
    tampered = capability_view.model_dump(mode="json")
    tampered["items"][0]["description"] += " changed"
    with pytest.raises(ValueError, match="view_hash_mismatch"):
        _ = PlannerCapabilityViewSetV1.model_validate(tampered)
