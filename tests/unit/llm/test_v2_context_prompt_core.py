from __future__ import annotations

from inspect import signature
from typing import TypedDict

from pydantic import TypeAdapter

from market_support_crewai_agent.runtime.context.stage_inputs import (
    PlannerPromptInputV1,
)
from market_support_crewai_agent.runtime.planning import finalize_execution_plan_v2
from market_support_crewai_agent.runtime.prompts.assembler import (
    assemble_prompt_program,
)
from market_support_crewai_agent.runtime.prompts.context import (
    StrictStageInputV1,
    render_prompt_context_layers,
)
from market_support_crewai_agent.runtime.prompts.router import route_intent
from market_support_crewai_agent.runtime.reply_history import compact_assistant_result
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from tests.unit.planning._execution_plan_v2_fixtures import (
    weekly_with_material_clarification_fixture,
)


class _PendingPlanV1(TypedDict):
    ambiguity_slots: list[str]
    artifact_kind: str
    capabilities: list[str]
    material_pack_option: str | None
    response_mode: str


class _CompactAssistantResultV1(TypedDict):
    pending_plan: _PendingPlanV1


def test_prompt_assembly_accepts_only_strict_stage_inputs() -> None:
    # Given: the public prompt assembly and context-rendering annotations.
    assembly_signature = str(signature(assemble_prompt_program))
    rendering_signature = str(signature(render_prompt_context_layers))

    # Then: both boundaries expose only the strict stage-input union.
    assert "ctx: 'StrictStageInputV1'" in assembly_signature
    assert "ctx: 'StrictStageInputV1'" in rendering_signature
    assert PlannerPromptInputV1 in StrictStageInputV1.__args__


def test_route_intent_requires_policy_manifest_v2() -> None:
    # Given: the active deterministic intent-hint boundary.
    route_signature = str(signature(route_intent))

    # When/Then: policy authority is the canonical V2 model.
    assert "policy: 'PolicyManifestV2'" in route_signature


def test_reply_history_requires_execution_plan_v2() -> None:
    # Given: the assistant-history serialization boundary.
    history_signature = str(signature(compact_assistant_result))

    # When/Then: pending-plan metadata accepts only the canonical V2 plan.
    assert "plan: 'ExecutionPlanV2'" in history_signature


def test_reply_history_compacts_v2_multi_unit_clarification_metadata() -> None:
    # Given: a V2 plan with an executable unit and a separate clarification unit.
    fixture = weekly_with_material_clarification_fixture()
    plan = finalize_execution_plan_v2(
        fixture.spec,
        fixture.policy,
        fixture.scope,
        origin="planner",
    )
    response = ReplyResponse(
        reply=PrimaryReply(
            kind="clarification",
            text="Please choose a material-pack option.",
            mentions=[],
        ),
        actions=[],
    )

    # When: the result is serialized for bounded conversation history.
    payload = TypeAdapter(_CompactAssistantResultV1).validate_json(
        compact_assistant_result(response, plan)
    )

    # Then: unit-scoped V2 metadata is flattened without restoring legacy fields.
    assert payload["pending_plan"] == {
        "ambiguity_slots": ["material_pack_option"],
        "artifact_kind": "weekly_report",
        "capabilities": ["weekly_report"],
        "material_pack_option": None,
        "response_mode": "action",
    }
