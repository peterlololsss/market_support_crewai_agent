from __future__ import annotations

from market_support_crewai_agent.runtime.context.stage_inputs import (
    build_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.prompts.router import (
    model_family_from_settings,
    route_intent,
    select_stage_input_prompt_program,
    user_facing_fragment_ids,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.planning import make_request
from tests.unit.llm._stage_input_fixtures import stage_sources


def _planner_program(message: str):
    request = make_request(message=message)
    scope = business_scope_authority_v1(request.business_scope)
    policy = PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(request, scope),
        policy_ledger_summary_v1((), 0),
    )
    gate = route_intent(request, policy)
    planner_source, _, _, _ = stage_sources(message)
    program = select_stage_input_prompt_program(
        build_planner_prompt_input_v1(planner_source),
        "ds_v4pro",
    )
    return gate, program


def test_route_intent_is_non_authoritative_audit_hint() -> None:
    # Given: semantically different support messages.
    first_gate, _ = _planner_program("发一下周报")
    second_gate, _ = _planner_program("介绍下你们公司")

    # When/Then: the deterministic audit hint does not choose product or action authority.
    assert first_gate.artifact_hint == "unclear"
    assert second_gate.artifact_hint == "unclear"
    assert first_gate.outbound_action_hint is False
    assert second_gate.outbound_action_hint is False
    assert first_gate.confidence == 0.0
    assert second_gate.confidence == 0.0


def test_representative_messages_use_same_registered_group_program() -> None:
    # Given: multiple user messages routed through the same scene and stage.
    programs = tuple(
        _planner_program(message)[1]
        for message in (
            "发一下周报",
            "介绍下你们公司",
            "请找销售老师",
            "材料和周报都给我",
        )
    )

    # When/Then: message content cannot change registered instructions or ceilings.
    assert {program.program_id for program in programs} == {
        "planner_intent.wecom_group.v1@1"
    }
    assert len({program.fragment_ids for program in programs}) == 1
    assert len({program.static_bytes for program in programs}) == 1
    assert all(program.static_bytes == program.baseline_bytes for program in programs)


def test_ds_v4pro_uses_new_precedence_source_and_one_scene_fragment() -> None:
    # Given: the ds_v4pro group planner registration.
    fragment_ids = user_facing_fragment_ids(
        "planner_intent",
        "ds_v4pro",
        "group",
    )

    # When/Then: it has the ds model fragment, new precedence source, and one scene.
    assert "model.ds_v4pro.structured" in fragment_ids
    assert "model.generic.structured" not in fragment_ids
    assert fragment_ids.count("instruction.registered_over_untrusted_data.v1") == 1
    assert tuple(value for value in fragment_ids if value.startswith("scene.")) == (
        "scene.wecom_group.planner_intent.v1",
    )


def test_non_ds_families_reuse_generic_precedence_source_once() -> None:
    # Given: every non-ds user-facing model family.
    families = ("deepseek", "gpt", "claude", "generic")

    # When/Then: each uses only the byte-frozen generic structured source.
    for family in families:
        fragment_ids = user_facing_fragment_ids(
            "knowledge_composer",
            family,
            "direct",
        )
        assert fragment_ids.count("model.generic.structured") == 1
        assert "model.ds_v4pro.structured" not in fragment_ids
        assert "instruction.registered_over_untrusted_data.v1" not in fragment_ids


def test_planner_override_model_family_is_stage_scoped() -> None:
    # Given: distinct planner and composer model settings.
    settings = Settings(
        llm_model="deepseek-v4-pro",
        planner_llm_model="gemini-3-flash-preview",
    )

    # When/Then: each stage derives its own registered model family.
    assert model_family_from_settings(settings) == "ds_v4pro"
    assert model_family_from_settings(settings, stage="planner_intent") == "generic"
