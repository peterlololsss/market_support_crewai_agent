from __future__ import annotations

from market_support_crewai_agent.runtime.domain.planning import (
    compile_plan_spec,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from tests.helpers.planning import make_plan_spec, make_request


def test_compile_plan_spec_builds_weekly_report_action():
    request = make_request(available_artifacts=[{"type": "material_pack", "options": []}, {"type": "weekly_report"}, {"type": "monthly_report"}])
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="weekly_report",
            action_intent="send",
        ),
        request,
        policy,
    )

    assert plan.response_mode == "action"
    assert plan.capabilities == ["weekly_report"]
    assert [item.resolve_type for item in plan.adapter_resolves] == [
        "weekly_report",
        "sales_mention",
    ]
    assert plan.action_intents[0].action_type == "send_weekly_report"
    assert plan.action_intents[0].material_pack_option is None
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_does_not_scope_report_send_by_material_pack_option():
    request = make_request(available_artifacts=[{"type": "material_pack", "options": ["指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}])
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="weekly_report",
            action_intent="send",
            material_pack_option="指增",
        ),
        request,
        policy,
    )

    assert plan.response_mode == "action"
    assert plan.material_pack_option is None
    assert plan.adapter_resolves[0].material_pack_option is None
    assert plan.action_intents[0].material_pack_option is None
    assert validate_execution_plan(plan, policy).valid


def test_material_pack_scope_must_be_request_option():
    request = make_request(available_artifacts=[{"type": "material_pack", "options": ["中证A500"]}, {"type": "weekly_report"}, {"type": "monthly_report"}])
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="material_pack",
            action_intent="send",
            material_pack_option="中证500",
        ),
        request,
        policy,
    )

    validation = validate_execution_plan(plan, policy)

    assert not validation.valid
    assert validation.issues[0].code == "material_pack_scope_not_allowed"


def test_compile_plan_spec_uses_document_context_for_knowledge_answer():
    request = make_request(message="介绍一下中证1000")
    policy = compile_policy(request, doc_mcp_enabled=True)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
        ),
        request,
        policy,
    )

    assert plan.response_mode == "knowledge_answer"
    assert plan.capabilities == ["document_context"]
    assert plan.answer_capabilities == ["document_context"]
    assert plan.action_intents == []
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_routes_report_product_question_to_evidence():
    request = make_request(message="周报里有哪些产品")
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["weekly_report"],
            evidence_query="report_scope_products",
        ),
        request,
        policy,
    )

    assert plan.response_mode == "knowledge_answer"
    assert plan.capabilities == ["weekly_report"]
    assert plan.ambiguity_slots == []
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_routes_general_product_availability_to_material_pack_send():
    request = make_request(
        message=(
            "你们代销的灵活对冲"
            "的产品有哪些呀"
        ),
        available_artifacts=[{"type": "material_pack", "options": ["flexible_hedge"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
    )
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="material_pack",
            action_intent="send",
            material_pack_option="flexible_hedge",
        ),
        request,
        policy,
    )

    assert plan.response_mode == "action"
    assert plan.capabilities == ["material_pack"]
    assert [item.resolve_type for item in plan.adapter_resolves] == [
        "material_pack",
        "sales_mention",
    ]
    assert plan.action_intents[0].action_type == "send_material_pack"
    assert plan.action_intents[0].material_pack_option == "flexible_hedge"
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_drops_material_pack_option_when_request_has_single_current_pack():
    request = make_request(
        message="你们代销的灵活对冲的产品有哪些呀",
        available_artifacts=[{"type": "material_pack", "options": []}, {"type": "weekly_report"}, {"type": "monthly_report"}],
    )
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="material_pack",
            action_intent="send",
            material_pack_option="灵活对冲",
        ),
        request,
        policy,
    )

    assert plan.response_mode == "action"
    assert plan.material_pack_option is None
    assert plan.adapter_resolves[0].material_pack_option is None
    assert plan.action_intents[0].material_pack_option is None
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_builds_multiple_send_actions():
    request = make_request(available_artifacts=[{"type": "material_pack", "options": ["指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}])
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            plan_units=[
                {
                    "artifact_kind": "material_pack",
                    "action_intent": "send",
                    "material_pack_option": "指增",
                },
                {
                    "artifact_kind": "weekly_report",
                    "action_intent": "send",
                },
            ],
        ),
        request,
        policy,
    )

    assert plan.response_mode == "action"
    assert plan.artifact_kind == "multi_action"
    assert plan.capabilities == ["material_pack", "weekly_report"]
    assert [item.resolve_type for item in plan.adapter_resolves] == [
        "material_pack",
        "weekly_report",
        "sales_mention",
    ]
    assert [item.action_type for item in plan.action_intents] == [
        "send_material_pack",
        "send_weekly_report",
    ]
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_builds_mixed_answer_and_send_plan():
    request = make_request(message="介绍一下策略，然后发下周报", available_artifacts=[{"type": "material_pack", "options": []}, {"type": "weekly_report"}, {"type": "monthly_report"}])
    policy = compile_policy(request, doc_mcp_enabled=True)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            plan_units=[
                {
                    "artifact_kind": "knowledge_answer",
                    "action_intent": "answer",
                    "requested_capabilities": ["document_context"],
                },
                {
                    "artifact_kind": "weekly_report",
                    "action_intent": "send",
                },
            ],
        ),
        request,
        policy,
    )

    assert plan.response_mode == "action"
    assert plan.capabilities == ["document_context", "weekly_report"]
    assert plan.answer_capabilities == ["document_context"]
    assert [item.action_type for item in plan.action_intents] == ["send_weekly_report"]
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_keeps_answer_when_one_clause_abstains():
    request = make_request(
        message="什么是另类数据因子，1000指增1号自建仓以来的收益",
        available_artifacts=[{"type": "material_pack", "options": []}, {"type": "weekly_report"}, {"type": "monthly_report"}],
    )
    policy = compile_policy(request, doc_mcp_enabled=True)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            plan_units=[
                {
                    "artifact_kind": "knowledge_answer",
                    "action_intent": "answer",
                    "requested_capabilities": ["document_context"],
                },
                {
                    "artifact_kind": "unclear",
                    "action_intent": "abstain",
                },
            ],
        ),
        request,
        policy,
    )

    assert plan.response_mode == "knowledge_answer"
    assert plan.capabilities == ["document_context"]
    assert plan.answer_capabilities == ["document_context"]
    assert plan.action_intents == []
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_refusal_has_no_actions_or_evidence():
    request = make_request(message="违规请求")
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="refusal",
            action_intent="refuse",
            compliance={
                "is_compliant": False,
                "reason_code": "unknown",
                "reason": "blocked",
            },
        ),
        request,
        policy,
    )

    assert plan.response_mode == "refusal"
    assert plan.capabilities == []
    assert plan.adapter_resolves == []
    assert plan.action_intents == []
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_carries_artifact_clarification_slot():
    request = make_request()
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="unclear",
            action_intent="none",
            ambiguity_slots=["artifact"],
        ),
        request,
        policy,
    )

    assert plan.response_mode == "clarification"
    assert plan.ambiguity_slots == ["artifact"]
    assert validate_execution_plan(plan, policy).valid


def test_report_query_is_not_a_user_clarification_slot():
    request = make_request()
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="weekly_report",
            action_intent="answer",
            ambiguity_slots=["report_query"],
        ),
        request,
        policy,
    )

    validation = validate_execution_plan(plan, policy)

    assert plan.response_mode == "clarification"
    assert plan.ambiguity_slots == []
    assert not validation.valid
    assert validation.issues[0].code == "clarification_missing_supported_slot"
