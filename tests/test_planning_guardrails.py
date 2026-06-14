from __future__ import annotations

from market_support_crewai_agent.runtime.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.planning import (
    IntentFrame,
    compile_intent_frame,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.policy import compile_policy
from market_support_crewai_agent.schemas import ReplyRequest


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "请发一下周报",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["指增"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_frame(**overrides) -> IntentFrame:
    payload = {
        "user_need": "send weekly report",
        "artifact_kind": "weekly_report",
        "action_intent": "send",
        "report_scope": "channel_all",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    return IntentFrame.model_validate(payload)


def compile_plan(frame: IntentFrame, request: ReplyRequest):
    return compile_intent_frame(
        frame,
        request,
        canonicalize_request(request),
        compile_policy(request),
    )


def test_compile_intent_frame_uses_registry_for_weekly_action():
    request = make_request()
    plan = compile_plan(make_frame(), request)

    result = validate_execution_plan(plan, compile_policy(request))

    assert result.valid is True
    assert plan.artifact_kind == "weekly_report"
    assert plan.response_mode == "action"
    assert plan.capabilities == ["weekly_report"]
    assert [item.resolve_type for item in plan.adapter_resolves] == [
        "weekly_report",
        "sales_mention",
    ]
    assert plan.action_intents[0].action_type == "send_weekly_report"
    assert plan.action_intents[0].capability == "weekly_report"
    assert plan.action_intents[0].report_scope == "channel_all"


def test_compile_intent_frame_rejects_policy_disallowed_material_action():
    request = make_request(available_materials=["weekly"])
    plan = compile_plan(
        make_frame(
            user_need="send material pack",
            artifact_kind="material_pack",
            report_scope="none",
        ),
        request,
    )

    result = validate_execution_plan(plan, compile_policy(request))

    assert result.valid is False
    assert result.issues[0].code == "capability_not_allowed"
    assert result.issues[0].severity == "fatal"


def test_compile_intent_frame_defaults_report_scope_from_selected_strategy():
    request = make_request()
    plan = compile_plan(make_frame(report_scope="none"), request)

    result = validate_execution_plan(plan, compile_policy(request))

    assert result.valid is True
    assert plan.action_intents[0].report_scope == "strategy"
    assert plan.action_intents[0].strategy == "指增"


def test_compile_intent_frame_clarifies_strategy_scope_without_strategy():
    request = make_request(available_strategies=[])
    plan = compile_plan(make_frame(report_scope="strategy"), request)

    result = validate_execution_plan(plan, compile_policy(request))

    assert result.valid is True
    assert plan.response_mode == "clarification"
    assert plan.ambiguity_slots == ["strategy"]
    assert plan.action_intents == []


def test_compile_intent_frame_channel_all_ignores_selected_strategy_for_action():
    request = make_request()
    plan = compile_plan(
        make_frame(report_scope="channel_all", selected_strategy="中证1000"),
        request,
    )

    result = validate_execution_plan(plan, compile_policy(request))

    assert result.valid is True
    assert plan.action_intents[0].report_scope == "channel_all"
    assert plan.action_intents[0].strategy is None


def test_compile_intent_frame_for_refusal_has_no_actions_or_evidence():
    request = make_request()
    plan = compile_plan(
        make_frame(
            user_need="refuse expected return question",
            artifact_kind="refusal",
            action_intent="refuse",
            report_scope="none",
            compliance={
                "is_compliant": False,
                "reason_code": "expected_or_target_return",
                "reason": "expected return requests must be refused",
            },
        ),
        request,
    )

    result = validate_execution_plan(plan, compile_policy(request))

    assert result.valid is True
    assert plan.response_mode == "refusal"
    assert plan.action_intents == []
    assert plan.adapter_resolves == []
    assert plan.capabilities == []


def test_compile_intent_frame_allows_document_evidence_when_policy_enables_mcp():
    request = make_request(message="介绍一下衍复中证1000指数增强策略")
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = compile_intent_frame(
        make_frame(
            user_need="answer product knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            report_scope="none",
            requested_capabilities=["document_context"],
        ),
        request,
        canonicalize_request(request),
        policy,
    )

    result = validate_execution_plan(plan, policy)

    assert result.valid is True
    assert plan.response_mode == "knowledge_answer"
    assert plan.capabilities == ["document_context"]
    assert plan.action_intents == []


def test_compile_intent_frame_disables_document_answer_when_policy_disables_mcp():
    request = make_request(message="介绍一下衍复中证1000指数增强策略")
    plan = compile_plan(
        make_frame(
            user_need="answer product knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            report_scope="none",
            requested_capabilities=["document_context"],
        ),
        request,
    )

    result = validate_execution_plan(plan, compile_policy(request))

    assert result.valid is True
    assert plan.response_mode == "unable"
    assert plan.capabilities == []
