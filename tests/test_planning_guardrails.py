from __future__ import annotations

from market_support_crewai_agent.runtime.planning import (
    ReplyPlan,
    fallback_plan,
    validate_plan,
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


def make_plan(**overrides) -> ReplyPlan:
    payload = {
        "user_need": "send weekly report",
        "intent": "send_weekly_report",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report request",
        },
        "evidence_requests": [
            {
                "capability": "resolve_weekly_report",
                "reason": "confirm weekly report can be sent",
            }
        ],
        "business_checks": [
            {
                "check": "check_weekly_report_resolvable",
                "reason": "weekly report send requires adapter resolve",
            }
        ],
        "required_adapter_resolves": ["weekly_report"],
        "candidate_actions": [{"type": "send_weekly_report", "report_scope": "channel_all"}],
        "confidence": 0.8,
    }
    payload.update(overrides)
    return ReplyPlan.model_validate(payload)


def test_validate_plan_allows_policy_scoped_weekly_action():
    request = make_request()
    plan = make_plan()

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is True
    assert result.issues == ()


def test_validate_plan_rejects_action_missing_required_resolve():
    request = make_request()
    plan = make_plan(required_adapter_resolves=[])

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is False
    assert result.issues[0].code == "action_missing_required_resolve"


def test_validate_plan_rejects_report_action_missing_selector():
    request = make_request()
    plan = make_plan(candidate_actions=[{"type": "send_weekly_report"}])

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is False
    assert any(
        issue.code == "report_action_selector_missing"
        for issue in result.issues
    )


def test_validate_plan_rejects_strategy_report_selector_without_strategy():
    request = make_request()
    plan = make_plan(
        candidate_actions=[
            {
                "type": "send_weekly_report",
                "report_scope": "strategy",
            }
        ]
    )

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is False
    assert result.issues[0].code == (
        "report_action_strategy_selector_missing_strategy"
    )


def test_validate_plan_rejects_channel_all_report_selector_with_strategy():
    request = make_request()
    plan = make_plan(
        candidate_actions=[
            {
                "type": "send_weekly_report",
                "report_scope": "channel_all",
                "strategy": "中证1000",
            }
        ]
    )

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is False
    assert result.issues[0].code == "report_action_channel_all_selector_has_strategy"


def test_validate_plan_rejects_policy_disallowed_action():
    request = make_request(available_materials=["weekly"])
    plan = make_plan(
        intent="send_material_pack",
        evidence_requests=[
            {
                "capability": "resolve_material_pack",
                "reason": "confirm material pack can be sent",
            }
        ],
        business_checks=[
            {
                "check": "check_material_pack_resolvable",
                "reason": "material pack send requires adapter resolve",
            }
        ],
        required_adapter_resolves=["material_pack"],
        candidate_actions=[{"type": "send_material_pack"}],
    )

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is False
    assert result.issues[0].code == "action_not_allowed"
    assert result.issues[0].severity == "fatal"


def test_validate_plan_rejects_non_compliant_side_effect_action():
    request = make_request()
    plan = make_plan(
        intent="refusal",
        compliance={
            "is_compliant": False,
            "reason_code": "expected_or_target_return",
            "reason": "expected return requests must be refused",
        },
    )

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is False
    assert result.issues[0].code == "non_compliant_plan_has_actions"
    assert result.issues[0].severity == "fatal"


def test_validate_plan_rejects_unknown_compliance_with_action():
    request = make_request()
    plan = make_plan(
        compliance={
            "is_compliant": None,
            "reason_code": "unknown",
            "reason": "not enough context",
        }
    )

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is False
    assert result.issues[0].code == "unknown_compliance_has_actions"


def test_validate_plan_allows_document_evidence_when_policy_enables_mcp():
    request = make_request(message="介绍一下衍复中证1000指数增强策略")
    plan = make_plan(
        user_need="answer product knowledge question",
        intent="knowledge_qa",
        evidence_requests=[
            {
                "capability": "query_internal_company_info",
                "reason": "answer from approved document context",
            }
        ],
        business_checks=[],
        required_adapter_resolves=[],
        candidate_actions=[],
    )

    result = validate_plan(plan, compile_policy(request, doc_mcp_enabled=True))

    assert result.valid is True


def test_validate_plan_rejects_document_evidence_when_policy_disables_mcp():
    request = make_request(message="介绍一下衍复中证1000指数增强策略")
    plan = make_plan(
        user_need="answer product knowledge question",
        intent="knowledge_qa",
        evidence_requests=[
            {
                "capability": "query_internal_company_info",
                "reason": "answer from approved document context",
            }
        ],
        business_checks=[],
        required_adapter_resolves=[],
        candidate_actions=[],
    )

    result = validate_plan(plan, compile_policy(request))

    assert result.valid is False
    assert result.issues[0].code == "evidence_capability_not_allowed"
    assert result.issues[0].severity == "fatal"


def test_fallback_plan_is_valid_no_reply_shape():
    plan = fallback_plan("planner failed")

    result = validate_plan(plan, compile_policy(make_request()))

    assert plan.intent == "no_reply"
    assert plan.candidate_actions == []
    assert result.valid is True
