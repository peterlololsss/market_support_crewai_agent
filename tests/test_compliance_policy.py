from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.compliance_policy import (
    NON_COMPLIANT_REASON_CODES,
    compliance_policy_prompt_lines,
    safe_fallback_text,
)
from market_support_crewai_agent.runtime.guardrails import (
    build_deterministic_fallback,
    validate_reply,
)
from market_support_crewai_agent.runtime.planning import ReplyPlan
from market_support_crewai_agent.runtime.policy import compile_policy
from market_support_crewai_agent.runtime.reply_agent import _planner_prompt
from market_support_crewai_agent.schemas import (
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
    SendWeeklyReportAction,
)


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "请问产品预计收益多少？",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_refusal_plan(reason_code: str) -> ReplyPlan:
    return ReplyPlan.model_validate(
        {
            "user_need": "refuse non-compliant request",
            "intent": "refusal",
            "compliance": {
                "is_compliant": False,
                "reason_code": reason_code,
                "reason": "planner selected a non-compliant reason code",
            },
            "confidence": 0.9,
        }
    )


@pytest.mark.parametrize("reason_code", NON_COMPLIANT_REASON_CODES)
def test_safe_fallback_text_is_valid_for_non_compliant_plans(reason_code):
    request = make_request()
    plan = make_refusal_plan(reason_code)
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(
            kind="unable_to_answer",
            text=safe_fallback_text(reason_code),
        ),
        actions=[],
    )

    result = validate_reply(
        response,
        compile_policy(request),
        derive_business_facts([], request),
        request,
        plan=plan,
        evidence_facts=[],
    )

    assert result.valid is True


@pytest.mark.parametrize(
    ("message", "reason_code", "expected_fragment"),
    [
        ("产品到期能有多少收益？", "expected_or_target_return", "不设置预计收益"),
        ("产品是保本的吧？", "principal_or_risk_guarantee", "不承诺保本"),
        ("其他管理人你们怎么看？", "peer_or_competitor_comparison", "不做横向评论"),
        ("加你微信了，通过一下", "private_contact_request", "当前群内沟通"),
        ("你们衍复自营盘收益多少？", "proprietary_trading_or_core_strategy", "Alpha 多因子研究框架"),
        ("发我一个四级估值表吧", "restricted_internal_document", "内部敏感材料"),
        ("赎回费可以免了吗？", "fee_waiver_request", "按照基金合同"),
        ("客户达不到直销门槛，想看下产品材料", "qualified_investor_or_threshold", "未确认适当性"),
        ("周末可以一起看电影吗？", "unrelated_request", "与衍复产品或服务无关"),
    ],
)
def test_non_compliant_reason_codes_force_safe_fallback(
    message,
    reason_code,
    expected_fragment,
):
    request = make_request(message=message)
    plan = make_refusal_plan(reason_code)
    bad_response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="可以，我帮您处理。"),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )

    validation = validate_reply(
        bad_response,
        compile_policy(request),
        derive_business_facts([], request),
        request,
        plan=plan,
        evidence_facts=[],
    )
    fallback = build_deterministic_fallback(
        validation,
        bad_response,
        compile_policy(request),
        derive_business_facts([], request),
        request,
    )

    assert validation.valid is False
    assert validation.repairable is False
    assert fallback.reply.kind == "unable_to_answer"
    assert expected_fragment in fallback.reply.text
    assert fallback.reply.mentions == []
    assert fallback.actions == []


def test_planner_prompt_uses_harness_compliance_policy_allowlist():
    request = make_request(message="赎回费可以免了吗？")
    prompt = _planner_prompt(
        request,
        history=[],
        action_history=[],
        canonical_context=canonicalize_request(request),
        policy=compile_policy(request),
    )

    assert "Compliance policy reason-code allowlist" in prompt
    for line in compliance_policy_prompt_lines():
        assert line in prompt
