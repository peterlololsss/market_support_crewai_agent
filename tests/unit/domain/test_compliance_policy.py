from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.compliance_policy import (
    NON_COMPLIANT_REASON_CODES,
    compliance_policy_prompt_lines,
    refusal_text_for_reason,
)
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.validation.reply_validator import validate_reply
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.llm.prompting.context import PromptAssemblyContext
from market_support_crewai_agent.runtime.llm.prompting.router import (
    route_intent,
    select_prompt_program,
)
from tests.helpers.planning import compile_test_plan
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
        "available_artifacts": [
            {"type": "material_pack", "options": ["中证500", "中证1000"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_refusal_plan(reason_code: str):
    request = make_request()
    return compile_test_plan(
        request,
        policy=compile_policy(request),
        user_need="refuse non-compliant request",
        artifact_kind="refusal",
        action_intent="refuse",
        compliance={
            "is_compliant": False,
            "reason_code": reason_code,
            "reason": "planner selected a non-compliant reason code",
        },
        confidence=0.9,
    )


@pytest.mark.parametrize("reason_code", NON_COMPLIANT_REASON_CODES)
def test_refusal_text_for_reason_is_valid_for_non_compliant_plans(reason_code):
    request = make_request()
    plan = make_refusal_plan(reason_code)
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(
            kind="unable_to_answer",
            text=refusal_text_for_reason(reason_code),
        ),
        actions=[],
    )

    result = validate_reply(
        response,
        ResponseDirective(
            mode="refusal",
            reply_kind="unable_to_answer",
            text=refusal_text_for_reason(reason_code),
            reason_code=reason_code,
        ),
        plan,
        derive_business_facts([], request),
        [],
        compile_policy(request),
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
def test_non_compliant_reason_codes_require_refusal_text_for_reason(
    message,
    reason_code,
    expected_fragment,
):
    request = make_request(message=message)
    plan = make_refusal_plan(reason_code)
    bad_response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="可以，我帮您处理。"),
        actions=[
            SendWeeklyReportAction(
                type="send_weekly_report",
                action_id="act-1",
                resolve_type="weekly_report",
                resolve_ref="weekly:ref",
                period="20260529",
                report_date="2026-05-29",
            )
        ],
    )

    validation = validate_reply(
        bad_response,
        ResponseDirective(
            mode="refusal",
            reply_kind="unable_to_answer",
            text=refusal_text_for_reason(reason_code),
            reason_code=reason_code,
        ),
        plan,
        derive_business_facts([], request),
        [],
        compile_policy(request),
    )

    assert validation.valid is False
    assert {issue.code for issue in validation.issues} >= {
        "non_compliant_reply_has_actions",
        "non_compliant_reply_kind",
        "non_compliant_reply_text",
    }
    assert expected_fragment in refusal_text_for_reason(reason_code)


def test_planner_prompt_uses_harness_compliance_policy_allowlist():
    request = make_request(message="赎回费可以免了吗？")
    canonical_context = canonicalize_request(request)
    policy = compile_policy(request)
    program = select_prompt_program(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            canonical_context=canonical_context,
            policy=policy,
            intent_gate=route_intent(request, canonical_context, policy),
        )
    )

    assert "compliance.reason_codes" in program.fragment_ids
    assert "Compliance reason-code allowlist" in program.prompt_text
    for line in compliance_policy_prompt_lines():
        assert line in program.prompt_text


def test_compliance_prompt_uses_universal_taxonomy_for_blocked_requests():
    request = make_request(message="能保本吗？")
    canonical_context = canonicalize_request(request)
    policy = compile_policy(request)
    program = select_prompt_program(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            canonical_context=canonical_context,
            policy=policy,
            intent_gate=route_intent(request, canonical_context, policy),
        )
    )

    assert "planner.intent_taxonomy" in program.fragment_ids
    assert "compliance.reason_codes" in program.fragment_ids
    assert "compliance.refusal_examples" not in program.fragment_ids
    assert "principal_or_risk_guarantee" in program.prompt_text


def test_compliance_prompt_program_records_fragment_hashes():
    request = make_request(message="预期收益多少？")
    canonical_context = canonicalize_request(request)
    policy = compile_policy(request)
    program = select_prompt_program(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            canonical_context=canonical_context,
            policy=policy,
            intent_gate=route_intent(request, canonical_context, policy),
        )
    )

    assert program.prompt_hash.startswith("sha256:")
    assert set(program.fragment_ids) == set(program.fragment_hashes)
    assert all(value.startswith("sha256:") for value in program.fragment_hashes.values())


def test_compliance_prompt_prioritizes_named_service_person_handoff():
    lines = compliance_policy_prompt_lines()
    customer_service_line = next(
        line for line in lines if line.startswith("- customer_service_request:")
    )
    private_contact_line = next(
        line for line in lines if line.startswith("- private_contact_request:")
    )

    assert "named Yanfu internal service person" in customer_service_line
    assert "priority over private_contact_request" in customer_service_line
    assert "Do not use this for normal human support" in private_contact_line
