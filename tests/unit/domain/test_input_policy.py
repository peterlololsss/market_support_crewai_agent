from __future__ import annotations

from typing import TypeAlias

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.planning.input_policy import (
    InputPolicyRule,
    match_input_policy,
)
from market_support_crewai_agent.runtime.domain.planning.validation import (
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.orchestration.decision import DecisionEngine
from market_support_crewai_agent.runtime.orchestration.response_renderer import (
    render_directive,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY,
    HANDOFF_TEXT_METADATA_KEY,
)
from market_support_crewai_agent.schemas import ReplyRequest

AvailableArtifactPayload: TypeAlias = dict[str, str | list[str]]
RequestOverrideValue: TypeAlias = str | bool | list[str] | list[AvailableArtifactPayload]
RequestPayload: TypeAlias = dict[str, RequestOverrideValue]


def make_request(**overrides: RequestOverrideValue) -> ReplyRequest:
    payload: RequestPayload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "hello",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_artifacts": [
            {"type": "material_pack", "options": []},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def test_t0_mentions_route_to_guardrail_handoff():
    for message in (
        "你们有没有T0策略",
        "T0怎么操作",
        "这个产品支持t0吗",
        "t0安排",
        "Ｔ ０可以做吗",
        "T+0策略",
    ):
        request = make_request(message=message)
        policy = compile_policy(request)
        result = match_input_policy(request, policy)

        assert result.status == "guardrail_handoff"
        assert result.reason_code == "t0_human_support_required"
        assert result.rule_id == "t0_handoff"
        assert result.plan is not None
        assert result.plan.response_mode == "handoff"
        assert result.plan.capabilities == ["sales_mention"]
        assert [item.resolve_type for item in result.plan.adapter_resolves] == [
            "sales_mention"
        ]
        assert validate_execution_plan(result.plan, policy).valid
        decision = result.plan.guardrail_decisions[0]
        assert decision.phase == "input"
        assert decision.outcome == "block"
        assert decision.reason_code == "t0_human_support_required"
        assert decision.metadata[HANDOFF_TEXT_METADATA_KEY]
        assert decision.metadata[HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY]


def test_non_t0_message_still_leaves_planner():
    request = make_request(message="这个产品可以做吗")
    policy = compile_policy(request)
    result = match_input_policy(request, policy)

    assert result.status == "no_match"
    assert result.plan is None


def test_t0_handoff_rule_respects_policy_without_sales_mention_resolve():
    request = make_request(
        message="这个T0可以做吗",
        allowed_read_capabilities=["query_internal_company_info"],
    )
    policy = compile_policy(request)

    result = match_input_policy(request, policy)

    assert result.status == "guardrail_handoff"
    assert result.plan is not None
    assert result.plan.capabilities == []
    assert result.plan.adapter_resolves == []
    assert result.plan.action_intents == []
    assert validate_execution_plan(result.plan, policy).valid


def test_input_policy_accepts_injected_rule_table():
    request = make_request(message="人工处理这个测试触发")
    policy = compile_policy(request)
    rule = InputPolicyRule(
        rule_id="custom_handoff",
        reason_code="custom_handoff_required",
        contains=("测试触发",),
        user_need="custom human support",
        handoff_text="这个测试问题需要销售/支持同事确认",
        handoff_unavailable_text="这个测试问题需要销售/支持同事确认",
        human_reason="custom handoff required.",
    )

    result = match_input_policy(request, policy, rules=(rule,))

    assert result.status == "guardrail_handoff"
    assert result.rule_id == "custom_handoff"
    assert result.reason_code == "custom_handoff_required"
    assert result.plan is not None
    assert validate_execution_plan(result.plan, policy).valid


def test_human_handoff_renderer_still_uses_guardrail_text():
    request = make_request(message="这个T0可以做吗")
    policy = compile_policy(request)
    result = match_input_policy(request, policy)
    assert result.plan is not None
    plan = result.plan
    assert validate_execution_plan(plan, policy).valid
    facts = [
        EvidenceFact(
            fact_type="sales_mention_resolvable",
            value=True,
            resolve_type="sales_mention",
            metadata={"status": "resolved", "resolve_ref": "sales:ref"},
        )
    ]
    business_facts = derive_business_facts(facts, request)

    directive = DecisionEngine().decide(
        plan,
        business_facts,
        facts,
        request,
        policy,
    )
    response = render_directive(directive, plan, business_facts, facts)

    assert response.reply.kind == "human_handoff"
    assert response.reply.text == "这个问题需要老师您向群内请销售/支持同事确认哦。我帮您艾特ta~。"
    assert response.reply.mentions[0].type == "sales"
    assert response.reply.mentions[0].reason is None
    assert response.actions == []
