from __future__ import annotations

from typing import TypeAlias

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.planning.models import (
    AdapterResolveSpec,
    ComplianceDecision,
    ExecutionPlan,
)
from market_support_crewai_agent.runtime.domain.planning.input_policy import (
    match_input_policy,
)
from market_support_crewai_agent.runtime.domain.planning.validation import (
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest, compile_policy
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.orchestration.decision import DecisionEngine
from market_support_crewai_agent.runtime.orchestration.response_renderer import (
    render_directive,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    HANDOFF_TEXT_METADATA_KEY,
    GuardrailDecision,
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


def test_t0_mentions_do_not_bypass_document_planner():
    for message in ("T0怎么操作", "这个产品支持t0吗", "t0安排"):
        request = make_request(message=message)
        policy = compile_policy(request)
        result = match_input_policy(request, policy)

        assert result.status == "no_match"
        assert result.rule_id == ""
        assert result.plan is None


def test_t0_policy_leaves_adapter_read_allowlist_to_planner():
    request = make_request(
        message="这个T0可以做吗",
        allowed_read_capabilities=["query_internal_company_info"],
    )
    policy = compile_policy(request)
    result = match_input_policy(request, policy)

    assert result.status == "no_match"
    assert result.plan is None


def test_human_handoff_renderer_still_uses_guardrail_text():
    request = make_request(message="这个T0可以做吗")
    policy = compile_policy(request)
    plan = _handoff_plan(policy)
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


def _handoff_plan(policy: PolicyManifest) -> ExecutionPlan:
    plan = ExecutionPlan(
        user_need="human support",
        artifact_kind="human_support",
        response_mode="handoff",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="customer_service_request",
            reason="human support request",
        ),
        capabilities=["sales_mention"],
        adapter_resolves=[AdapterResolveSpec(resolve_type="sales_mention")],
        guardrail_decisions=[
            GuardrailDecision(
                outcome="block",
                phase="input",
                reason_code="human_support_required",
                human_readable_reason="这里用户需要您的帮助",
                metadata={
                    HANDOFF_TEXT_METADATA_KEY: "这个问题需要老师您向群内请销售/支持同事确认哦。我帮您艾特ta~",
                },
            )
        ],
        confidence=1.0,
    )
    assert validate_execution_plan(plan, policy).valid
    return plan
