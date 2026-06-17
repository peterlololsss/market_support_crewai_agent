from __future__ import annotations

from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.planning import (
    compile_plan_spec,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from tests.helpers.planning import make_plan_spec, make_request


def test_compile_plan_spec_builds_weekly_report_action():
    request = make_request(available_strategies=[])
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="weekly_report",
            action_intent="send",
            report_scope="channel_all",
        ),
        request,
        canonicalize_request(request),
        policy,
    )

    assert plan.response_mode == "action"
    assert plan.capabilities == ["weekly_report"]
    assert [item.resolve_type for item in plan.adapter_resolves] == [
        "weekly_report",
        "sales_mention",
    ]
    assert plan.action_intents[0].action_type == "send_weekly_report"
    assert plan.action_intents[0].report_scope == "channel_all"
    assert plan.action_intents[0].strategy is None
    assert validate_execution_plan(plan, policy).valid


def test_compile_plan_spec_builds_strategy_scoped_report_action():
    request = make_request(available_strategies=["指增"])
    policy = compile_policy(request)

    plan = compile_plan_spec(
        make_plan_spec(
            request,
            artifact_kind="weekly_report",
            action_intent="send",
            report_scope="strategy",
        ),
        request,
        canonicalize_request(request),
        policy,
    )

    assert plan.response_mode == "action"
    assert plan.selected_strategy == "指增"
    assert plan.adapter_resolves[0].strategy == "指增"
    assert plan.action_intents[0].report_scope == "strategy"
    assert plan.action_intents[0].strategy == "指增"
    assert validate_execution_plan(plan, policy).valid


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
        canonicalize_request(request),
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
        canonicalize_request(request),
        policy,
    )

    assert plan.response_mode == "refusal"
    assert plan.capabilities == []
    assert plan.adapter_resolves == []
    assert plan.action_intents == []
    assert validate_execution_plan(plan, policy).valid
