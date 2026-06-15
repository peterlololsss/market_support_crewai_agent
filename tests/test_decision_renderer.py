from __future__ import annotations

from market_support_crewai_agent.runtime.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.decision import DecisionEngine
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.planning import IntentFrame, compile_intent_frame
from market_support_crewai_agent.runtime.policy import compile_policy
from market_support_crewai_agent.runtime.response_renderer import render_directive
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


def make_plan(request: ReplyRequest, **overrides):
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
    policy = compile_policy(request, doc_mcp_enabled=True)
    return compile_intent_frame(
        IntentFrame.model_validate(payload),
        request,
        canonicalize_request(request),
        policy,
    )


def resolved_fact(resolve_type: str, resolve_ref: str, **metadata) -> EvidenceFact:
    fact_type = {
        "material_pack": "material_pack_resolvable",
        "weekly_report": "weekly_report_resolvable",
        "monthly_report": "monthly_report_resolvable",
        "sales_mention": "sales_mention_resolvable",
    }[resolve_type]
    payload = {"status": "resolved", "resolve_ref": resolve_ref}
    payload.update(metadata)
    return EvidenceFact(
        fact_type=fact_type,
        value=True,
        resolve_type=resolve_type,
        metadata=payload,
    )


def test_decision_and_renderer_build_report_action_from_business_facts():
    request = make_request()
    plan = make_plan(request)
    policy = compile_policy(request)
    facts = [
        resolved_fact(
            "weekly_report",
            "weekly:ref",
            period="20260612",
            report_date="2026-06-12",
        ),
        resolved_fact("sales_mention", "sales:ref"),
    ]
    business_facts = derive_business_facts(facts, request)

    directive = DecisionEngine().decide(plan, business_facts, facts, request, policy)
    response = render_directive(directive, plan, business_facts, facts)

    assert directive.contract_version == "response-directive"
    assert directive.mode == "action"
    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_weekly_report"
    assert response.actions[0].resolve_ref == "weekly:ref"
    assert response.actions[0].period == "20260612"


def test_decision_requires_knowledge_composer_only_with_document_context():
    request = make_request(message="介绍一下衍复")
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = make_plan(
        request,
        user_need="answer knowledge question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        report_scope="none",
        requested_capabilities=["document_context"],
    )
    document_fact = EvidenceFact(
        fact_type="document_context",
        value="文档证据",
        source_type="document_mcp",
        source_id="doc-1",
    )
    directive = DecisionEngine().decide(
        plan,
        derive_business_facts([document_fact], request),
        [document_fact],
        request,
        policy,
    )

    assert directive.mode == "knowledge_answer"
    assert directive.requires_knowledge_composer is True

    missing = DecisionEngine().decide(
        plan,
        derive_business_facts([], request),
        [],
        request,
        policy,
    )

    assert missing.mode == "unable"
    assert missing.requires_knowledge_composer is False
    assert "文档证据" in missing.text


def test_decision_hands_off_unavailable_action_when_sales_resolves():
    request = make_request()
    plan = make_plan(request)
    policy = compile_policy(request)
    facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=False,
            resolve_type="weekly_report",
            metadata={"status": "missing"},
        ),
        resolved_fact("sales_mention", "sales:ref"),
    ]
    directive = DecisionEngine().decide(
        plan,
        derive_business_facts(facts, request),
        facts,
        request,
        policy,
    )

    assert directive.mode == "handoff"
    assert directive.reply_kind == "human_handoff"
    assert directive.mentions[0].type == "sales"
    assert directive.action_intents == []


def test_decision_uses_llm_composer_for_smalltalk_without_actions():
    request = make_request(message="hi")
    policy = compile_policy(request)
    plan = make_plan(
        request,
        user_need="greeting",
        artifact_kind="smalltalk",
        action_intent="none",
        report_scope="none",
        compliance={
            "is_compliant": True,
            "reason_code": "unrelated_request",
            "reason": "greeting",
        },
    )
    facts = []
    business_facts = derive_business_facts(facts, request)

    directive = DecisionEngine().decide(plan, business_facts, facts, request, policy)

    assert directive.mode == "smalltalk"
    assert directive.reply_kind == "answer"
    assert directive.text == ""
    assert directive.action_intents == []
    assert directive.requires_knowledge_composer is True
    assert directive.composer_stage == "smalltalk_composer"
