from __future__ import annotations

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.orchestration.decision import DecisionEngine
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.orchestration.response_renderer import render_directive
from market_support_crewai_agent.runtime.validation.reply_validator import validate_reply
from market_support_crewai_agent.schemas import ReplyRequest
from tests.helpers.planning import compile_test_plan


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
        "available_artifacts": [
            {"type": "material_pack", "options": ["指增"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_plan(request: ReplyRequest, **overrides):
    payload = {
        "user_need": "send weekly report",
        "artifact_kind": "weekly_report",
        "action_intent": "send",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    policy = compile_policy(request, doc_mcp_enabled=True)
    return compile_test_plan(request, policy=policy, **payload)


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


def test_decision_adds_weekly_report_rationale_for_dynamic_metric_send():
    request = make_request(message="500最近回撤修复得怎么样")
    plan = make_plan(
        request,
        user_need="send weekly report for recent drawdown recovery",
        risk_flags=["weekly_report_rationale_required"],
    )
    policy = compile_policy(request)
    facts = [
        resolved_fact(
            "weekly_report",
            "weekly:ref",
            period="20260612",
            report_date="2026-06-12",
        ),
    ]
    business_facts = derive_business_facts(facts, request)

    directive = DecisionEngine().decide(plan, business_facts, facts, request, policy)
    response = render_directive(directive, plan, business_facts, facts)

    assert response.reply.text == "这个问题需要看最新周报里的近期表现数据，我先把周报发你，具体以报告为准。"
    assert response.actions[0].type == "send_weekly_report"


def test_decision_uses_composer_for_ambiguous_action_clarification():
    request = make_request(
        message="发一下材料",
        available_artifacts=[{"type": "material_pack", "options": ["中证1000指增", "中证A500指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
    )
    policy = compile_policy(request)
    plan = make_plan(
        request,
        artifact_kind="material_pack",
        action_intent="send",
    )
    facts = [
        EvidenceFact(
            fact_type="material_pack_resolvable",
            value=False,
            resolve_type="material_pack",
            metadata={
                "status": "ambiguous",
                "candidates": ["中证1000指增", "中证A500指增"],
            },
        )
    ]
    directive = DecisionEngine().decide(
        plan,
        derive_business_facts(facts, request),
        facts,
        request,
        policy,
    )

    assert directive.mode == "clarification"
    assert directive.text == ""
    assert directive.requires_knowledge_composer is True
    assert directive.composer_stage == "knowledge_composer"


def test_decision_treats_adapter_ambiguous_without_candidates_as_unavailable():
    request = make_request(message="发一下材料", available_artifacts=[{"type": "material_pack", "options": ["中证1000指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}])
    policy = compile_policy(request)
    plan = make_plan(
        request,
        artifact_kind="material_pack",
        action_intent="send",
    )
    facts = [
        EvidenceFact(
            fact_type="material_pack_resolvable",
            value=False,
            resolve_type="material_pack",
            metadata={"status": "ambiguous", "candidates": []},
        )
    ]

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts(facts, request),
        facts,
        request,
        policy,
    )

    assert directive.mode == "unable"
    assert directive.reply_kind == "unable_to_answer"
    assert directive.requires_knowledge_composer is False


def test_decision_requires_knowledge_composer_only_with_document_context():
    request = make_request(message="介绍一下衍复")
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = make_plan(
        request,
        user_need="answer knowledge question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
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
    assert missing.reason_code == "document_context_missing"
    assert "文档证据" not in missing.text


def test_decision_keeps_actions_for_mixed_answer_and_send_composer_path():
    request = make_request(message="介绍一下策略，然后发下周报")
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = make_plan(
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
    )
    facts = [
        resolved_fact(
            "weekly_report",
            "weekly:ref",
            period="20260612",
            report_date="2026-06-12",
        ),
        EvidenceFact(
            fact_type="document_context",
            value="文档证据",
            source_type="document_mcp",
            source_id="doc-1",
        ),
    ]

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts(facts, request),
        facts,
        request,
        policy,
    )

    assert directive.mode == "action"
    assert directive.action_intents[0].action_type == "send_weekly_report"
    assert directive.requires_knowledge_composer is True


def test_decision_keeps_action_when_mixed_answer_evidence_is_missing():
    request = make_request(
        message="\u4ec0\u4e48\u662f\u53e6\u7c7b\u6570\u636e\u56e0\u5b50\n\u4f60\u4eec\u4ee3\u9500\u7684\u7075\u6d3b\u5bf9\u51b2\u7684\u4ea7\u54c1\u6709\u54ea\u4e9b\u5440",
        available_artifacts=[{"type": "material_pack", "options": []}, {"type": "weekly_report"}, {"type": "monthly_report"}],
        channel_type="non_bank",
    )
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = make_plan(
        request,
        plan_units=[
            {
                "artifact_kind": "knowledge_answer",
                "action_intent": "answer",
                "requested_capabilities": ["document_context"],
            },
            {
                "artifact_kind": "material_pack",
                "action_intent": "send",
                "material_pack_option": "\u7075\u6d3b\u5bf9\u51b2",
            },
        ],
    )
    facts = [resolved_fact("material_pack", "material:ref")]
    business_facts = derive_business_facts(facts, request)

    directive = DecisionEngine().decide(
        plan,
        business_facts,
        facts,
        request,
        policy,
    )
    response = render_directive(directive, plan, business_facts, facts)
    validation = validate_reply(response, directive, plan, business_facts, facts, policy)

    assert directive.mode == "action"
    assert directive.reply_kind == "unable_to_answer"
    assert response.reply.kind == "unable_to_answer"
    assert response.actions[0].type == "send_material_pack"
    assert response.actions[0].material_pack_option is None
    assert validation.valid



def test_decision_allows_knowledge_composer_with_report_scope_evidence():
    request = make_request(message="why is A500 missing from this weekly report")
    policy = compile_policy(request)
    plan = make_plan(
        request,
        user_need="answer report scope question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["weekly_report"],
    )
    report_fact = EvidenceFact(
        fact_type="report_scope_summary",
        value=True,
        source_type="adapter_report_scope",
        source_id="weekly_report",
        resolve_type="weekly_report",
        metadata={"period": "20260612", "expected_product_count": 12},
    )

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts([report_fact], request),
        [report_fact],
        request,
        policy,
    )

    assert directive.mode == "knowledge_answer"
    assert directive.requires_knowledge_composer is True


def test_decision_renders_report_period_duration_without_composer():
    request = make_request(message="这个周报是什么时间段")
    policy = compile_policy(request)
    plan = make_plan(
        request,
        user_need="answer report period question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["weekly_report"],
    )
    report_fact = EvidenceFact(
        fact_type="report_period",
        value="20260612",
        source_type="adapter_resolve",
        source_id="weekly_report",
        resolve_type="weekly_report",
        metadata={
            "period": "20260612",
            "report_date": "2026-06-12",
            "period_start": "2026-06-08",
            "period_end": "2026-06-12",
        },
    )

    directive = DecisionEngine().decide(
        plan,
        derive_business_facts([report_fact], request),
        [report_fact],
        request,
        policy,
    )

    assert directive.mode == "knowledge_answer"
    assert directive.requires_knowledge_composer is False
    assert "2026-06-08" in directive.text
    assert "2026-06-12" in directive.text

    response = render_directive(
        directive,
        plan,
        derive_business_facts([report_fact], request),
        [report_fact],
    )
    assert response.reply.kind == "answer"
    assert "2026-06-08" in response.reply.text
    assert "2026-06-12" in response.reply.text


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
