from __future__ import annotations

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    DomainContextBuilder,
)
from market_support_crewai_agent.runtime.domain.compliance_policy import refusal_text_for_reason
from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    retrieval_source_guard,
)
from market_support_crewai_agent.runtime.validation.execution_tool_guard import (
    execution_tool_guard,
)
from market_support_crewai_agent.runtime.validation.reply_validator import (
    remove_pre_execution_send_claims,
    validate_reply,
)
from market_support_crewai_agent.runtime.domain.planning import (
    AdapterResolveSpec,
    ExecutionPlan,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.schemas import (
    PrimaryReply,
    ReplyResponse,
)
from market_support_crewai_agent.settings import Settings
from tests.helpers.structured_guardrails import (
    make_directive,
    make_plan,
    make_request,
    resolved_fact,
    weekly_action,
)


def test_validate_reply_allows_resolved_registry_action():
    request = make_request()
    plan = make_plan()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[weekly_action()],
    )
    facts = [resolved_fact("weekly_report", "weekly:ref")]

    result = validate_reply(
        response,
        make_directive(plan),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request),
    )

    assert result.valid is True


def test_validate_reply_reports_unresolved_outbound_action():
    request = make_request()
    plan = make_plan()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[weekly_action()],
    )

    result = validate_reply(
        response,
        make_directive(plan),
        plan,
        derive_business_facts([], request),
        [],
        compile_policy(request),
    )

    assert result.valid is False
    assert result.issues[0].code == "action_not_resolvable"
    assert result.issues[0].metadata["resolve_type"] == "weekly_report"


def test_validate_reply_blocks_action_not_in_execution_plan():
    request = make_request(message="请发月报")
    plan = make_plan(
        request,
        user_need="send monthly report",
        artifact_kind="monthly_report",
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[weekly_action()],
    )
    facts = [resolved_fact("weekly_report", "weekly:ref")]

    result = validate_reply(
        response,
        make_directive(plan, action_intents=plan.action_intents),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request),
    )

    assert result.valid is False
    assert result.issues[0].code == "action_not_in_directive"


def test_validate_reply_blocks_resolve_ref_mismatch():
    request = make_request()
    plan = make_plan()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[weekly_action(resolve_ref="weekly:wrong")],
    )
    facts = [resolved_fact("weekly_report", "weekly:ref")]

    result = validate_reply(
        response,
        make_directive(plan),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request),
    )

    assert result.valid is False
    assert result.issues[0].code == "action_resolve_ref_mismatch"
    assert result.issues[0].message == "action resolve_ref does not match adapter evidence"


def test_remove_pre_execution_send_claims_keeps_answer_text():
    text = (
        "本周报覆盖以下2只产品：\n\n"
        "Product1\n"
        "Product2\n\n"
        "月报已发送，请查收。"
    )

    cleaned = remove_pre_execution_send_claims(text)

    assert "Product1" in cleaned
    assert "Product2" in cleaned
    assert "已发送" not in cleaned
    assert "请查收" not in cleaned


def test_validate_reply_blocks_action_reply_text():
    request = make_request(message="请发周报")
    plan = make_plan()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="这份周报如下。"),
        actions=[weekly_action()],
    )
    facts = [resolved_fact("weekly_report", "weekly:ref")]

    result = validate_reply(
        response,
        make_directive(plan),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request),
    )

    assert result.valid is False
    assert any(
        issue.code == "outbound_action_reply_text_not_empty"
        for issue in result.issues
    )


def test_validate_reply_allows_action_reply_text_when_supplied_by_directive():
    request = make_request(message="请发周报，再说一下时间段")
    plan = make_plan()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="时间段是2026-05-26至2026-05-29。"),
        actions=[weekly_action()],
    )
    facts = [resolved_fact("weekly_report", "weekly:ref")]

    result = validate_reply(
        response,
        make_directive(
            plan,
            text="时间段是2026-05-26至2026-05-29。",
        ),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request),
    )

    assert result.valid is True


def test_validate_reply_allows_action_reply_text_from_knowledge_composer_with_evidence():
    request = make_request(message="weekly products, then send monthly")
    plan = make_plan()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="weekly report products: Product1"),
        actions=[weekly_action()],
    )
    facts = [
        resolved_fact("weekly_report", "weekly:ref"),
        EvidenceFact(
            fact_type="report_scope_products",
            value=True,
            source_type="adapter_report_scope",
            source_id="weekly_report",
            resolve_type="weekly_report",
            metadata={"products": [{"product_name": "Product1"}]},
        ),
    ]

    result = validate_reply(
        response,
        make_directive(
            plan,
            requires_knowledge_composer=True,
            composer_stage="knowledge_composer",
        ),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request),
    )

    assert result.valid is True


def test_validate_reply_checks_send_action_evidence_outside_composer_citations():
    request = make_request(message="介绍策略，然后发周报", available_artifacts=[{"type": "material_pack", "options": []}, {"type": "weekly_report"}, {"type": "monthly_report"}])
    plan = make_plan(
        request,
        plan_units=[
            {
                "artifact_kind": "knowledge_answer",
                "action_intent": "answer",
                "requested_capabilities": ["document_context"],
                "evidence_query": "介绍策略",
            },
            {
                "artifact_kind": "weekly_report",
                "action_intent": "send",
            },
        ],
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="策略说明。"),
        actions=[weekly_action()],
    )
    facts = [
        resolved_fact("weekly_report", "weekly:ref"),
        EvidenceFact(
            fact_type="document_context",
            value="策略说明。",
            source_type="document_mcp",
            source_id="doc-1",
            artifact_type="document_context",
        ),
    ]
    composer_output = ComposerReplyOutput(
        response_mode="answer",
        claims=["策略说明。"],
        evidence_ids=["document_mcp:doc-1:document_context"],
        reply=response.reply,
    )

    result = validate_reply(
        response,
        make_directive(
            plan,
            requires_knowledge_composer=True,
            composer_stage="knowledge_composer",
        ),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request, doc_mcp_enabled=True),
        composer_output=composer_output,
    )

    assert result.valid is True


def test_validate_reply_allows_report_action_without_report_scope_selector():
    request = make_request(message="请发中证1000周报")
    plan = make_plan(
        request,
        user_need="send weekly report",
        artifact_kind="weekly_report",
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[weekly_action()],
    )
    facts = [
        resolved_fact("weekly_report", "weekly:ref"),
    ]

    result = validate_reply(
        response,
        make_directive(plan),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request),
    )

    assert result.valid is True


def test_validate_reply_blocks_non_compliant_response_shape():
    request = make_request(message="请问产品预计收益多少？")
    plan = make_plan(
        request,
        user_need="refuse expected return request",
        artifact_kind="refusal",
        action_intent="refuse",
        compliance={
            "is_compliant": False,
            "reason_code": "expected_or_target_return",
            "reason": "expected return requests must be refused",
        },
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="可以参考周报。"),
        actions=[],
    )

    result = validate_reply(
        response,
        make_directive(
            plan,
            mode="refusal",
            reply_kind="unable_to_answer",
            text=refusal_text_for_reason(plan.compliance.reason_code),
            action_intents=[],
        ),
        plan,
        derive_business_facts([], request),
        [],
        compile_policy(request),
    )

    assert result.valid is False
    assert {issue.code for issue in result.issues} >= {
        "non_compliant_reply_kind",
        "non_compliant_reply_text",
    }


def test_validate_reply_requires_document_evidence_for_knowledge_answer():
    request = make_request(message="介绍一下衍复中证1000")
    plan = make_plan(
        request,
        user_need="answer knowledge question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal product question",
        },
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="衍复是一家量化私募。"),
        actions=[],
    )

    result = validate_reply(
        response,
        make_directive(
            plan,
            mode="knowledge_answer",
            reply_kind="answer",
            requires_knowledge_composer=True,
            action_intents=[],
        ),
        plan,
        derive_business_facts([], request),
        [],
        compile_policy(request, doc_mcp_enabled=True),
    )

    assert result.valid is False
    assert result.issues[0].code == "knowledge_answer_without_document_evidence"


def test_execution_tool_guard_blocks_invalid_artifact_id_before_execution():
    request = make_request()
    domain_context = DomainContextBuilder().build(request)
    plan = ExecutionPlan(
        user_need="send weekly report",
        artifact_kind="weekly_report",
        response_mode="action",
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report request",
        },
        capabilities=["weekly_report"],
        adapter_resolves=[
            AdapterResolveSpec(
                resolve_type="weekly_report",
                artifact_id="artifact:not-in-domain-context",
            )
        ],
    )

    decision = execution_tool_guard(
        plan=plan,
        policy=compile_policy(request),
        domain_context=domain_context,
    )

    assert decision.outcome == "block"
    assert decision.phase == "execution_tool"
    assert decision.reason_code == "invalid_artifact_id"


def test_execution_tool_guard_blocks_document_tool_when_policy_disallows_capability():
    request = make_request()
    plan = ExecutionPlan(
        user_need="answer from document",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal knowledge request",
        },
        capabilities=["document_context"],
    )

    decision = execution_tool_guard(
        plan=plan,
        policy=compile_policy(request),
        domain_context=DomainContextBuilder().build(request),
    )

    assert decision.outcome == "block"
    assert decision.reason_code == "tool_not_allowed"
    assert decision.metadata["tool_name"] == "document_mcp.get_documents"


def test_execution_tool_guard_allows_document_tool_through_capability_policy():
    request = make_request()
    plan = ExecutionPlan(
        user_need="answer from document",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal knowledge request",
        },
        capabilities=["document_context"],
    )

    decision = execution_tool_guard(
        plan=plan,
        policy=compile_policy(request, doc_mcp_enabled=True),
        domain_context=DomainContextBuilder().build(request),
    )

    assert decision.outcome == "allow"
    assert "document_mcp.get_documents" in decision.metadata["tools_seen"]


def test_validate_reply_allows_report_scope_evidence_for_knowledge_answer():
    request = make_request(message="why is A500 missing from this weekly report")
    plan = make_plan(
        request,
        user_need="answer report scope question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["weekly_report"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report question",
        },
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="The weekly report scope evidence says A500 was not matched."),
        actions=[],
    )
    report_fact = EvidenceFact(
        fact_type="report_scope_match",
        value="not_found",
        source_type="adapter_report_scope",
        source_id="weekly_report",
        resolve_type="weekly_report",
        metadata={"period": "20260612", "match": {"status": "not_found"}},
    )

    result = validate_reply(
        response,
        make_directive(
            plan,
            mode="knowledge_answer",
            reply_kind="answer",
            requires_knowledge_composer=True,
            action_intents=[],
        ),
        plan,
        derive_business_facts([report_fact], request),
        [report_fact],
        compile_policy(request),
    )

    assert result.valid is True


def test_validate_reply_allows_report_period_evidence_for_knowledge_answer():
    request = make_request(message="这个周报是什么时间段")
    plan = make_plan(
        request,
        user_need="answer report period question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["weekly_report"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report question",
        },
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="最新周报时间段为2026-06-08至2026-06-12。"),
        actions=[],
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

    result = validate_reply(
        response,
        make_directive(
            plan,
            mode="knowledge_answer",
            reply_kind="answer",
            requires_knowledge_composer=True,
            action_intents=[],
        ),
        plan,
        derive_business_facts([report_fact], request),
        [report_fact],
        compile_policy(request),
    )

    assert result.valid is True


def test_validate_reply_allows_knowledge_composer_to_downgrade_to_unable():
    request = make_request(message="介绍一下衍复")
    plan = make_plan(
        request,
        user_need="answer company question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal company question",
        },
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="unable_to_answer", text="老师，这个信息我这边暂时无法确认，先不回答避免信息不准确。"),
        actions=[],
    )

    result = validate_reply(
        response,
        make_directive(
            plan,
            mode="knowledge_answer",
            reply_kind="answer",
            requires_knowledge_composer=True,
            action_intents=[],
        ),
        plan,
        derive_business_facts([], request),
        [],
        compile_policy(request, doc_mcp_enabled=True),
    )

    assert result.valid is True


def test_validate_reply_allows_whitelisted_image_marker_from_evidence():
    request = make_request(message="你们有微信公众号吗")
    plan = make_plan(
        request,
        user_need="answer WeChat public account question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal company question",
        },
    )
    facts = [
        EvidenceFact(
            fact_type="document_context",
            value="Q：你们有微信公众号吗？\nA：%%comp_wx_qr_code.png%%",
            source_type="document_mcp",
            source_id="company",
        )
    ]
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="%%comp_wx_qr_code.png%%"),
        actions=[],
    )

    result = validate_reply(
        response,
        make_directive(
            plan,
            mode="knowledge_answer",
            reply_kind="answer",
            requires_knowledge_composer=True,
            action_intents=[],
        ),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request, doc_mcp_enabled=True),
    )

    assert result.valid is True


def test_validate_reply_blocks_image_marker_outside_whitelist():
    request = make_request(message="发个图")
    plan = make_plan(
        request,
        user_need="answer image question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal company question",
        },
    )
    facts = [
        EvidenceFact(
            fact_type="document_context",
            value="A：%%unknown.png%%",
            source_type="document_mcp",
            source_id="company",
        )
    ]
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="%%unknown.png%%"),
        actions=[],
    )

    result = validate_reply(
        response,
        make_directive(
            plan,
            mode="knowledge_answer",
            reply_kind="answer",
            requires_knowledge_composer=True,
            action_intents=[],
        ),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request, doc_mcp_enabled=True),
    )

    assert result.valid is False
    assert result.issues[0].code == "image_marker_not_allowed"


def test_validate_reply_blocks_image_marker_missing_from_evidence():
    request = make_request(message="你们有微信公众号吗")
    plan = make_plan(
        request,
        user_need="answer WeChat public account question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal company question",
        },
    )
    facts = [
        EvidenceFact(
            fact_type="document_context",
            value="A：欢迎搜索衍复投资公众号。",
            source_type="document_mcp",
            source_id="company",
        )
    ]
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="%%comp_wx_qr_code.png%%"),
        actions=[],
    )

    result = validate_reply(
        response,
        make_directive(
            plan,
            mode="knowledge_answer",
            reply_kind="answer",
            requires_knowledge_composer=True,
            action_intents=[],
        ),
        plan,
        derive_business_facts(facts, request),
        facts,
        compile_policy(request, doc_mcp_enabled=True),
    )

    assert result.valid is False
    assert result.issues[0].code == "image_marker_not_in_evidence"
