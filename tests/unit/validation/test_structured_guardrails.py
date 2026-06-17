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
    material_answer_plan,
    material_product_fact,
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
        report_scope="channel_all",
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


def test_validate_reply_does_not_use_send_claim_keywords_as_final_guard():
    request = make_request(
        message="材料包里有哪些产品",
        available_strategies=["指增"],
    )
    plan = material_answer_plan(request)
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="材料包包含：Product A。周报已发送，请查收。"),
        actions=[],
    )
    fact = material_product_fact("Product A")
    evidence_id = "adapter_material_pack_content:material_pack:material_pack_product_list"
    composer_output = ComposerReplyOutput(
        response_id="resp-1",
        response_mode="answer",
        claims=["材料包包含 Product A"],
        evidence_ids=[evidence_id],
        reply=response.reply,
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
        derive_business_facts([fact], request),
        [fact],
        compile_policy(request),
        domain_context=DomainContextBuilder().build(request, available_artifacts=[fact]),
        composer_output=composer_output,
    )

    assert result.valid is True


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
    request = make_request(message="è¯·å‘å‘¨æŠ¥ï¼Œå†è¯´ä¸€ä¸‹æ—¶é—´æ®µ")
    plan = make_plan()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="æ—¶é—´æ®µæ˜¯2026-05-26è‡³2026-05-29ã€‚"),
        actions=[weekly_action()],
    )
    facts = [resolved_fact("weekly_report", "weekly:ref")]

    result = validate_reply(
        response,
        make_directive(
            plan,
            text="æ—¶é—´æ®µæ˜¯2026-05-26è‡³2026-05-29ã€‚",
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


def test_validate_reply_blocks_report_action_when_evidence_excludes_strategy():
    request = make_request(message="请发中证1000周报")
    plan = make_plan(
        request,
        user_need="send weekly report covering 中证1000",
        artifact_kind="weekly_report",
        report_scope="strategy",
        selected_strategy="中证1000",
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[weekly_action(report_scope="strategy", strategy="中证1000")],
    )
    facts = [
        resolved_fact("weekly_report", "weekly:ref", strategy="中证1000"),
        EvidenceFact(
            fact_type="report_contains_strategy",
            value=False,
            resolve_type="weekly_report",
            metadata={"strategy": "中证1000"},
        ),
    ]

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
        issue.code == "report_action_strategy_unavailable"
        for issue in result.issues
    )


def test_validate_reply_blocks_non_compliant_response_shape():
    request = make_request(message="请问产品预计收益多少？")
    plan = make_plan(
        request,
        user_need="refuse expected return request",
        artifact_kind="refusal",
        action_intent="refuse",
        report_scope="none",
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
        report_scope="none",
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


def test_retrieval_guard_blocks_weekly_evidence_for_material_pack_question():
    request = make_request(
        message="材料包里有哪些产品",
        available_strategies=["指增"],
    )
    plan = material_answer_plan(request)
    weekly_fact = EvidenceFact(
        fact_type="report_scope_products",
        value=True,
        source_type="adapter_report_scope",
        source_id="weekly_report",
        resolve_type="weekly_report",
        metadata={"products": [{"product_name": "Product A"}]},
    )

    decision = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[weekly_fact],
        domain_context=DomainContextBuilder().build(request),
    )

    assert decision.outcome == "abstain"
    assert decision.phase == "retrieval_source"
    assert decision.reason_code == "required_evidence_missing"


def test_retrieval_guard_blocks_wrong_channel_evidence():
    request = make_request(available_strategies=["指增"])
    plan = material_answer_plan(request)
    fact = material_product_fact(
        "Product A",
        scope=ArtifactScope(channel_id="adapter_channel:bank:other channel"),
    )

    decision = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[fact],
        domain_context=DomainContextBuilder().build(request),
    )

    assert decision.outcome == "abstain"
    assert decision.reason_code == "channel_scope_mismatch"


def test_retrieval_guard_blocks_wrong_strategy_evidence():
    request = make_request(available_strategies=["指增"])
    plan = material_answer_plan(request).model_copy(update={"selected_strategy": "指增"})
    fact = material_product_fact("Product A", strategy="中证500")

    decision = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[fact],
        domain_context=DomainContextBuilder().build(request),
    )

    assert decision.outcome == "abstain"
    assert decision.reason_code == "strategy_scope_mismatch"


def test_retrieval_guard_blocks_history_when_current_material_artifact_required():
    request = make_request(available_strategies=["指增"])
    plan = material_answer_plan(request)
    fact = material_product_fact(
        "Product A",
        source_type="action_ledger",
        artifact_type="history",
    )

    decision = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[fact],
        domain_context=DomainContextBuilder().build(request),
    )

    assert decision.outcome == "abstain"
    assert decision.reason_code == "history_source_not_current_artifact"


def test_retrieval_guard_allows_valid_material_pack_artifact():
    request = make_request(available_strategies=["指增"])
    plan = material_answer_plan(request)
    fact = material_product_fact("Product A", strategy="指增")

    decision = retrieval_source_guard(
        plan=plan,
        policy=compile_policy(request),
        evidence_facts=[fact],
        domain_context=DomainContextBuilder().build(request),
    )

    assert decision.outcome == "allow"
    assert decision.evidence_seen


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


def test_output_guard_does_not_use_product_keyword_scan_as_final_authority():
    request = make_request(available_strategies=["指增"])
    plan = material_answer_plan(request)
    fact = material_product_fact("Product A")
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="材料包包含：Product B。"),
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
        derive_business_facts([fact], request),
        [fact],
        compile_policy(request),
        domain_context=DomainContextBuilder().build(request),
    )

    assert result.valid is True


def test_runtime_keeps_output_when_only_old_product_keyword_scan_would_fail():
    request = make_request(available_strategies=["指增"])
    plan = material_answer_plan(request)
    fact = material_product_fact("Product A")
    directive = make_directive(
        plan,
        mode="knowledge_answer",
        reply_kind="answer",
        requires_knowledge_composer=True,
        action_intents=[],
    )
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False)
    )

    attempt = runtime._validated_attempt(
        plan=plan,
        plan_validation=PlanValidationResult(valid=True),
        preflight=AdapterPreflightSnapshot.empty(),
        evidence_facts=[fact],
        business_facts=derive_business_facts([fact], request),
        domain_context=DomainContextBuilder().build(
            request,
            available_artifacts=[fact],
        ),
        directive=directive,
        response=ReplyResponse(
            response_id="resp-1",
            reply=PrimaryReply(kind="answer", text="材料包包含：Product B。"),
            actions=[],
        ),
        policy=compile_policy(request),
        guardrail_decisions=[],
    )

    assert attempt.response.reply.kind == "answer"
    assert attempt.response.reply.text == "材料包包含：Product B。"
    assert attempt.response.actions == []
    assert attempt.reply_validation.valid is True


def test_output_guard_requires_composer_to_cite_allowed_evidence_ids():
    request = make_request(available_strategies=["指增"])
    plan = material_answer_plan(request)
    fact = material_product_fact("Product A")
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="材料包包含：Product A。"),
        actions=[],
    )
    composer_output = ComposerReplyOutput(
        response_id="resp-1",
        response_mode="answer",
        claims=["材料包包含 Product A"],
        evidence_ids=[],
        reply=response.reply,
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
        derive_business_facts([fact], request),
        [fact],
        compile_policy(request),
        domain_context=DomainContextBuilder().build(request),
        composer_output=composer_output,
    )

    assert result.valid is False
    assert result.issues[0].code == "unsupported_evidence_claim"
    assert result.issues[0].metadata["claims"] == ["材料包包含 Product A"]


def test_validate_reply_rejects_claim_citing_history_source():
    request = make_request(
        message="材料包里有哪些产品",
        available_strategies=["指增"],
    )
    plan = material_answer_plan(request)
    history_fact = material_product_fact(
        "Old Product",
        source_type="conversation_history",
        artifact_type="material_pack",
    )
    evidence_id = "conversation_history:material_pack:material_pack_product_list"
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="材料包包含：Old Product。"),
        actions=[],
    )
    composer_output = ComposerReplyOutput(
        response_id="resp-1",
        response_mode="answer",
        claims=["材料包包含 Old Product"],
        evidence_ids=[evidence_id],
        reply=response.reply,
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
        derive_business_facts([history_fact], request),
        [history_fact],
        compile_policy(request),
        domain_context=DomainContextBuilder().build(request, available_artifacts=[history_fact]),
        composer_output=composer_output,
    )

    assert result.valid is False
    assert result.issues[0].code == "unsupported_evidence_claim"
    assert result.issues[0].metadata["invalid_evidence_ids"] == [evidence_id]



def test_validate_reply_allows_report_scope_evidence_for_knowledge_answer():
    request = make_request(message="why is A500 missing from this weekly report")
    plan = make_plan(
        request,
        user_need="answer report scope question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        report_scope="none",
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
        report_scope="none",
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
        report_scope="none",
        requested_capabilities=["document_context"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal company question",
        },
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="unable_to_answer", text="当前没有足够证据安全回复。"),
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
        report_scope="none",
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
        report_scope="none",
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
        report_scope="none",
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
