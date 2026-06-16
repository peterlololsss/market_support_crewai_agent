from __future__ import annotations

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.compliance_policy import refusal_text_for_reason
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.guardrails import (
    remove_pre_execution_send_claims,
    validate_reply,
)
from market_support_crewai_agent.runtime.domain.planning import (
    IntentFrame,
    compile_intent_frame,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.schemas import (
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
    SendMaterialPackAction,
    SendMonthlyReportAction,
    SendWeeklyReportAction,
)


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "hello",
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


def make_plan(request: ReplyRequest | None = None, **overrides):
    request = request or make_request()
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
    return compile_intent_frame(
        IntentFrame.model_validate(payload),
        request,
        canonicalize_request(request),
        compile_policy(request, doc_mcp_enabled=True),
    )


def weekly_action(**overrides) -> SendWeeklyReportAction:
    payload = {
        "type": "send_weekly_report",
        "action_id": "act-1",
        "resolve_type": "weekly_report",
        "resolve_ref": "weekly:ref",
        "report_scope": "channel_all",
        "strategy": None,
        "period": "20260529",
        "report_date": "2026-05-29",
    }
    payload.update(overrides)
    return SendWeeklyReportAction.model_validate(payload)


def monthly_action(**overrides) -> SendMonthlyReportAction:
    payload = {
        "type": "send_monthly_report",
        "action_id": "act-1",
        "resolve_type": "monthly_report",
        "resolve_ref": "monthly:ref",
        "report_scope": "channel_all",
        "strategy": None,
        "period": "202605",
        "report_date": "2026-05-31",
    }
    payload.update(overrides)
    return SendMonthlyReportAction.model_validate(payload)


def material_action(**overrides) -> SendMaterialPackAction:
    payload = {
        "type": "send_material_pack",
        "action_id": "act-1",
        "resolve_type": "material_pack",
        "resolve_ref": "material:ref",
    }
    payload.update(overrides)
    return SendMaterialPackAction.model_validate(payload)


def make_directive(plan, **overrides) -> ResponseDirective:
    payload = {
        "mode": plan.response_mode,
        "reply_kind": "answer" if plan.response_mode == "action" else "unable_to_answer",
        "text": "",
        "action_intents": plan.action_intents if plan.response_mode == "action" else [],
    }
    payload.update(overrides)
    return ResponseDirective.model_validate(payload)


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


def test_validate_reply_blocks_pre_execution_success_claim():
    request = make_request(message="请发周报")
    plan = make_plan()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="周报已发送，请查收。"),
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
    assert any(issue.code == "sent_claim_without_ledger_evidence" for issue in result.issues)


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
