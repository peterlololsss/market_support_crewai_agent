from __future__ import annotations

from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.evidence import (
    EvidenceFact,
    evidence_facts_from_preflight,
)
from market_support_crewai_agent.runtime.guardrails import (
    apply_reply_guardrails,
    build_deterministic_fallback,
    validate_reply,
)
from market_support_crewai_agent.runtime.planning import ReplyPlan
from market_support_crewai_agent.runtime.policy import compile_policy
from market_support_crewai_agent.schemas import (
    AdapterResolveResult,
    PrimaryReply,
    ReplyMention,
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


def resolve_item(resolve_type: str, status: str, **overrides) -> AdapterPreflightItem:
    payload = {
        "contract_version": "adapter-resolve.v1",
        "resolve_type": resolve_type,
        "status": status,
        "display_name": "测试渠道",
        "reason_code": "ok" if status == "resolved" else "not_resolved",
        "candidates": [],
        "channel_type": "bank",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": [],
        "resolved_at": 1,
    }
    payload.update(overrides)
    return AdapterPreflightItem(
        resolve_type=resolve_type,
        result=AdapterResolveResult.model_validate(payload),
    )


def guard_with_preflight(
        response: ReplyResponse,
        preflight: AdapterPreflightSnapshot,
        request: ReplyRequest | None = None,
) -> ReplyResponse:
    effective_request = request or make_request()
    outcome = apply_reply_guardrails(
        response=response,
        policy=compile_policy(effective_request),
        business_facts=derive_business_facts(
            evidence_facts_from_preflight(preflight),
            effective_request,
        ),
        request=effective_request,
    )
    return outcome.response


def test_validate_reply_reports_unresolved_side_effect_action():
    request = make_request()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report")],
    )

    result = validate_reply(
        response,
        compile_policy(request),
        derive_business_facts([], request),
        request,
    )

    assert result.valid is False
    assert result.issues[0].code == "action_not_resolvable"
    assert result.issues[0].metadata["resolve_type"] == "weekly_report"


def test_apply_reply_guardrails_allows_resolved_side_effect_action():
    request = make_request()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report")],
    )
    facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
        )
    ]

    outcome = apply_reply_guardrails(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
    )

    assert outcome.validation.valid is True
    assert outcome.fallback_used is False
    assert outcome.response == response


def test_validate_reply_blocks_action_not_proposed_by_validated_plan():
    request = make_request(message="请发月报")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "send monthly report",
            "intent": "send_monthly_report",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal monthly report request",
            },
            "required_adapter_resolves": ["monthly_report"],
            "candidate_actions": [{"type": "send_monthly_report", "report_scope": "channel_all"}],
            "confidence": 0.8,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report")],
    )
    facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
        )
    ]

    result = validate_reply(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
        plan=plan,
        evidence_facts=facts,
    )

    assert result.valid is False
    assert result.issues[0].code == "action_not_in_plan_candidate"
    assert result.issues[0].metadata["action_type"] == "send_weekly_report"


def test_validate_reply_blocks_report_action_when_plan_selector_is_unknown():
    request = make_request(message="请发周报")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "send weekly report",
            "intent": "send_weekly_report",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal weekly report request",
            },
            "required_adapter_resolves": ["weekly_report"],
            "candidate_actions": [{"type": "send_weekly_report"}],
            "confidence": 0.8,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
        )
    ]
    business_facts = derive_business_facts(facts, request)

    result = validate_reply(
        response,
        compile_policy(request),
        business_facts,
        request,
        plan=plan,
        evidence_facts=facts,
    )
    fallback = build_deterministic_fallback(
        result,
        response,
        compile_policy(request),
        business_facts,
        request,
    )

    assert result.valid is False
    assert result.issues[0].code == "report_action_selector_missing"
    assert fallback.reply.kind == "clarification"
    assert fallback.actions == []


def test_validate_reply_allows_strategy_scoped_report_action_when_evidence_matches():
    request = make_request(message="请发中证1000周报")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "send weekly report covering 中证1000",
            "intent": "send_weekly_report",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal weekly report request",
            },
            "required_adapter_resolves": ["weekly_report"],
            "candidate_actions": [
                {
                    "type": "send_weekly_report",
                    "report_scope": "strategy",
                    "strategy": "中证1000",
                }
            ],
            "confidence": 0.8,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
            metadata={"status": "resolved", "strategy": "中证1000"},
        ),
        EvidenceFact(
            fact_type="report_contains_strategy",
            value=True,
            resolve_type="weekly_report",
            metadata={"strategy": "中证1000"},
        ),
    ]

    result = validate_reply(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
        plan=plan,
        evidence_facts=facts,
    )

    assert result.valid is True


def test_ambiguous_plan_blocks_side_effect_action_even_when_resolved():
    request = make_request(message="500和1000周报都发一下")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "confirm which report scope should be sent",
            "intent": "clarification",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal report request but ambiguous scope",
            },
            "required_adapter_resolves": ["weekly_report"],
            "candidate_actions": [{"type": "send_weekly_report", "report_scope": "channel_all"}],
            "ambiguity": True,
            "ambiguity_reason": "需要确认发送500、1000，还是渠道周报整包",
            "confidence": 0.7,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
        )
    ]
    business_facts = derive_business_facts(facts, request)

    result = validate_reply(
        response,
        compile_policy(request),
        business_facts,
        request,
        plan=plan,
        evidence_facts=facts,
    )
    fallback = build_deterministic_fallback(
        result,
        response,
        compile_policy(request),
        business_facts,
        request,
    )

    assert result.valid is False
    assert result.issues[0].code == "ambiguous_plan_has_actions"
    assert fallback.reply.kind == "clarification"
    assert "需要确认发送500、1000" in fallback.reply.text
    assert fallback.actions == []


def test_ambiguous_plan_allows_clarification_without_actions():
    request = make_request(message="500和1000周报都发一下")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "clarify report scope",
            "intent": "clarification",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal report request but ambiguous scope",
            },
            "ambiguity": True,
            "ambiguity_reason": "需要确认具体周报范围",
            "confidence": 0.7,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="clarification", text="需要确认具体周报范围。"),
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


def test_pre_execution_success_claim_is_rewritten_without_dropping_valid_action():
    request = make_request(message="请发周报")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "send weekly report",
            "intent": "send_weekly_report",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal weekly report request",
            },
            "required_adapter_resolves": ["weekly_report"],
            "candidate_actions": [{"type": "send_weekly_report", "report_scope": "channel_all"}],
            "confidence": 0.8,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="以上是最新周报链接，请查收。"),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
        )
    ]
    business_facts = derive_business_facts(facts, request)

    result = validate_reply(
        response,
        compile_policy(request),
        business_facts,
        request,
        plan=plan,
        evidence_facts=facts,
    )
    fallback = build_deterministic_fallback(
        result,
        response,
        compile_policy(request),
        business_facts,
        request,
    )

    assert result.valid is False
    assert result.issues[0].code == "pre_execution_success_claim"
    assert fallback.reply.text == ""
    assert fallback.actions == response.actions


def test_side_effect_action_reply_text_is_stripped_without_dropping_action():
    request = make_request(message="请发周报")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "send weekly report",
            "intent": "send_weekly_report",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal weekly report request",
            },
            "required_adapter_resolves": ["weekly_report"],
            "candidate_actions": [{"type": "send_weekly_report", "report_scope": "channel_all"}],
            "confidence": 0.8,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="我来处理。"),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
        )
    ]
    business_facts = derive_business_facts(facts, request)

    result = validate_reply(
        response,
        compile_policy(request),
        business_facts,
        request,
        plan=plan,
        evidence_facts=facts,
    )
    fallback = build_deterministic_fallback(
        result,
        response,
        compile_policy(request),
        business_facts,
        request,
    )

    assert result.valid is False
    assert result.issues[0].code == "side_effect_action_reply_text_not_empty"
    assert fallback.reply.text == ""
    assert fallback.actions == response.actions


def test_sent_claim_without_ledger_evidence_is_blocked():
    request = make_request(message="刚发那个周报怎么没中证1000")
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="刚发的周报是20260529版本。"),
        actions=[],
    )
    business_facts = derive_business_facts([], request)

    result = validate_reply(
        response,
        compile_policy(request),
        business_facts,
        request,
        evidence_facts=[],
    )
    fallback = build_deterministic_fallback(
        result,
        response,
        compile_policy(request),
        business_facts,
        request,
    )

    assert result.valid is False
    assert result.issues[0].code == "sent_claim_without_ledger_evidence"
    assert result.issues[0].metadata["material_type"] == "weekly"
    assert fallback.reply.kind == "clarification"
    assert "已执行的发送记录" in fallback.reply.text
    assert fallback.actions == []


def test_sent_claim_with_matching_executed_weekly_ledger_evidence_is_allowed():
    request = make_request(message="刚发那个周报怎么没中证1000")
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="刚发的周报是20260529版本。"),
        actions=[],
    )
    facts = [
        EvidenceFact(
            fact_type="recent_executed_action",
            value=True,
            source_type="action_ledger",
            metadata={
                "action_type": "send_material",
                "material_type": "weekly",
                "action_id": "act-weekly",
                "version": "20260529",
                "material_ref_available": True,
            },
        )
    ]

    result = validate_reply(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
        evidence_facts=facts,
    )

    assert result.valid is True


def test_sent_claim_blocks_when_executed_ledger_material_type_does_not_match():
    request = make_request(message="刚发那个月报是哪版")
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="刚发的月报是2026年5月版本。"),
        actions=[],
    )
    facts = [
        EvidenceFact(
            fact_type="recent_executed_action",
            value=True,
            source_type="action_ledger",
            metadata={
                "action_type": "send_material",
                "material_type": "weekly",
                "action_id": "act-weekly",
                "version": "20260529",
                "material_ref_available": True,
            },
        )
    ]

    result = validate_reply(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
        evidence_facts=facts,
    )

    assert result.valid is False
    assert result.issues[0].code == "sent_claim_without_ledger_evidence"
    assert result.issues[0].metadata["material_type"] == "monthly"


def test_apply_reply_guardrails_falls_back_for_ambiguous_material_pack():
    request = make_request(available_strategies=["指增", "量化"])
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendMaterialPackAction(type="send_material_pack")],
    )
    facts = [
        EvidenceFact(
            fact_type="material_pack_resolvable",
            value=False,
            resolve_type="material_pack",
            metadata={"status": "ambiguous", "candidates": ["指增", "量化"]},
        )
    ]

    outcome = apply_reply_guardrails(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
    )

    assert outcome.validation.valid is False
    assert outcome.validation.issues[0].code == "material_pack_ambiguous"
    assert outcome.fallback_used is True
    assert outcome.response.reply.kind == "clarification"
    assert "指增" in outcome.response.reply.text
    assert "量化" in outcome.response.reply.text
    assert outcome.response.actions == []


def test_bank_multi_strategy_material_pack_requires_confirmation():
    request = make_request(
        message="发一下材料包",
        channel_type="bank",
        available_strategies=["沪深300指增", "中证1000指增"],
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendMaterialPackAction(type="send_material_pack")],
    )
    facts = [
        EvidenceFact(
            fact_type="material_pack_resolvable",
            value=True,
            resolve_type="material_pack",
            metadata={"status": "resolved", "reason_code": "ok"},
        )
    ]

    outcome = apply_reply_guardrails(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
    )

    assert outcome.validation.valid is False
    assert outcome.validation.issues[0].code == (
        "bank_material_pack_requires_strategy_confirmation"
    )
    assert outcome.response.reply.kind == "clarification"
    assert "沪深300指增" in outcome.response.reply.text
    assert "中证1000指增" in outcome.response.reply.text
    assert outcome.response.actions == []


def test_bank_multi_strategy_material_pack_allows_confirmed_strategy():
    request = make_request(
        message="发一下1000材料包",
        channel_type="bank",
        available_strategies=["沪深300指增", "中证1000指增"],
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[
            SendMaterialPackAction(
                type="send_material_pack",
                strategy="中证1000指增",
            )
        ],
    )
    facts = [
        EvidenceFact(
            fact_type="material_pack_resolvable",
            value=True,
            resolve_type="material_pack",
            metadata={
                "status": "resolved",
                "reason_code": "ok",
                "strategy": "中证1000指增",
            },
        )
    ]

    outcome = apply_reply_guardrails(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
    )

    assert outcome.validation.valid is True
    assert outcome.response.actions == response.actions


def test_non_bank_multi_strategy_material_pack_allows_full_series_action():
    request = make_request(
        message="发一下材料包",
        channel_type="non_bank",
        available_strategies=["沪深300指增", "中证1000指增"],
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendMaterialPackAction(type="send_material_pack")],
    )
    facts = [
        EvidenceFact(
            fact_type="material_pack_resolvable",
            value=True,
            resolve_type="material_pack",
            metadata={"status": "resolved", "reason_code": "ok"},
        )
    ]

    outcome = apply_reply_guardrails(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
    )

    assert outcome.validation.valid is True
    assert outcome.response.actions == response.actions


def test_apply_reply_guardrails_falls_back_for_unsupported_scope_claim():
    request = make_request()
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="这个策略不在周报生成范围内。"),
        actions=[],
    )
    facts = [
        EvidenceFact(
            fact_type="sales_mention_resolvable",
            value=True,
            resolve_type="sales_mention",
        ),
        EvidenceFact(
            fact_type="report_scope_status",
            value="unknown",
            resolve_type="weekly_report",
        ),
    ]

    outcome = apply_reply_guardrails(
        response,
        compile_policy(request),
        derive_business_facts(facts, request),
        request,
    )

    assert outcome.validation.valid is False
    assert outcome.validation.issues[0].code == "unsupported_report_scope_claim"
    assert outcome.response.reply.kind == "human_handoff"
    assert len(outcome.response.reply.mentions) == 1
    assert outcome.response.actions == []


def test_guardrails_block_missing_report_and_mention_sales_when_resolved():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    preflight = AdapterPreflightSnapshot(
        items=[
            resolve_item("weekly_report", "missing"),
            resolve_item("sales_mention", "resolved"),
        ]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded.reply.kind == "human_handoff"
    assert guarded.reply.text.startswith("目前这个渠道下我没有看到可发送的对应材料")
    assert len(guarded.reply.mentions) == 1
    assert guarded.reply.mentions[0].type == "sales"
    assert guarded.actions == []


def test_guardrails_block_action_when_preflight_item_is_missing():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    preflight = AdapterPreflightSnapshot(
        items=[resolve_item("sales_mention", "resolved")]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded.reply.kind == "human_handoff"
    assert guarded.actions == []
    assert len(guarded.reply.mentions) == 1


def test_guardrails_block_weekly_report_action_when_strategy_is_excluded():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    preflight = AdapterPreflightSnapshot(
        items=[
            resolve_item(
                "weekly_report",
                "resolved",
                scope_status="excluded",
                contains_strategy=False,
                strategy="中证1000",
                period="20260529",
            ),
            resolve_item("sales_mention", "resolved"),
        ]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded.reply.kind == "human_handoff"
    assert "周报暂未覆盖中证1000" in guarded.reply.text
    assert len(guarded.reply.mentions) == 1
    assert guarded.actions == []


def test_guardrails_block_monthly_report_action_when_strategy_missing_from_report():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendMonthlyReportAction(type="send_monthly_report", action_id="act-1")],
    )
    preflight = AdapterPreflightSnapshot(
        items=[
            resolve_item(
                "monthly_report",
                "resolved",
                contains_strategy=False,
                scope_status="unknown",
                strategy="小市值",
                period="2026-05",
            ),
        ]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded.reply.kind == "unable_to_answer"
    assert "月报未包含小市值" in guarded.reply.text
    assert guarded.actions == []


def test_guardrails_allow_report_action_when_strategy_scope_is_unknown():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    preflight = AdapterPreflightSnapshot(
        items=[
            resolve_item(
                "weekly_report",
                "resolved",
                contains_strategy=None,
                scope_status="unknown",
                strategy="中证1000",
                period="20260529",
            ),
        ]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded.reply.kind == "answer"
    assert guarded.actions == response.actions


def test_guardrails_block_sales_mention_when_target_missing():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(
            kind="human_handoff",
            text="我帮你请销售同事确认",
            mentions=[ReplyMention(type="sales", reason="确认口径")],
        ),
        actions=[],
    )
    preflight = AdapterPreflightSnapshot(
        items=[resolve_item("sales_mention", "missing")]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded.actions == []
    assert guarded.reply.mentions == []
    assert "暂未配置" in guarded.reply.text


def test_guardrails_block_sales_mention_when_preflight_item_is_missing():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(
            kind="human_handoff",
            text="我帮你请销售同事确认",
            mentions=[ReplyMention(type="sales", reason="确认口径")],
        ),
        actions=[],
    )
    preflight = AdapterPreflightSnapshot(
        items=[resolve_item("weekly_report", "resolved")]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded.actions == []
    assert guarded.reply.mentions == []
    assert "暂未配置" in guarded.reply.text


def test_guardrails_allow_report_scope_claim_with_excluded_evidence():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="这个策略不在周报生成范围内。"),
        actions=[],
    )
    preflight = AdapterPreflightSnapshot(
        items=[resolve_item("weekly_report", "resolved", scope_status="excluded")]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded == response


def test_guardrails_block_report_content_claim_without_contains_strategy_evidence():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="周报不包含这个策略。"),
        actions=[],
    )
    preflight = AdapterPreflightSnapshot(
        items=[resolve_item("weekly_report", "resolved", contains_strategy=None)]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded.response_id == "resp-1"
    assert guarded.reply.kind == "unable_to_answer"
    assert "没有足够的周报范围证据" in guarded.reply.text
    assert guarded.actions == []


def test_guardrails_allow_report_content_claim_with_negative_evidence():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="周报不包含这个策略。"),
        actions=[],
    )
    preflight = AdapterPreflightSnapshot(
        items=[resolve_item("weekly_report", "resolved", contains_strategy=False)]
    )

    guarded = guard_with_preflight(response, preflight)

    assert guarded == response


def test_guardrails_block_action_without_preflight_evidence():
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendMonthlyReportAction(type="send_monthly_report", action_id="act-1")],
    )

    guarded = guard_with_preflight(response, AdapterPreflightSnapshot.empty())

    assert guarded.reply.kind == "unable_to_answer"
    assert guarded.actions == []


def test_validate_reply_blocks_knowledge_answer_without_document_evidence():
    request = make_request(message="介绍一下中证1000指增")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "answer product knowledge question",
            "intent": "knowledge_qa",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal product question",
            },
            "confidence": 0.8,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="中证1000指增是指数增强策略。"),
        actions=[],
    )

    result = validate_reply(
        response,
        compile_policy(request, doc_mcp_enabled=True),
        derive_business_facts([], request),
        request,
        plan=plan,
        evidence_facts=[],
    )

    assert result.valid is False
    assert result.issues[0].code == "knowledge_answer_without_document_evidence"


def test_validate_reply_allows_knowledge_answer_with_document_evidence():
    request = make_request(message="介绍一下中证1000指增")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "answer product knowledge question",
            "intent": "knowledge_qa",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal product question",
            },
            "confidence": 0.8,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="中证1000指增是指数增强策略。"),
        actions=[],
    )
    evidence_facts = [
        EvidenceFact(
            fact_type="document_context",
            value="Q：介绍一下中证1000指增\nA：中证1000指增是指数增强策略。",
            source_type="document_mcp",
            source_id="衍复中证1000指数增强策略",
        )
    ]

    result = validate_reply(
        response,
        compile_policy(request, doc_mcp_enabled=True),
        derive_business_facts(evidence_facts, request),
        request,
        plan=plan,
        evidence_facts=evidence_facts,
    )

    assert result.valid is True


def test_validate_reply_does_not_treat_document_unavailable_as_grounding():
    request = make_request(message="介绍一下中证1000指增")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "answer product knowledge question",
            "intent": "knowledge_qa",
            "compliance": {
                "is_compliant": True,
                "reason_code": "compliant_product_request",
                "reason": "normal product question",
            },
            "confidence": 0.8,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="中证1000指增是指数增强策略。"),
        actions=[],
    )
    evidence_facts = [
        EvidenceFact(
            fact_type="document_context_unavailable",
            value=False,
            source_type="document_mcp",
            source_id="document_mcp",
            metadata={"reason_code": "document_mcp_error"},
        )
    ]

    result = validate_reply(
        response,
        compile_policy(request, doc_mcp_enabled=True),
        derive_business_facts(evidence_facts, request),
        request,
        plan=plan,
        evidence_facts=evidence_facts,
    )

    assert result.valid is False
    assert result.issues[0].code == "knowledge_answer_without_document_evidence"


def test_validate_reply_blocks_non_compliant_answer_and_actions():
    request = make_request(message="请问产品预计收益多少？")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "refuse expected return request",
            "intent": "refusal",
            "compliance": {
                "is_compliant": False,
                "reason_code": "expected_or_target_return",
                "reason": "expected return requests must be refused",
            },
            "confidence": 0.9,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text="预计收益可以看周报。"),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    evidence_facts = [
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
        )
    ]
    business_facts = derive_business_facts(evidence_facts, request)

    result = validate_reply(
        response,
        compile_policy(request),
        business_facts,
        request,
        plan=plan,
        evidence_facts=evidence_facts,
    )
    fallback = build_deterministic_fallback(
        result,
        response,
        compile_policy(request),
        business_facts,
        request,
    )

    assert result.valid is False
    assert result.issues[0].code == "non_compliant_reply_has_actions"
    assert result.repairable is False
    assert fallback.reply.kind == "unable_to_answer"
    assert "不设置预计收益" in fallback.reply.text
    assert fallback.actions == []


def test_validate_reply_blocks_non_compliant_sales_mentions_even_when_resolved():
    request = make_request(message="加你微信了，通过一下")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "refuse private contact request",
            "intent": "refusal",
            "compliance": {
                "is_compliant": False,
                "reason_code": "private_contact_request",
                "reason": "private contact requests are not allowed",
            },
            "confidence": 0.9,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(
            kind="human_handoff",
            text="我帮你请销售同事处理。",
            mentions=[ReplyMention(type="sales", reason="private contact request")],
        ),
        actions=[],
    )
    evidence_facts = [
        EvidenceFact(
            fact_type="sales_mention_resolvable",
            value=True,
            resolve_type="sales_mention",
        )
    ]

    result = validate_reply(
        response,
        compile_policy(request),
        derive_business_facts(evidence_facts, request),
        request,
        plan=plan,
        evidence_facts=evidence_facts,
    )

    assert result.valid is False
    assert result.issues[0].code == "non_compliant_reply_has_mentions"
    assert result.repairable is False


def test_validate_reply_replaces_non_compliant_custom_text_with_safe_fallback():
    request = make_request(message="产品是否保本？")
    plan = ReplyPlan.model_validate(
        {
            "user_need": "refuse principal guarantee request",
            "intent": "refusal",
            "compliance": {
                "is_compliant": False,
                "reason_code": "principal_or_risk_guarantee",
                "reason": "principal guarantee requests must be refused",
            },
            "confidence": 0.9,
        }
    )
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="unable_to_answer", text="这个不能展开。"),
        actions=[],
    )
    business_facts = derive_business_facts([], request)

    result = validate_reply(
        response,
        compile_policy(request),
        business_facts,
        request,
        plan=plan,
        evidence_facts=[],
    )
    fallback = build_deterministic_fallback(
        result,
        response,
        compile_policy(request),
        business_facts,
        request,
    )

    assert result.valid is False
    assert result.issues[0].code == "non_compliant_reply_text"
    assert result.repairable is False
    assert fallback.reply.kind == "unable_to_answer"
    assert "不承诺保本" in fallback.reply.text
    assert fallback.actions == []
