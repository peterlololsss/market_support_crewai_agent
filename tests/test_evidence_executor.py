from __future__ import annotations

import asyncio

from market_support_crewai_agent.runtime.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.evidence_executor import EvidenceExecutor
from market_support_crewai_agent.runtime.planning import (
    ExecutionPlan,
    IntentFrame,
    compile_intent_frame,
)
from market_support_crewai_agent.runtime.policy import compile_policy
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    AdapterResolveResult,
    ReplyRequest,
)


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "1000所有号的周报我想看看",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_plan(**overrides) -> ExecutionPlan:
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
    request = make_request()
    return compile_intent_frame(
        IntentFrame.model_validate(payload),
        request,
        canonicalize_request(request),
        compile_policy(request),
    )


class FakePreflightService:
    def __init__(self):
        self.calls = []

    async def collect(
            self,
            request,
            canonical_context=None,
            resolve_types=None,
            resolve_strategies=None,
    ):
        self.calls.append(
            (
                request,
                canonical_context,
                tuple(resolve_types or []),
                dict(resolve_strategies or {}),
            )
        )
        return AdapterPreflightSnapshot(
            items=[
                AdapterPreflightItem(
                    resolve_type="weekly_report",
                    result=AdapterResolveResult.model_validate(
                        {
                            "contract_version": "adapter-resolve",
                            "resolve_type": "weekly_report",
                            "status": "resolved",
                            "display_name": request.dist_channel_name,
                            "reason_code": "ok",
                            "candidates": [],
                            "channel_type": "bank",
                            "available_materials": ["weekly"],
                            "available_strategies": ["中证500", "中证1000"],
                            "resolved_at": 1,
                            "resolve_ref": "weekly:ref",
                            "strategy": (resolve_strategies or {}).get("weekly_report"),
                            "period": "20260529",
                            "report_date": "2026-05-29",
                            "contains_strategy": True,
                            "scope_status": "included",
                        }
                    ),
                )
            ]
        )


def test_evidence_executor_runs_preflight_and_derives_business_facts():
    request = make_request()
    canonical_context = canonicalize_request(request)
    fake_preflight = FakePreflightService()
    executor = EvidenceExecutor(fake_preflight)

    result = asyncio.run(
        executor.execute(
            request,
            canonical_context,
            make_plan(),
            compile_policy(request),
        )
    )

    assert fake_preflight.calls == [
        (request, canonical_context, ("weekly_report", "sales_mention"), {})
    ]
    assert result.preflight.items[0].status == "resolved"
    assert result.evidence_facts[0].fact_type == "weekly_report_resolvable"
    assert result.business_facts.weekly_report.status == "available"
    assert result.business_facts.weekly_report.strategy is None
    assert result.business_facts.requested_strategy_status == "available"


def test_evidence_executor_passes_plan_strategy_selector_to_preflight():
    request = make_request(message="这个报告发一下")
    canonical_context = canonicalize_request(request)
    fake_preflight = FakePreflightService()
    executor = EvidenceExecutor(fake_preflight)
    plan = make_plan(
        report_scope="strategy",
        selected_strategy="中证1000",
    )

    result = asyncio.run(
        executor.execute(
            request,
            canonical_context,
            plan,
            compile_policy(request),
        )
    )

    assert fake_preflight.calls == [
        (
            request,
            canonical_context,
            ("weekly_report", "sales_mention"),
            {"weekly_report": "中证1000"},
        )
    ]
    assert result.business_facts.weekly_report.strategy == "中证1000"


def test_evidence_executor_allows_plan_without_adapter_resolves():
    request = make_request(message="hi")
    canonical_context = canonicalize_request(request)

    class EmptyPreflightService:
        def __init__(self):
            self.calls = []

        async def collect(
                self,
                request,
                canonical_context=None,
                resolve_types=None,
                resolve_strategies=None,
        ):
            self.calls.append(
                (
                    request,
                    canonical_context,
                    tuple(resolve_types or []),
                    dict(resolve_strategies or {}),
                )
            )
            return AdapterPreflightSnapshot.empty()

    preflight = EmptyPreflightService()
    executor = EvidenceExecutor(preflight)
    policy = compile_policy(request)
    plan = compile_intent_frame(
        IntentFrame.model_validate(
            {
                "user_need": "smalltalk greeting",
                "artifact_kind": "smalltalk",
                "action_intent": "none",
                "compliance": {
                    "is_compliant": True,
                    "reason_code": "unrelated_request",
                    "reason": "greeting",
                },
                "confidence": 0.9,
            }
        ),
        request,
        canonical_context,
        policy,
    )

    result = asyncio.run(
        executor.execute(
            request,
            canonical_context,
            plan,
            policy,
        )
    )

    assert preflight.calls == [(request, canonical_context, (), {})]
    assert result.preflight == AdapterPreflightSnapshot.empty()
    assert result.evidence_facts == []


def test_evidence_executor_adds_approved_static_knowledge_context():
    request = make_request(
        message="你们有微信公众号吗",
        allowed_read_capabilities=["query_internal_company_info"],
    )
    canonical_context = canonicalize_request(request)
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = compile_intent_frame(
        IntentFrame.model_validate(
            {
                "user_need": "answer public account question",
                "artifact_kind": "knowledge_answer",
                "action_intent": "answer",
                "requested_capabilities": ["document_context"],
                "evidence_query": "衍复投资 微信公众号 二维码",
                "compliance": {
                    "is_compliant": True,
                    "reason_code": "compliant_product_request",
                    "reason": "normal company question",
                },
                "confidence": 0.9,
            }
        ),
        request,
        canonical_context,
        policy,
    )

    class EmptyPreflightService:
        async def collect(
                self,
                request,
                canonical_context=None,
                resolve_types=None,
                resolve_strategies=None,
        ):
            del request, canonical_context, resolve_types, resolve_strategies
            return AdapterPreflightSnapshot.empty()

    result = asyncio.run(
        EvidenceExecutor(EmptyPreflightService()).execute(
            request,
            canonical_context,
            plan,
            policy,
        )
    )

    static_facts = [
        fact
        for fact in result.evidence_facts
        if fact.source_type == "approved_static_knowledge"
    ]
    assert static_facts
    assert static_facts[0].fact_type == "document_context"
    assert "%%comp_wx_qr_code.png%%" in str(static_facts[0].value)
    assert static_facts[0].metadata["content_is_data_only"] is True


def test_evidence_executor_merges_recent_executed_action_facts():
    request = make_request()
    canonical_context = canonicalize_request(request)
    ledger = ActionLedger()
    ledger.record_feedback(
        ActionFeedbackRequest.model_validate(
            {
                "conversation_key": request.conversation_key,
                "group_id": request.group_id,
                "sender_id": request.sender_id,
                "context_id": "msg-previous",
                "response_id": "resp-previous",
                "executions": [
                    {
                        "action_type": "send_weekly_report",
                        "status": "executed",
                        "action_id": "act-previous-weekly",
                        "resolve_ref": "weekly:ref",
                        "material_type": "weekly",
                        "strategy": "中证1000",
                        "material_id": "weekly:opaque",
                        "version": "20260529",
                    }
                ],
            }
        )
    )
    fake_preflight = FakePreflightService()
    executor = EvidenceExecutor(fake_preflight)

    result = asyncio.run(
        executor.execute(
            request,
            canonical_context,
            make_plan(),
            compile_policy(request),
            action_history=ledger.recent_executed_for_conversation(
                request.conversation_key,
            ),
        )
    )

    assert any(
        fact.fact_type == "recent_executed_action"
        and fact.source_type == "action_ledger"
        for fact in result.evidence_facts
    )
    assert result.business_facts.recent_executed_actions[0].action_id == (
        "act-previous-weekly"
    )
    assert result.business_facts.recent_executed_actions[0].material_type == "weekly"
