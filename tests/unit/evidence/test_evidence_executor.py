from __future__ import annotations

import asyncio

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.knowledge.approved_knowledge import (
    ApprovedKnowledgeEvidenceService,
    ApprovedKnowledgeSelection,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    AdapterResolveResult,
    ReplyRequest,
)
from tests.helpers.planning import compile_test_plan


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
        "available_artifacts": [
            {"type": "material_pack", "options": ["中证500", "中证1000"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_plan(**overrides) -> ExecutionPlan:
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
    request = make_request()
    return compile_test_plan(request, **payload)


class FakePreflightService:
    def __init__(self):
        self.calls = []

    async def collect(
            self,
            request,
            resolve_types=None,
            resolve_material_pack_options=None,
    ):
        self.calls.append(
            (
                request,
                tuple(resolve_types or []),
                dict(resolve_material_pack_options or {}),
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
                            "available_artifacts": [
                                {"type": "weekly_report"},
                            ],
                            "resolved_at": 1,
                            "resolve_ref": "weekly:ref",
                            "period": "20260529",
                            "report_date": "2026-05-29",
                        }
                    ),
                )
            ]
        )


def test_evidence_executor_runs_preflight_and_derives_business_facts():
    request = make_request()
    fake_preflight = FakePreflightService()
    executor = EvidenceExecutor(fake_preflight)

    result = asyncio.run(
        executor.execute(
            request,
            make_plan(),
            compile_policy(request),
        )
    )

    assert fake_preflight.calls == [
        (request, ("weekly_report", "sales_mention"), {})
    ]
    assert result.preflight.items[0].status == "resolved"
    assert result.evidence_facts[0].fact_type == "weekly_report_resolvable"
    assert result.business_facts.weekly_report.status == "available"
    assert result.business_facts.weekly_report.material_pack_option is None
    assert result.business_facts.requested_material_pack_option_status == "unknown"


def test_evidence_executor_passes_material_pack_option_to_preflight():
    request = make_request(message="这个材料包发一下")
    fake_preflight = FakePreflightService()
    executor = EvidenceExecutor(fake_preflight)
    plan = make_plan(
        artifact_kind="material_pack",
        material_pack_option="中证1000",
    )

    result = asyncio.run(
        executor.execute(
            request,
            plan,
            compile_policy(request),
        )
    )

    assert fake_preflight.calls == [
        (
            request,
            ("material_pack", "sales_mention"),
            {"material_pack": "中证1000"},
        )
    ]
    assert result.business_facts.weekly_report.material_pack_option is None


def test_evidence_executor_allows_plan_without_adapter_resolves():
    request = make_request(message="hi")

    class EmptyPreflightService:
        def __init__(self):
            self.calls = []

        async def collect(
                self,
                request,
                    resolve_types=None,
                resolve_material_pack_options=None,
        ):
            self.calls.append(
                (
                        request,
                        tuple(resolve_types or []),
                        dict(resolve_material_pack_options or {}),
                    )
                )
            return AdapterPreflightSnapshot.empty()

    preflight = EmptyPreflightService()
    executor = EvidenceExecutor(preflight)
    policy = compile_policy(request)
    plan = compile_test_plan(
        request,
        policy=policy,
        user_need="smalltalk greeting",
        artifact_kind="smalltalk",
        action_intent="none",
        compliance={
            "is_compliant": True,
            "reason_code": "unrelated_request",
            "reason": "greeting",
        },
        confidence=0.9,
    )

    result = asyncio.run(
        executor.execute(
            request,
            plan,
            policy,
        )
    )

    assert preflight.calls == [(request, (), {})]
    assert result.preflight == AdapterPreflightSnapshot.empty()
    assert result.evidence_facts == []


def test_evidence_executor_adds_approved_static_knowledge_context():
    request = make_request(
        message="你们有微信公众号吗",
        allowed_read_capabilities=["query_internal_company_info"],
    )
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = compile_test_plan(
        request,
        policy=policy,
        user_need="answer public account question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        evidence_query="衍复投资 微信公众号 二维码",
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal company question",
        },
        confidence=0.9,
    )

    class EmptyPreflightService:
        async def collect(
                self,
                request,
                    resolve_types=None,
                resolve_material_pack_options=None,
        ):
            del request, resolve_types, resolve_material_pack_options
            return AdapterPreflightSnapshot.empty()

    class FakeSelector:
        async def select(self, **kwargs):
            del kwargs
            return ApprovedKnowledgeSelection(
                selected_entry_ids=("company_public_account",),
                selected_image_asset_ids=("company_public_account_qr",),
                confidence="high",
            )

    result = asyncio.run(
        EvidenceExecutor(
            EmptyPreflightService(),
            approved_knowledge_service=ApprovedKnowledgeEvidenceService(
                selector=FakeSelector()
            ),
        ).execute(
            request,
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
                        "artifact": {
                            "type": "weekly_report",
                            "resolve_ref": "weekly:ref",
                            "artifact_ref": "weekly:opaque",
                            "period": "20260529",
                            "report_date": "2026-05-29",
                        },
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
    assert (
        result.business_facts.recent_executed_actions[0].artifact["type"]
        == "weekly_report"
    )
