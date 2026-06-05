from __future__ import annotations

import asyncio

from market_support_crewai_agent.runtime.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.evidence_executor import EvidenceExecutor
from market_support_crewai_agent.runtime.planning import ReplyPlan
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


def make_plan(**overrides) -> ReplyPlan:
    payload = {
        "user_need": "send weekly report",
        "intent": "send_weekly_report",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report request",
        },
        "evidence_requests": [
            {
                "capability": "resolve_weekly_report",
                "reason": "confirm weekly report can be sent",
            }
        ],
        "business_checks": [
            {
                "check": "check_weekly_report_resolvable",
                "reason": "weekly report send requires adapter resolve",
            }
        ],
        "required_adapter_resolves": ["weekly_report"],
        "candidate_actions": [{"type": "send_weekly_report", "report_scope": "channel_all"}],
        "confidence": 0.8,
    }
    payload.update(overrides)
    return ReplyPlan.model_validate(payload)


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
                            "contract_version": "adapter-resolve.v1",
                            "resolve_type": "weekly_report",
                            "status": "resolved",
                            "display_name": request.dist_channel_name,
                            "reason_code": "ok",
                            "candidates": [],
                            "channel_type": "bank",
                            "available_materials": ["weekly"],
                            "available_strategies": ["中证500", "中证1000"],
                            "resolved_at": 1,
                            "strategy": (resolve_strategies or {}).get(
                                "weekly_report",
                                canonical_context.selected_strategy,
                            ),
                            "period": "20260529",
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
    assert result.business_facts.weekly_report.strategy == "中证1000"
    assert result.business_facts.requested_strategy_status == "available"


def test_evidence_executor_passes_plan_strategy_selector_to_preflight():
    request = make_request(message="这个报告发一下")
    canonical_context = canonicalize_request(request)
    fake_preflight = FakePreflightService()
    executor = EvidenceExecutor(fake_preflight)
    plan = make_plan(
        evidence_requests=[
            {
                "capability": "resolve_weekly_report",
                "reason": "confirm weekly report can be sent",
                "strategy": "中证1000",
            }
        ],
        candidate_actions=[
            {
                "type": "send_weekly_report",
                "report_scope": "strategy",
                "strategy": "中证1000",
            }
        ],
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
                        "action_type": "send_material",
                        "status": "executed",
                        "action_id": "act-previous-weekly",
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
