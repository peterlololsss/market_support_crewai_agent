from __future__ import annotations

import asyncio
from types import SimpleNamespace

from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.state.audit import AuditStore, build_audit_trace
from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import evidence_facts_from_preflight
from market_support_crewai_agent.runtime.validation.guardrails import validate_reply
from market_support_crewai_agent.runtime.domain.planning import (
    IntentFrame,
    compile_intent_frame,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.schemas import (
    AdapterResolveResult,
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.settings import Settings


def make_request(message: str = "请发一下周报", **overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": message,
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


def make_frame(**overrides) -> IntentFrame:
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
    return IntentFrame.model_validate(payload)


def resolve_item(resolve_type: str, status: str, **overrides) -> AdapterPreflightItem:
    payload = {
        "contract_version": "adapter-resolve",
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
    if status == "resolved":
        payload["resolve_ref"] = f"{resolve_type}:ref"
    payload.update(overrides)
    return AdapterPreflightItem(
        resolve_type=resolve_type,
        result=AdapterResolveResult.model_validate(payload),
    )


class FakePlannerAgent:
    def __init__(self, frame: IntentFrame):
        self.frame = frame

    async def kickoff_async(self, prompt, response_format):
        return SimpleNamespace(pydantic=self.frame, raw="")


class ResolvedWeeklyPreflight:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_strategies=None,
    ):
        del request, canonical_context, resolve_types, resolve_strategies
        return AdapterPreflightSnapshot(
            items=[
                resolve_item(
                    "weekly_report",
                    "resolved",
                    period="20260529",
                    report_date="2026-05-29",
                    scope_status="included",
                    contains_strategy=True,
                    resolve_ref="weekly:ref",
                )
            ]
        )


def test_build_audit_trace_records_adapter_safe_runtime_state():
    request = make_request()
    policy = compile_policy(request)
    plan = compile_intent_frame(
        make_frame(),
        request,
        canonicalize_request(request),
        policy,
    )
    plan_validation = validate_execution_plan(plan, policy)
    preflight = AdapterPreflightSnapshot(
        items=[
            resolve_item(
                "weekly_report",
                "resolved",
                period="20260529",
                report_date="2026-05-29",
                scope_status="included",
                contains_strategy=True,
                resolve_ref="weekly:ref",
            )
        ]
    )
    evidence_facts = evidence_facts_from_preflight(preflight)
    business_facts = derive_business_facts(evidence_facts, request)
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[
            SendWeeklyReportAction(
                type="send_weekly_report",
                action_id="act-1",
                resolve_type="weekly_report",
                resolve_ref="weekly:ref",
                report_scope="channel_all",
                strategy=None,
                period="20260529",
                report_date="2026-05-29",
            )
        ],
    )
    directive = ResponseDirective(
        mode="action",
        reply_kind="answer",
        action_intents=plan.action_intents,
        reason_code="action_ready",
    )
    reply_validation = validate_reply(
        response,
        directive,
        plan,
        business_facts,
        evidence_facts,
        policy,
    )

    trace = build_audit_trace(
        request=request,
        settings=Settings(llm_api_key="test-key"),
        policy=policy,
        plan=plan,
        directive=directive,
        plan_validation=plan_validation,
        action_history=[],
        preflight=preflight,
        evidence_facts=evidence_facts,
        business_facts=business_facts,
        response=response,
        reply_validation=reply_validation,
    )

    assert trace.contract_version == "audit-trace"
    assert trace.context_id == "msg-1"
    assert len(trace.policy_hash) == 16
    assert trace.plan_validation["valid"] is True
    assert trace.response_directive["mode"] == "action"
    assert trace.canonical_entities["selected_strategy"] == "指增"
    assert trace.reply_validation is not None
    assert trace.reply_validation["valid"] is True
    assert trace.business_facts["weekly_report"]["status"] == "available"
    assert trace.business_facts["weekly_report"]["resolve_ref_available"] is True
    assert trace.llm_executions == []
    assert trace.action_preconditions == [
        {
            "action_index": 1,
            "action_id": "act-1",
            "action_type": "send_weekly_report",
            "resolve_type": "weekly_report",
            "resolve_status": "resolved",
            "plan_report_scope": "channel_all",
            "plan_strategy": None,
            "action_strategy": None,
            "adapter_strategy": None,
            "adapter_ref_available": True,
            "action_ref_available": True,
            "contains_strategy": True,
            "scope_status": "included",
            "period": "20260529",
            "report_date": "2026-05-29",
        }
    ]
    assert "weekly:ref" not in str(trace.to_dict())


def test_runtime_records_trace_for_deterministic_action_response():
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_frame())  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(make_request()))

    trace = audit_store.latest()
    assert response.actions[0].type == "send_weekly_report"
    assert response.actions[0].resolve_ref == "weekly:ref"
    assert trace is not None
    assert trace.reply_validation is not None
    assert trace.reply_validation["valid"] is True
    assert trace.adapter_execution_status == "pending_adapter_execution"
    assert trace.llm_executions[0]["response_format"] == "IntentFrame"
    assert trace.llm_executions[0]["stage"] == "planner_intent"
    assert trace.llm_executions[0]["prompt_profile_id"] == "planner_intent.ds_v4pro"
    assert trace.llm_executions[0]["prompt_hash"].startswith("sha256:")
    assert "planner.intent_taxonomy" in trace.llm_executions[0]["prompt_fragment_ids"]
    assert trace.versions["prompt_profile_ids"] == ["planner_intent.ds_v4pro"]
    assert all(item["stage"] != "knowledge_composer" for item in trace.llm_executions)
