from __future__ import annotations

import asyncio
from types import SimpleNamespace

from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.state.audit import AuditStore, build_audit_trace
from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import evidence_facts_from_preflight
from market_support_crewai_agent.runtime.validation.reply_validator import validate_reply
from market_support_crewai_agent.runtime.domain.planning import (
    compile_plan_spec,
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
from tests.helpers.planning import make_plan_spec


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
        "available_artifacts": [
            {"type": "material_pack", "options": ["指增"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_weekly_plan_spec(**overrides):
    request = overrides.pop("request", None)
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
    return make_plan_spec(request=request, **payload)


def resolve_item(resolve_type: str, status: str, **overrides) -> AdapterPreflightItem:
    payload = {
        "contract_version": "adapter-resolve",
        "resolve_type": resolve_type,
        "status": status,
        "display_name": "测试渠道",
        "reason_code": "ok" if status == "resolved" else "not_resolved",
        "candidates": [],
        "channel_type": "bank",
        "available_artifacts": [
            {"type": "material_pack", "options": []},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
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
    def __init__(self, plan_spec):
        self.plan_spec = plan_spec

    async def kickoff_async(self, prompt, response_format):
        return SimpleNamespace(pydantic=self.plan_spec, raw="")


class ResolvedWeeklyPreflight:
    async def collect(
        self,
        request,
        resolve_types=None,
        resolve_material_pack_options=None,
    ):
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot(
            items=[
                resolve_item(
                    "weekly_report",
                    "resolved",
                    period="20260529",
                    report_date="2026-05-29",
                    resolve_ref="weekly:ref",
                )
            ]
        )


def test_build_audit_trace_records_adapter_safe_runtime_state():
    request = make_request()
    policy = compile_policy(request)
    plan = compile_plan_spec(
        make_weekly_plan_spec(request=request),
        request,
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
    assert trace.policy_scope["material_pack_options"] == ["指增"]
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
            "plan_material_pack_option": None,
            "action_material_pack_option": None,
            "adapter_material_pack_option": None,
            "adapter_ref_available": True,
            "action_ref_available": True,
            "period": "20260529",
            "report_date": "2026-05-29",
        }
    ]
    assert "weekly:ref" not in str(trace.to_dict())


def test_runtime_records_trace_for_deterministic_action_response():
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_weekly_plan_spec())  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(make_request()))

    trace = audit_store.latest()
    assert response.actions[0].type == "send_weekly_report"
    assert response.actions[0].resolve_ref == "weekly:ref"
    assert trace is not None
    assert trace.reply_validation is not None
    assert trace.reply_validation["valid"] is True
    assert trace.adapter_execution_status == "pending_adapter_execution"
    assert trace.llm_executions == []
    assert trace.versions["prompt_profile_ids"] == []
    assert trace.versions["runtime_trace"] == "runtime-trace-v1"
    assert trace.runtime_trace["schema_version"] == "runtime-trace-v1"
    trace_event_names = [event["name"] for event in trace.runtime_trace["events"]]
    assert "candidate.build" in trace_event_names
    assert all(item["stage"] != "knowledge_composer" for item in trace.llm_executions)
