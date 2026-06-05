from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
    NoopAdapterPreflightService,
)
from market_support_crewai_agent.runtime.audit import (
    AuditStore,
    build_audit_trace,
)
from market_support_crewai_agent.runtime.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.evidence import evidence_facts_from_preflight
from market_support_crewai_agent.runtime.guardrails import validate_reply
from market_support_crewai_agent.runtime.planning import ReplyPlan, validate_plan
from market_support_crewai_agent.runtime.policy import compile_policy
from market_support_crewai_agent.runtime.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.schemas import (
    AdapterResolveResult,
    PrimaryReply,
    ReplyMention,
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


class FakePlannerAgent:
    def __init__(self, plan: ReplyPlan):
        self.plan = plan

    async def kickoff_async(self, prompt, response_format):
        return SimpleNamespace(pydantic=self.plan, raw="")


class SequencedPlannerAgent:
    def __init__(self, plans: list[ReplyPlan]):
        self.plans = plans
        self.prompts: list[str] = []

    async def kickoff_async(self, prompt, response_format):
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.plans) - 1)
        return SimpleNamespace(pydantic=self.plans[index], raw="")


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
                    scope_status="included",
                    contains_strategy=True,
                    card_ref="wecom-adapter:hidden",
                )
            ]
        )


class MissingWeeklyPreflight:
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
                    "missing",
                    reason_code="weekly_report_unavailable",
                ),
                resolve_item("sales_mention", "resolved"),
            ]
        )


def test_build_audit_trace_records_adapter_safe_runtime_state():
    request = make_request()
    policy = compile_policy(request)
    plan = make_plan()
    plan_validation = validate_plan(plan, policy)
    preflight = AdapterPreflightSnapshot(
        items=[
            resolve_item(
                "weekly_report",
                "resolved",
                period="20260529",
                scope_status="included",
                contains_strategy=True,
                card_ref="wecom-adapter:hidden",
            )
        ]
    )
    evidence_facts = evidence_facts_from_preflight(preflight)
    business_facts = derive_business_facts(evidence_facts, request)
    response = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
    )
    reply_validation = validate_reply(response, policy, business_facts, request)

    trace = build_audit_trace(
        request=request,
        settings=Settings(llm_api_key="test-key"),
        policy=policy,
        plan=plan,
        plan_validation=plan_validation,
        action_history=[],
        preflight=preflight,
        evidence_facts=evidence_facts,
        business_facts=business_facts,
        response=response,
        reply_validation=reply_validation,
        fallback_used=False,
    )

    assert trace.contract_version == "audit-trace.v1"
    assert trace.context_id == "msg-1"
    assert len(trace.policy_hash) == 16
    assert trace.plan_validation["valid"] is True
    assert trace.canonical_entities["selected_strategy"] == "指增"
    assert trace.canonical_entities["strategy_status"] == "resolved"
    assert trace.reply_validation is not None
    assert trace.reply_validation["valid"] is True
    assert trace.business_facts["weekly_report"]["status"] == "available"
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
            "contains_strategy": True,
            "scope_status": "included",
            "period": "20260529",
            "report_date": None,
        }
    ]
    assert trace.adapter_execution_status == "pending_adapter_execution"
    assert trace.versions["planner_prompt"] == "planner-prompt.v1"
    assert "wecom-adapter:hidden" not in json.dumps(trace.to_dict(), ensure_ascii=False)


def test_runtime_records_crewai_usage_metadata_in_audit_trace():
    request = make_request()
    audit_store = AuditStore()

    class MetadataPlannerAgent:
        async def kickoff_async(self, prompt, response_format):
            del prompt, response_format
            return SimpleNamespace(
                pydantic=make_plan(),
                raw='{"user_need": "send weekly report"}',
                agent_role="Market Support Reply Planner",
                usage_metrics={
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                    "api_key": "should-not-appear",
                },
                plan="internal CrewAI plan text should not be logged",
                todos=[SimpleNamespace(status="completed")],
                replan_count=1,
                last_replan_reason="initial plan validated",
                messages=[{"role": "system", "content": "hidden"}],
            )

    class MetadataComposerAgent:
        async def kickoff_async(self, prompt, response_format):
            del prompt, response_format
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-agent",
                    reply=PrimaryReply(kind="answer", text=""),
                    actions=[
                        SendWeeklyReportAction(
                            type="send_weekly_report",
                            action_id="act-weekly",
                        )
                    ],
                ),
                raw='{"reply": {"kind": "answer"}}',
                agent_role="Market Support Reply Composer",
                usage_metrics={"total_tokens": 9},
                todos=[SimpleNamespace(status="failed")],
            )

    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: MetadataPlannerAgent()  # type: ignore[method-assign]
    runtime._build_agent = lambda: MetadataComposerAgent()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))
    trace = audit_store.latest()

    assert response.actions[0].type == "send_weekly_report"
    assert trace is not None
    assert [item["stage"] for item in trace.llm_executions] == [
        "planner",
        "composer",
    ]
    planner_execution = trace.llm_executions[0]
    composer_execution = trace.llm_executions[1]
    assert planner_execution["agent_role"] == "Market Support Reply Planner"
    assert planner_execution["response_format"] == "ReplyPlan"
    assert planner_execution["usage_metrics"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert planner_execution["pydantic_type"] == "ReplyPlan"
    assert planner_execution["raw_length"] == 35
    assert planner_execution["plan_present"] is True
    assert planner_execution["todo_count"] == 1
    assert planner_execution["completed_todo_count"] == 1
    assert planner_execution["failed_todo_count"] == 0
    assert planner_execution["replan_count"] == 1
    assert planner_execution["last_replan_reason"] == "initial plan validated"
    assert composer_execution["response_format"] == "ReplyResponse"
    assert composer_execution["usage_metrics"] == {"total_tokens": 9}
    assert composer_execution["failed_todo_count"] == 1
    trace_json = json.dumps(trace.to_dict(), ensure_ascii=False)
    assert "should-not-appear" not in trace_json
    assert "internal CrewAI plan text should not be logged" not in trace_json
    assert "hidden" not in trace_json


def test_runtime_records_audit_trace_for_reply_guardrail_fallback():
    request = make_request()
    audit_store = AuditStore()

    class FakePreflight:
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
                        "missing",
                        reason_code="weekly_report_unavailable",
                    ),
                    resolve_item("sales_mention", "resolved"),
                ]
            )

    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=FakePreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_plan())  # type: ignore[method-assign]

    class FakeComposerAgent:
        async def kickoff_async(self, prompt, response_format):
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-agent",
                    reply=PrimaryReply(kind="answer", text=""),
                    actions=[
                        SendWeeklyReportAction(
                            type="send_weekly_report",
                            action_id="act-weekly",
                        )
                    ],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeComposerAgent()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))
    trace = audit_store.latest()

    assert response.reply.kind == "human_handoff"
    assert trace is not None
    assert audit_store.count() == 1
    assert trace.fallback_used is True
    assert trace.fallback_reason == "action_not_resolvable"
    assert trace.reply_validation is not None
    assert trace.reply_validation["issues"][0]["code"] == "action_not_resolvable"
    assert trace.business_facts["weekly_report"]["status"] == "unavailable"
    assert trace.business_facts["sales_mention"]["status"] == "available"
    assert trace.final_actions == []
    assert trace.adapter_execution_status == "no_actions"


def test_runtime_records_audit_trace_for_invalid_plan_early_fallback():
    request = make_request()
    audit_store = AuditStore()
    invalid_plan = make_plan(
        compliance={
            "is_compliant": None,
            "reason_code": "unknown",
            "reason": "not enough context",
        }
    )
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=NoopAdapterPreflightService(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(invalid_plan)  # type: ignore[method-assign]

    class ComposerShouldNotRun:
        async def kickoff_async(self, prompt, response_format):
            raise AssertionError("composer should not run when plan is invalid")

    runtime._build_agent = lambda: ComposerShouldNotRun()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))
    trace = audit_store.latest()

    assert response.reply.kind == "no_reply"
    assert trace is not None
    assert trace.fallback_used is True
    assert trace.fallback_reason == "plan_or_contract_fallback"
    assert trace.reply_validation is None
    assert trace.plan_validation["valid"] is False
    assert trace.plan_validation["issues"][0]["code"] == "unknown_compliance_has_actions"
    assert trace.repair_attempts[0]["stage"] == "planner"
    assert trace.repair_attempts[0]["status"] == "repaired_invalid"
    assert trace.adapter_preflight == []
    assert trace.business_facts["evidence_fact_count"] == 0


def test_runtime_repairs_invalid_plan_once_before_composition():
    request = make_request()
    audit_store = AuditStore()
    invalid_plan = make_plan(
        compliance={
            "is_compliant": None,
            "reason_code": "unknown",
            "reason": "not enough context",
        }
    )
    repaired_plan = make_plan(
        user_need="ask a clarification",
        intent="clarification",
        evidence_requests=[],
        business_checks=[],
        required_adapter_resolves=[],
        candidate_actions=[],
    )
    planner = SequencedPlannerAgent([invalid_plan, repaired_plan])
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=NoopAdapterPreflightService(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: planner  # type: ignore[method-assign]

    class FakeComposerAgent:
        calls = 0

        async def kickoff_async(self, prompt, response_format):
            self.calls += 1
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-agent",
                    reply=PrimaryReply(kind="answer", text="ok"),
                    actions=[],
                ),
                raw="",
            )

    composer = FakeComposerAgent()
    runtime._build_agent = lambda: composer  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))
    trace = audit_store.latest()

    assert response.reply.kind == "answer"
    assert response.reply.text == "ok"
    assert composer.calls == 1
    assert len(planner.prompts) == 2
    assert "Repair the previous ReplyPlan" in planner.prompts[1]
    assert '"code": "unknown_compliance_has_actions"' in planner.prompts[1]
    assert trace is not None
    assert trace.plan_validation["valid"] is True
    assert trace.repair_attempts[0]["stage"] == "planner"
    assert trace.repair_attempts[0]["status"] == "repaired_valid"
    assert trace.fallback_used is False


def test_runtime_repairs_invalid_planner_contract_once_before_composition():
    request = make_request()
    audit_store = AuditStore()
    repaired_plan = make_plan(
        user_need="ask a clarification",
        intent="clarification",
        evidence_requests=[],
        business_checks=[],
        required_adapter_resolves=[],
        candidate_actions=[],
    )

    class InvalidContractThenPlan:
        def __init__(self):
            self.prompts: list[str] = []

        async def kickoff_async(self, prompt, response_format):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return SimpleNamespace(pydantic=None, raw='{"not": "ReplyPlan"}')
            return SimpleNamespace(pydantic=repaired_plan, raw="")

    planner = InvalidContractThenPlan()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=NoopAdapterPreflightService(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: planner  # type: ignore[method-assign]

    class FakeComposerAgent:
        async def kickoff_async(self, prompt, response_format):
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-agent",
                    reply=PrimaryReply(kind="answer", text="ok"),
                    actions=[],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeComposerAgent()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))
    trace = audit_store.latest()

    assert response.reply.kind == "answer"
    assert len(planner.prompts) == 2
    assert "Repair the previous ReplyPlan" in planner.prompts[1]
    assert '"code": "planner_output_invalid_contract"' in planner.prompts[1]
    assert trace is not None
    assert trace.repair_attempts[0]["stage"] == "planner"
    assert trace.repair_attempts[0]["status"] == "repaired_valid"
    assert trace.repair_attempts[0]["initial_validation"]["issues"][0]["code"] == (
        "planner_output_invalid_contract"
    )


def test_runtime_repairs_repairable_reply_once_before_fallback():
    request = make_request()
    audit_store = AuditStore()

    class FakePreflight:
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
                        "missing",
                        reason_code="weekly_report_unavailable",
                    ),
                    resolve_item("sales_mention", "resolved"),
                ]
            )

    class FakeComposerAgent:
        def __init__(self):
            self.prompts = []

        async def kickoff_async(self, prompt, response_format):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return SimpleNamespace(
                    pydantic=ReplyResponse(
                        response_id="resp-agent",
                        reply=PrimaryReply(kind="answer", text=""),
                        actions=[
                            SendWeeklyReportAction(
                                type="send_weekly_report",
                                action_id="act-weekly",
                            )
                        ],
                    ),
                    raw="",
                )
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-repaired",
                    reply=PrimaryReply(
                        kind="human_handoff",
                        text="目前这个渠道下我没有看到可发送的对应材料，我帮你请销售/支持同事确认。",
                        mentions=[
                            ReplyMention(type="sales", reason="确认周报材料")
                        ],
                    ),
                    actions=[],
                ),
                raw="",
            )

    composer = FakeComposerAgent()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=FakePreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_plan())  # type: ignore[method-assign]
    runtime._build_agent = lambda: composer  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))
    trace = audit_store.latest()

    assert response.response_id == "resp-repaired"
    assert response.reply.kind == "human_handoff"
    assert response.actions == []
    assert len(composer.prompts) == 2
    assert "Repair the previous ReplyResponse" in composer.prompts[1]
    assert trace is not None
    assert trace.fallback_used is False
    assert trace.fallback_reason == ""
    assert len(trace.repair_attempts) == 1
    assert trace.repair_attempts[0]["status"] == "repaired_valid"
    assert trace.repair_attempts[0]["initial_validation"]["issues"][0]["code"] == "action_not_resolvable"
    assert trace.reply_validation is not None
    assert trace.reply_validation["valid"] is True


def test_runtime_records_composer_repair_timeout_and_falls_back():
    request = make_request()
    audit_store = AuditStore()

    class SlowRepairComposerAgent:
        def __init__(self):
            self.calls = 0

        async def kickoff_async(self, prompt, response_format):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    pydantic=ReplyResponse(
                        response_id="resp-agent",
                        reply=PrimaryReply(kind="answer", text=""),
                        actions=[
                            SendWeeklyReportAction(
                                type="send_weekly_report",
                                action_id="act-weekly",
                            )
                        ],
                    ),
                    raw="",
                )

            await asyncio.sleep(1)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-repaired",
                    reply=PrimaryReply(kind="answer", text="ok"),
                    actions=[],
                ),
                raw="",
            )

    composer = SlowRepairComposerAgent()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", llm_timeout_seconds=0.01),
        conversation_store=ConversationStore(),
        preflight_service=MissingWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_plan())  # type: ignore[method-assign]
    runtime._build_agent = lambda: composer  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))
    trace = audit_store.latest()

    assert composer.calls == 2
    assert response.reply.kind == "human_handoff"
    assert response.actions == []
    assert trace is not None
    assert trace.fallback_used is True
    assert trace.fallback_reason == "action_not_resolvable"
    assert len(trace.repair_attempts) == 1
    assert trace.repair_attempts[0]["status"] == "repair_failed"
    assert trace.repair_attempts[0]["error"] == "CrewAI composer repair timed out"


def test_runtime_does_not_repair_fatal_reply_validation():
    request = make_request("请发周报")
    audit_store = AuditStore()

    class FakeComposerAgent:
        def __init__(self):
            self.calls = 0

        async def kickoff_async(self, prompt, response_format):
            self.calls += 1
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-agent",
                    reply=PrimaryReply(
                        kind="answer",
                        text="请看 /Users/ivan/secret/report.md",
                    ),
                    actions=[],
                ),
                raw="",
            )

    composer = FakeComposerAgent()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=NoopAdapterPreflightService(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_plan())  # type: ignore[method-assign]
    runtime._build_agent = lambda: composer  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(request))
    trace = audit_store.latest()

    assert composer.calls == 1
    assert response.reply.kind == "no_reply"
    assert response.reply.text == ""
    assert trace is not None
    assert trace.fallback_used is True
    assert trace.fallback_reason == "internal_locator_leak"
    assert trace.repair_attempts == []
