from __future__ import annotations

import asyncio
from types import SimpleNamespace

from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.audit import AuditStore
from market_support_crewai_agent.runtime.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.planning import IntentFrame
from market_support_crewai_agent.runtime.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.schemas import (
    AdapterResolveResult,
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
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
        "available_strategies": ["中证1000"],
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


class FakePlannerAgent:
    def __init__(self, frame: IntentFrame):
        self.frame = frame

    async def kickoff_async(self, prompt, response_format):
        return SimpleNamespace(pydantic=self.frame, raw="")


class FakeComposerAgent:
    async def kickoff_async(self, prompt, response_format):
        return SimpleNamespace(
            pydantic=ReplyResponse(
                response_id="resp-knowledge",
                reply=PrimaryReply(kind="answer", text="文档证据回答"),
                actions=[],
            ),
            raw="",
        )


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
                AdapterPreflightItem(
                    resolve_type="weekly_report",
                    result=AdapterResolveResult.model_validate(
                        {
                            "contract_version": "adapter-resolve",
                            "resolve_type": "weekly_report",
                            "status": "resolved",
                            "display_name": "测试渠道",
                            "reason_code": "ok",
                            "candidates": [],
                            "channel_type": "bank",
                            "available_materials": ["weekly"],
                            "available_strategies": ["中证1000"],
                            "resolved_at": 1,
                            "resolve_ref": "weekly:ref",
                            "period": "20260529",
                            "report_date": "2026-05-29",
                            "scope_status": "included",
                            "contains_strategy": True,
                        }
                    ),
                )
            ]
        )


class DocumentEvidenceExecutor:
    async def execute(self, request, canonical_context, plan, policy, action_history=None):
        del canonical_context, plan, policy, action_history
        facts = [
            EvidenceFact(
                fact_type="document_context",
                value="文档证据",
                source_type="document_mcp",
                source_id="doc-1",
            )
        ]
        return SimpleNamespace(
            preflight=AdapterPreflightSnapshot.empty(),
            evidence_facts=facts,
            business_facts=derive_business_facts(facts, request),
        )


def test_runtime_audit_records_intent_gate_and_prompt_programs():
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_frame())  # type: ignore[method-assign]

    asyncio.run(runtime.reply(make_request()))

    trace = audit_store.latest()
    assert trace is not None
    assert trace.intent_gate["artifact_hint"] == "weekly_report"
    assert len(trace.prompt_programs) == 1
    assert trace.prompt_programs[0]["stage"] == "planner_intent"
    assert trace.prompt_programs[0]["prompt_hash"].startswith("sha256:")
    assert "capability.weekly_report" in trace.prompt_programs[0]["fragment_ids"]


def test_deterministic_action_response_records_planner_program_only():
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_frame())  # type: ignore[method-assign]

    asyncio.run(runtime.reply(make_request()))

    trace = audit_store.latest()
    assert trace is not None
    assert [item["stage"] for item in trace.prompt_programs] == ["planner_intent"]
    assert all(item["stage"] != "knowledge_composer" for item in trace.llm_executions)


def test_knowledge_answer_records_planner_and_composer_programs():
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
        ),
        conversation_store=ConversationStore(),
        audit_store=audit_store,
    )
    runtime.evidence_executor = DocumentEvidenceExecutor()
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_frame(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            report_scope="none",
        )
    )
    runtime._build_agent = lambda *_args, **_kwargs: FakeComposerAgent()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(make_request("这个策略怎么样")))

    trace = audit_store.latest()
    assert response.reply.kind == "answer"
    assert trace is not None
    assert [item["stage"] for item in trace.prompt_programs] == [
        "planner_intent",
        "knowledge_composer",
    ]
    assert "evidence.document_grounding" in trace.prompt_programs[1]["fragment_ids"]


def test_audit_stores_hashes_and_fragment_ids_not_full_prompt_text():
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_frame())  # type: ignore[method-assign]

    asyncio.run(runtime.reply(make_request()))

    trace = audit_store.latest()
    payload = trace.to_dict()
    assert trace is not None
    assert trace.prompt_programs[0]["fragment_hashes"]
    assert trace.llm_executions[0]["prompt_hash"].startswith("sha256:")
    assert "prompt_text" not in str(payload)
    assert "<prompt_fragment" not in str(payload)
