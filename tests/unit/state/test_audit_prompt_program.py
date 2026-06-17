from __future__ import annotations

import asyncio
from types import SimpleNamespace

from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.state.audit import AuditStore
from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveResult,
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
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
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_weekly_plan_spec(**overrides) -> PlanSpec:
    request = overrides.pop("request", None)
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
    return make_plan_spec(request=request, **payload)


class FakePlannerAgent:
    def __init__(self, plan_spec: PlanSpec):
        self.plan_spec = plan_spec

    async def kickoff_async(self, prompt, response_format):
        return SimpleNamespace(pydantic=self.plan_spec, raw="")


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
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_weekly_plan_spec())  # type: ignore[method-assign]

    asyncio.run(runtime.reply(make_request()))

    trace = audit_store.latest()
    assert trace is not None
    assert trace.intent_gate["artifact_hint"] == "unclear"
    assert trace.intent_gate["side_effect_hint"] is False
    assert len(trace.prompt_programs) == 1
    assert trace.prompt_programs[0]["stage"] == "planner_intent"
    assert trace.prompt_programs[0]["prompt_hash"].startswith("sha256:")
    assert "planner.intent_taxonomy" in trace.prompt_programs[0]["fragment_ids"]
    assert trace.prompt_programs[0]["projection_id"].startswith("ctx-proj:")
    assert trace.prompt_programs[0]["model_visible_context_hash"].startswith("sha256:")


def test_deterministic_action_response_records_planner_program_only():
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_weekly_plan_spec())  # type: ignore[method-assign]

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
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        audit_store=audit_store,
    )
    runtime.evidence_executor = DocumentEvidenceExecutor()
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_weekly_plan_spec(
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
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_weekly_plan_spec())  # type: ignore[method-assign]

    asyncio.run(runtime.reply(make_request()))

    trace = audit_store.latest()
    payload = trace.to_dict()
    assert trace is not None
    assert trace.prompt_programs[0]["fragment_hashes"]
    assert trace.llm_executions[0]["prompt_hash"].startswith("sha256:")
    assert "prompt_text" not in str(payload)
    assert "<prompt_fragment" not in str(payload)
    assert "ctx-payload:" not in str(payload)


def test_audit_records_projection_metadata_without_large_payload():
    audit_store = AuditStore()
    huge = "LARGE-DOC-PAYLOAD" * 500

    class HugeDocumentEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            del canonical_context, plan, policy, action_history
            facts = [
                EvidenceFact(
                    fact_type="document_context",
                    value=huge,
                    source_type="document_mcp",
                    source_id="doc-huge",
                )
            ]
            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=facts,
                business_facts=derive_business_facts(facts, request),
            )

    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        audit_store=audit_store,
    )
    runtime.evidence_executor = HugeDocumentEvidenceExecutor()
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_weekly_plan_spec(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            report_scope="none",
        )
    )
    runtime._build_agent = lambda *_args, **_kwargs: FakeComposerAgent()  # type: ignore[method-assign]

    asyncio.run(runtime.reply(make_request("这个策略怎么样")))

    trace = audit_store.latest()
    assert trace is not None
    assert trace.prompt_programs[1]["projection_id"].startswith("ctx-proj:")
    assert trace.prompt_programs[1]["projection_decision_count"] > 0
    payload = trace.to_dict()
    assert huge not in str(payload)
    assert "truncated_for_audit" in str(payload)


def test_audit_records_alignment_verifier_program_and_verdicts():
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    runtime._build_planner_agent = lambda: FakePlannerAgent(make_weekly_plan_spec())  # type: ignore[method-assign]

    class FakeVerifierAgent:
        async def kickoff_async(self, prompt, response_format):
            assert "Candidate ReplyResponse JSON" in prompt
            assert "model-visible-context" in prompt
            assert "weekly:ref" not in prompt
            return SimpleNamespace(
                pydantic=ReplyAlignmentVerdict(
                    aligned=True,
                    safe_to_return=True,
                    confidence=0.9,
                ),
                raw="",
            )

    runtime._build_alignment_verifier_agent = lambda: FakeVerifierAgent()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(make_request()))

    trace = audit_store.latest()
    assert response.actions[0].type == "send_weekly_report"
    assert trace is not None
    assert [item["stage"] for item in trace.prompt_programs] == [
        "planner_intent",
        "alignment_verifier",
    ]
    assert trace.alignment_verdicts[0]["aligned"] is True
    assert trace.alignment_verdicts[0]["safe_to_return"] is True
    assert "weekly:ref" not in str(trace.to_dict())
