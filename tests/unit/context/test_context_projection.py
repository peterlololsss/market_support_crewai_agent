from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_support_crewai_agent.runtime.context.models import ContextProjectionPolicy, prompt_json
from market_support_crewai_agent.runtime.context.payload_store import ContextPayloadStore
from market_support_crewai_agent.runtime.context.projection import ContextProjectionManager
from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.ontology import ArtifactScope, DomainContextBuilder
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.llm.prompting.context import PromptAssemblyContext, render_prompt_context_layers
from market_support_crewai_agent.runtime.llm.prompting.router import route_intent
from market_support_crewai_agent.runtime.state.conversation_store import ConversationMessage
from market_support_crewai_agent.runtime.validation.answerability import AnswerabilityGate
from tests.helpers.planning import compile_test_plan, make_request


def _messages(count: int, *, content_prefix: str = "msg") -> list[ConversationMessage]:
    start = datetime(2026, 6, 17, tzinfo=timezone.utc)
    return [
        ConversationMessage(
            "user" if index % 2 == 0 else "assistant",
            f"{content_prefix}-{index}",
            start + timedelta(minutes=index),
        )
        for index in range(count)
    ]


def _material_plan(request, policy):
    return compile_test_plan(
        request,
        policy=policy,
        user_need="answer material pack products",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["material_pack"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal material question",
        },
        confidence=0.9,
    )


def _doc_plan(request, policy):
    return compile_test_plan(
        request,
        policy=policy,
        doc_mcp_enabled=True,
        user_need="answer document question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal knowledge question",
        },
        confidence=0.9,
    )


def test_projection_keeps_recent_turns_verbatim_and_summarizes_older_span():
    request = make_request(message="current")
    manager = ContextProjectionManager(
        ContextProjectionPolicy(recent_turns_verbatim_count=2)
    )

    projection = manager.project_for_stage(
        stage="planner_intent",
        request=request,
        canonical_context=canonicalize_request(request),
        policy=compile_policy(request),
        history=_messages(6),
    )

    recent = [block for block in projection.blocks if block.block_type == "recent_verbatim"]
    summaries = [block for block in projection.blocks if block.block_type == "compacted_summary"]
    assert len(recent) == 2
    assert len(summaries) == 1
    assert summaries[0].payload["original_message_count"] == 4
    assert "role" not in summaries[0].payload


def test_conversation_history_is_context_only_not_allowed_evidence():
    request = make_request(message="材料包里有什么产品")
    policy = compile_policy(request)
    plan = _material_plan(request, policy)
    history = [
        ConversationMessage(
            "assistant",
            "Product A was in material pack",
            datetime.now(timezone.utc),
        )
    ]

    projection = ContextProjectionManager().project_for_stage(
        stage="knowledge_composer",
        request=request,
        canonical_context=canonicalize_request(request),
        policy=policy,
        execution_plan=plan,
        history=history,
    )

    assert projection.allowed_evidence_ids == []
    assert all(block.block_type != "allowed_evidence" for block in projection.blocks)
    assert any(block.block_type == "recent_verbatim" for block in projection.blocks)
    assert projection.context_only_source_ids


def test_weekly_report_history_or_evidence_cannot_answer_material_pack():
    request = make_request(message="材料包里有什么产品")
    policy = compile_policy(request)
    plan = _material_plan(request, policy)
    weekly_fact = EvidenceFact(
        fact_type="report_scope_products",
        value=True,
        source_type="adapter_report_scope",
        source_id="weekly_report",
        resolve_type="weekly_report",
        metadata={"products": [{"product_name": "Weekly Product"}]},
        artifact_type="weekly_report",
        scope=ArtifactScope(channel_id="unknown"),
    )
    domain_context = DomainContextBuilder().build(request, available_artifacts=[weekly_fact])
    answerability = AnswerabilityGate().assess(
        request=request,
        canonical_context=canonicalize_request(request, domain_context=domain_context),
        domain_context=domain_context,
        plan=plan,
        policy=policy,
        evidence_facts=[weekly_fact],
    )

    projection = ContextProjectionManager().project_for_stage(
        stage="knowledge_composer",
        request=request,
        canonical_context=canonicalize_request(request, domain_context=domain_context),
        domain_context=domain_context,
        policy=policy,
        execution_plan=plan,
        evidence_facts=[weekly_fact],
        answerability_assessment=answerability,
    )
    prompt = prompt_json(projection.to_prompt_runtime_payload())

    assert answerability.recommended_response_mode == "abstain"
    assert projection.allowed_evidence_ids == []
    assert "Weekly Product" not in prompt
    assert any(block.redacted for block in projection.blocks if block.block_type == "disallowed_evidence")


def test_current_material_pack_evidence_is_allowed_and_report_evidence_redacted():
    request = make_request(message="材料包里有什么产品", material_pack_options=["指增"])
    policy = compile_policy(request)
    plan = _material_plan(request, policy)
    material_fact = EvidenceFact(
        fact_type="material_pack_product_list",
        value=True,
        source_type="adapter_material_pack_content",
        source_id="material_pack",
        resolve_type="material_pack",
        metadata={"products": [{"product_name": "Current Product"}]},
        artifact_type="material_pack",
        scope=ArtifactScope(channel_id="unknown"),
    )
    weekly_fact = EvidenceFact(
        fact_type="report_scope_products",
        value=True,
        source_type="adapter_report_scope",
        source_id="weekly_report",
        resolve_type="weekly_report",
        metadata={"products": [{"product_name": "Weekly Product"}]},
        artifact_type="weekly_report",
        scope=ArtifactScope(channel_id="unknown"),
    )

    projection = ContextProjectionManager().project_for_stage(
        stage="knowledge_composer",
        request=request,
        canonical_context=canonicalize_request(request),
        policy=policy,
        execution_plan=plan,
        evidence_facts=[material_fact, weekly_fact],
        business_facts=derive_business_facts([material_fact, weekly_fact], request),
    )
    prompt = prompt_json(projection.to_prompt_runtime_payload())

    assert any(block.block_type == "allowed_evidence" for block in projection.blocks)
    assert "Current Product" in prompt
    assert "Weekly Product" not in prompt
    assert any(block.redacted for block in projection.blocks if block.block_type == "disallowed_evidence")


def test_large_evidence_becomes_preview_with_reload_handle():
    request = make_request(message="介绍一下这个策略")
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = _doc_plan(request, policy)
    store = ContextPayloadStore()
    manager = ContextProjectionManager(
        ContextProjectionPolicy(large_result_preview_chars=80),
        payload_store=store,
    )
    huge_text = "DOCUMENT-CONTENT-" * 300
    fact = EvidenceFact(
        fact_type="document_context",
        value=True,
        source_type="document_mcp",
        source_id="doc-1",
        metadata={"content": huge_text, "status": "ok"},
    )

    projection = manager.project_for_stage(
        stage="knowledge_composer",
        request=request,
        canonical_context=canonicalize_request(request),
        policy=policy,
        execution_plan=plan,
        evidence_facts=[fact],
    )
    prompt = prompt_json(projection.to_prompt_runtime_payload())

    assert store.count() == 1
    assert "ctx-payload:" in prompt
    assert "large_result_preview" in prompt
    assert huge_text not in prompt
    assert "DOCUMENT-CONTENT-" in prompt


def test_large_old_conversation_message_is_summarized_not_inlined():
    request = make_request(message="follow up")
    huge = "OLD-HUGE-HISTORY" * 500
    history = [
        ConversationMessage("assistant", huge, datetime.now(timezone.utc)),
        ConversationMessage("user", "recent", datetime.now(timezone.utc)),
    ]

    projection = ContextProjectionManager(
        ContextProjectionPolicy(recent_turns_verbatim_count=1)
    ).project_for_stage(
        stage="planner_intent",
        request=request,
        canonical_context=canonicalize_request(request),
        policy=compile_policy(request),
        history=history,
    )
    prompt = prompt_json(projection.to_prompt_runtime_payload())

    assert any(decision.decision == "summarize" for decision in projection.decisions)
    assert huge not in prompt
    assert "Older conversation context only" in prompt


def test_prompt_renderer_uses_model_visible_context_boundary():
    request = make_request(message="current task")
    policy = compile_policy(request)
    projection = ContextProjectionManager().project_for_stage(
        stage="planner_intent",
        request=request,
        canonical_context=canonicalize_request(request),
        policy=policy,
    )

    layers = render_prompt_context_layers(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            canonical_context=canonicalize_request(request),
            policy=policy,
            intent_gate=route_intent(request, canonicalize_request(request), policy),
            model_visible_context=projection,
        )
    )

    assert layers["runtime"].count("Runtime Capability & Evidence Boundary JSON:") == 1
    assert projection.projection_id in layers["runtime"]
    assert layers["task"] == "Current user message:\ncurrent task"
