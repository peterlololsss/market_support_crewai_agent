from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_support_crewai_agent.runtime.context import projection as projection_module
from market_support_crewai_agent.runtime.context.models import ContextProjectionPolicy, prompt_json
from market_support_crewai_agent.runtime.context.payload_store import ContextPayloadStore
from market_support_crewai_agent.runtime.context.projection import ContextProjectionManager
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.llm.prompting.context import PromptAssemblyContext, render_prompt_context_layers
from market_support_crewai_agent.runtime.llm.prompting.router import route_intent
from market_support_crewai_agent.runtime.state.conversation_store import ConversationMessage
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
        policy=policy,
        execution_plan=plan,
        history=history,
    )

    assert projection.allowed_evidence_ids == []
    assert all(block.block_type != "allowed_evidence" for block in projection.blocks)
    assert any(block.block_type == "recent_verbatim" for block in projection.blocks)
    assert projection.context_only_source_ids


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
        policy=policy,
    )

    layers = render_prompt_context_layers(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            policy=policy,
            intent_gate=route_intent(request, policy),
            model_visible_context=projection,
        )
    )

    assert layers["runtime"].count("Runtime Capability & Evidence Boundary JSON:") == 1
    assert projection.projection_id in layers["runtime"]
    assert layers["task"] == "Current user message:\ncurrent task"


def test_projection_includes_runtime_clock_for_relative_dates():
    request = make_request(message="去年3季度的策略规模是多少？")
    policy = compile_policy(request)
    manager = ContextProjectionManager.with_runtime_clock(
        lambda: datetime(2026, 7, 2, 9, 30, tzinfo=timezone.utc)
    )

    projection = manager.project_for_stage(
        stage="planner_intent",
        request=request,
        policy=policy,
    )

    app_state = next(block.payload for block in projection.blocks if block.block_type == "app_state")
    runtime_clock = app_state["runtime_clock"]
    assert runtime_clock["current_date"] == "2026-07-02"
    assert runtime_clock["relative_years"]["去年"] == "2025"


def test_projection_from_settings_enables_runtime_clock(monkeypatch):
    monkeypatch.setattr(
        projection_module,
        "_shanghai_now",
        lambda: datetime(2026, 7, 2, 9, 30, tzinfo=timezone.utc),
    )
    request = make_request(message="去年4季度末的策略规模是多少？")
    policy = compile_policy(request)

    projection = ContextProjectionManager.from_settings().project_for_stage(
        stage="planner_intent",
        request=request,
        policy=policy,
    )

    app_state = next(block.payload for block in projection.blocks if block.block_type == "app_state")
    runtime_clock = app_state["runtime_clock"]
    assert runtime_clock["current_date"] == "2026-07-02"
    assert runtime_clock["relative_years"]["去年"] == "2025"

def test_planner_projection_exposes_contracts_not_examples():
    request = make_request(message="请按当前规则处理这个请求")
    policy = compile_policy(request, doc_mcp_enabled=True)

    projection = ContextProjectionManager().project_for_stage(
        stage="planner_intent",
        request=request,
        policy=policy,
    )
    prompt = prompt_json(projection.to_prompt_runtime_payload())
    app_state = next(block.payload for block in projection.blocks if block.block_type == "app_state")
    capability_registry = app_state["Capability registry JSON"]

    assert "capability_contracts" in capability_registry
    assert capability_registry["capability_contracts"]
    first_contract = capability_registry["capability_contracts"][0]
    assert {"id", "type", "runtime_capability", "evidence", "planner_guidance"} <= set(first_contract)
    assert "capability_cards" not in capability_registry
    assert "examples_positive" not in prompt
    assert "examples_negative" not in prompt
    assert "|pos=" not in prompt
    assert "|neg=" not in prompt
