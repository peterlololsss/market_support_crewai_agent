from __future__ import annotations

from market_support_crewai_agent.runtime.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.policy import compile_policy
from market_support_crewai_agent.runtime.prompt_context import PromptAssemblyContext
from market_support_crewai_agent.runtime.prompt_router import (
    model_family_from_settings,
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings


def make_request(message: str, **overrides) -> ReplyRequest:
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
        "available_strategies": ["中证A500", "中证1000", "中证500"],
        "channel_type": "non_bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def planner_program(message: str, **overrides):
    request = make_request(message, **overrides)
    canonical_context = canonicalize_request(request)
    policy = compile_policy(request, doc_mcp_enabled=True)
    gate = route_intent(request, canonical_context, policy)
    return gate, select_prompt_program(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            canonical_context=canonical_context,
            policy=policy,
            intent_gate=gate,
        )
    )


def test_material_request_selects_material_fragments_only():
    gate, program = planner_program("发一下中证1000材料")

    assert gate.artifact_hint == "material_pack"
    assert "capability.material_pack" in program.fragment_ids
    assert "examples.material_pack" in program.fragment_ids
    assert "capability.weekly_report" not in program.fragment_ids
    assert "capability.monthly_report" not in program.fragment_ids


def test_weekly_report_request_selects_weekly_fragments_only():
    gate, program = planner_program("发下这个渠道的周报")

    assert gate.artifact_hint == "weekly_report"
    assert "capability.weekly_report" in program.fragment_ids
    assert "examples.report_scope" in program.fragment_ids
    assert "capability.material_pack" not in program.fragment_ids
    assert "capability.monthly_report" not in program.fragment_ids


def test_monthly_report_request_selects_monthly_report():
    gate, program = planner_program("发下这个渠道的月报")

    assert gate.artifact_hint == "monthly_report"
    assert "capability.monthly_report" in program.fragment_ids
    assert "examples.report_scope" in program.fragment_ids
    assert "capability.weekly_report" not in program.fragment_ids


def test_knowledge_question_selects_document_context_examples():
    gate, program = planner_program("报告里有没有小市值")

    assert gate.artifact_hint == "knowledge_answer"
    assert "capability.document_context" in program.fragment_ids
    assert "examples.knowledge_answer" in program.fragment_ids


def test_compliance_blocked_request_selects_refusal_examples():
    gate, program = planner_program("能保本吗")

    assert gate.artifact_hint == "refusal"
    assert gate.compliance_hint == "blocked"
    assert "compliance.refusal_examples" in program.fragment_ids


def test_human_support_request_selects_sales_handoff():
    gate, program = planner_program("帮我问下销售")

    assert gate.artifact_hint == "human_support"
    assert "capability.sales_handoff" in program.fragment_ids
    assert "examples.handoff" in program.fragment_ids


def test_multi_artifact_send_selects_clarification_without_action_fragments():
    gate, program = planner_program("材料和周报都发一下")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is True
    assert "examples.multi_artifact_clarification" in program.fragment_ids
    assert "capability.material_pack" not in program.fragment_ids
    assert "capability.weekly_report" not in program.fragment_ids


def test_bank_material_request_injects_bank_material_rules():
    _, program = planner_program("发一下中证1000材料", channel_type="bank")

    assert "channel.bank_material_rules" in program.fragment_ids


def test_ds_v4pro_model_selects_ds_structured_fragment():
    assert model_family_from_settings(Settings(llm_model="deepseek-v4-pro")) == "ds_v4pro"
    _, program = planner_program("发一下中证1000材料")

    assert "model.ds_v4pro.structured" in program.fragment_ids
    assert "model.generic.structured" not in program.fragment_ids
