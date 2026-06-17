from __future__ import annotations

from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.llm.prompting.context import PromptAssemblyContext
from market_support_crewai_agent.runtime.llm.prompting.router import (
    model_family_from_settings,
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings


OLD_PLANNER_FRAGMENTS = {
    "capability.material_pack",
    "capability.weekly_report",
    "capability.monthly_report",
    "examples.material_pack",
    "examples.report_scope",
    "examples.knowledge_answer",
    "examples.handoff",
    "channel.bank_material_rules",
    "examples.multi_artifact_clarification",
}


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


def test_route_intent_is_non_authoritative_audit_hint():
    gate, _ = planner_program("发一下中证1000材料")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert gate.compliance_hint == "unknown"
    assert gate.confidence == 0.0
    assert gate.named_strategy_count == 1


def test_representative_messages_use_same_universal_planner_fragments():
    cases = [
        ("material_send", "麻烦同步一下中证1000的一页通", {}),
        ("weekly_report", "500最近回撤修复得怎么样", {}),
        ("monthly_report", "11月表现怎么样", {}),
        ("knowledge_answer", "月报里为什么没有年化收益率", {}),
        ("smalltalk", "你是谁", {}),
        ("refusal_phrase", "这个产品能保本吗", {}),
        ("human_support", "帮我问下销售", {}),
        ("multi_artifact", "材料和周报都给我", {}),
        ("bank_material", "发一下中证1000材料", {"channel_type": "bank"}),
    ]
    programs = [planner_program(message, **overrides)[1] for _, message, overrides in cases]

    expected = (
        "base.planner_intent",
        "model.ds_v4pro.structured",
        "planner.intent_taxonomy",
        "output.plan_spec_schema",
        "compliance.reason_codes",
    )
    assert {program.fragment_ids for program in programs} == {expected}
    for program in programs:
        assert "planner.intent_taxonomy" in program.fragment_ids
        assert not (OLD_PLANNER_FRAGMENTS & set(program.fragment_ids))


def test_planner_prompt_is_compact_for_default_fixture():
    _, program = planner_program("发一下中证1000材料")

    assert len(program.prompt_text) < 21000
    assert "Universal intent taxonomy for Xiaoyan market support." in program.prompt_text
    assert "Capability registry JSON" in program.prompt_text
    assert "material_pack.product_list" in program.prompt_text
    assert "PlanSpec compact schema:" in program.prompt_text
    assert "Canonical JSON schema:" not in program.prompt_text
    assert '"$defs"' not in program.prompt_text


def test_ds_v4pro_model_selects_ds_structured_fragment():
    assert model_family_from_settings(Settings(llm_model="deepseek-v4-pro")) == "ds_v4pro"
    _, program = planner_program("发一下中证1000材料")

    assert "model.ds_v4pro.structured" in program.fragment_ids
    assert "model.generic.structured" not in program.fragment_ids
