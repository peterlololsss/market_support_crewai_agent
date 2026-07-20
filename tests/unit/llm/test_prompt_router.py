from __future__ import annotations

from typing import TypeAlias

from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
)
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.context import (
    IntentGateResult,
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.llm.prompting.router import (
    model_family_from_settings,
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings
from tests.fixtures.xiaoyan_question_set import QUESTION_SET

AvailableArtifactPayload: TypeAlias = dict[str, str | list[str]]
RequestOverrideValue: TypeAlias = str | bool | list[AvailableArtifactPayload]
PlannerCase: TypeAlias = tuple[str, str, dict[str, RequestOverrideValue]]


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


def make_request(message: str, **overrides: RequestOverrideValue) -> ReplyRequest:
    payload: dict[str, RequestOverrideValue] = {
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
            {"type": "material_pack", "options": ["中证A500", "中证1000", "中证500"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "non_bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)




def _question_set_hits(text: str) -> list[str]:
    return [question.question for question in QUESTION_SET if question.question in text]

def planner_program(
    message: str,
    **overrides: RequestOverrideValue,
) -> tuple[IntentGateResult, PromptProgram]:
    request = make_request(message, **overrides)
    policy = compile_policy(request, doc_mcp_enabled=True)
    gate = route_intent(request, policy)
    return gate, select_prompt_program(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            policy=policy,
            intent_gate=gate,
        )
    )


def test_route_intent_is_non_authoritative_audit_hint():
    gate, _ = planner_program("发一下中证1000材料")

    assert gate.artifact_hint == "unclear"
    assert gate.outbound_action_hint is False
    assert gate.compliance_hint == "unknown"
    assert gate.confidence == 0.0
    assert gate.material_pack_option_count == 3


def test_representative_messages_use_same_universal_planner_fragments():
    cases: list[PlannerCase] = [
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


def test_planner_prompt_fits_context_budget_for_default_fixture():
    _, program = planner_program("发一下中证1000材料")

    assert len(program.prompt_text) < 1_000_000
    assert "Universal intent taxonomy for Xiaoyan market support." in program.prompt_text
    assert "Capability registry JSON" in program.prompt_text
    assert "material_pack.send" in program.prompt_text
    assert "PlanSpec compact schema:" in program.prompt_text
    assert "Canonical JSON schema:" not in program.prompt_text
    assert '"$defs"' not in program.prompt_text


def test_planner_prompt_keeps_artifact_vocabulary_when_send_capability_absent():
    _, program = planner_program(
        "材料包",
        available_artifacts=[{"type": "weekly_report"}],
    )

    assert "Known sales artifact vocabulary" in program.prompt_text
    assert "材料包/material pack" in program.prompt_text
    assert "absent capabilities are not selectable" in program.prompt_text
    assert '"id": "material_pack.send"' not in program.prompt_text
    assert "send_material_pack" not in program.prompt_text


def test_planner_prompt_documents_section_roles_without_fixture_examples():
    _, program = planner_program("请按当前规则处理这个请求")

    assert "Planner stage roles:" in program.prompt_text
    assert "base planner fragment" in program.prompt_text
    assert "intent taxonomy fragment" in program.prompt_text
    assert "Capability registry JSON" in program.prompt_text
    assert "manifest-derived capability contracts" in program.prompt_text
    assert "adapter resolve/preflight owns sendability" in program.prompt_text
    assert "evidence contracts own allowed fact/source/artifact boundaries" in program.prompt_text
    assert _question_set_hits(program.prompt_text) == []


def test_planner_capability_registry_is_contract_context_not_few_shot_examples():
    _, program = planner_program("请按当前规则处理这个请求")

    assert '"capability_contracts"' in program.prompt_text
    assert '"planner_guidance"' in program.prompt_text
    assert '"evidence"' in program.prompt_text
    assert "|pos=" not in program.prompt_text
    assert "|neg=" not in program.prompt_text
    assert "examples_positive" not in program.prompt_text
    assert "examples_negative" not in program.prompt_text


def test_ds_v4pro_model_selects_ds_structured_fragment():
    assert model_family_from_settings(Settings(llm_model="deepseek-v4-pro")) == "ds_v4pro"
    _, program = planner_program("发一下中证1000材料")

    assert "model.ds_v4pro.structured" in program.fragment_ids
    assert "model.generic.structured" not in program.fragment_ids


def test_planner_override_model_family_is_stage_scoped():
    settings = Settings(
        llm_model="deepseek-v4-pro",
        planner_llm_model="gemini-3-flash-preview",
    )

    assert model_family_from_settings(settings) == "ds_v4pro"
    assert model_family_from_settings(settings, stage="planner_intent") == "generic"


def test_direct_message_uses_dedicated_composer_prompt_and_schema():
    request = make_request(
        "请把这段话发到银河客户群",
        is_group=False,
        sender_id="arbitrary-dm-sender",
    )
    policy = compile_policy(
        request,
        doc_mcp_enabled=True,
        outbound_messaging_enabled=True,
    )

    program = select_prompt_program(
        PromptAssemblyContext(
            stage="direct_composer",
            model_family="ds_v4pro",
            request=request,
            policy=policy,
        )
    )

    assert program.profile.response_model is DirectComposerOutput
    assert program.fragment_ids == (
        "base.direct_composer",
        "model.ds_v4pro.structured",
        "output.direct_composer_schema",
        "style.wecom_concise_zh",
    )
    assert "DM capability boundary" in program.prompt_text
    assert "prepare_outbound_message" in program.prompt_text
    assert "execute_prepared_outbound_message" in program.prompt_text
