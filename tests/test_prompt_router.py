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


def test_bare_weekly_report_shorthand_selects_weekly_send():
    gate, program = planner_program(
        "[adapter_allowed_read_capabilities: query_internal_company_info]\n周报",
        channel_type="bank",
        available_strategies=["中证1000指增", "中证A500指增", "中证全指指增"],
    )

    assert gate.artifact_hint == "weekly_report"
    assert gate.side_effect_hint is True
    assert "capability.weekly_report" in program.fragment_ids
    assert "channel.bank_material_rules" not in program.fragment_ids


def test_weekly_performance_question_selects_weekly_report_action_fragments():
    gate, program = planner_program(
        "这周500怎么样？",
        available_strategies=["中证A500", "中证1000", "中证500"],
    )

    assert gate.artifact_hint == "weekly_report"
    assert gate.side_effect_hint is True
    assert "这周" in gate.matched_keywords
    assert "怎么样" in gate.matched_keywords
    assert "capability.weekly_report" in program.fragment_ids
    assert "examples.report_scope" in program.fragment_ids


def test_metric_question_without_period_still_selects_weekly_report():
    gate, program = planner_program(
        "1000策略超额最大回撤是多少？",
        available_strategies=["中证1000", "中证500"],
    )

    assert gate.artifact_hint == "weekly_report"
    assert gate.side_effect_hint is True
    assert "最大回撤" in gate.matched_keywords
    assert "capability.weekly_report" in program.fragment_ids


def test_monthly_report_request_selects_monthly_report():
    gate, program = planner_program("发下这个渠道的月报")

    assert gate.artifact_hint == "monthly_report"
    assert "capability.monthly_report" in program.fragment_ids
    assert "examples.report_scope" in program.fragment_ids
    assert "capability.weekly_report" not in program.fragment_ids


def test_bare_monthly_report_shorthand_selects_monthly_send():
    gate, program = planner_program("月报")

    assert gate.artifact_hint == "monthly_report"
    assert gate.side_effect_hint is True
    assert "capability.monthly_report" in program.fragment_ids


def test_calendar_month_performance_question_selects_monthly_report():
    gate, program = planner_program("11月表现怎么样？")

    assert gate.artifact_hint == "monthly_report"
    assert gate.side_effect_hint is True
    assert "表现" in gate.matched_keywords
    assert "capability.monthly_report" in program.fragment_ids


def test_report_question_does_not_route_as_send_shorthand():
    gate, program = planner_program("\u5468\u62a5\u91cc\u6709\u6ca1\u6709\u4e2d\u8bc11000")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert gate.matched_keywords == []
    assert "capability.document_context" in program.fragment_ids
    assert "examples.knowledge_answer" in program.fragment_ids
    assert "examples.smalltalk" not in program.fragment_ids


def test_metric_explanation_question_does_not_route_to_weekly_send():
    gate, program = planner_program("为什么月报里没有显示产品的年化收益率？")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert "capability.document_context" in program.fragment_ids
    assert "examples.knowledge_answer" in program.fragment_ids


def test_factor_contribution_question_does_not_route_to_weekly_send():
    gate, program = planner_program("1000指增超额收益占比？")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert "capability.document_context" in program.fragment_ids
    assert "examples.knowledge_answer" in program.fragment_ids


def test_knowledge_question_selects_document_context_examples_without_keyword_intent():
    gate, program = planner_program("\u62a5\u544a\u91cc\u6709\u6ca1\u6709\u5c0f\u5e02\u503c")

    assert gate.artifact_hint == "unclear"
    assert gate.matched_keywords == []
    assert "capability.document_context" in program.fragment_ids
    assert "examples.knowledge_answer" in program.fragment_ids


def test_material_element_question_selects_material_pack_fragments():
    gate, program = planner_program("中性一号的一页通有没有？")

    assert gate.artifact_hint == "material_pack"
    assert gate.side_effect_hint is True
    assert "一页通" in gate.matched_keywords
    assert "capability.material_pack" in program.fragment_ids
    assert "examples.material_pack" in program.fragment_ids


def test_open_calendar_question_selects_material_pack_fragments():
    gate, program = planner_program("请问星汉中性1号这月最晚几号能下单？")

    assert gate.artifact_hint == "material_pack"
    assert gate.side_effect_hint is True
    assert "最晚几号能下单" in gate.matched_keywords
    assert "capability.material_pack" in program.fragment_ids


def test_already_has_material_question_does_not_trigger_material_send():
    gate, program = planner_program("请问材料包里显示的收益都是业绩报酬计提后的收益吧？")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert "capability.document_context" in program.fragment_ids


def test_simple_redemption_process_question_does_not_trigger_material_send():
    gate, program = planner_program("赎回流程图发一下")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert "capability.material_pack" not in program.fragment_ids
    assert "capability.document_context" in program.fragment_ids


def test_company_question_gets_document_capability_without_keyword_intent():
    gate, program = planner_program(
        "yanfu有多少人",
        allowed_read_capabilities=["query_internal_company_info"],
    )

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert gate.matched_keywords == []
    assert "capability.document_context" in program.fragment_ids
    assert "examples.knowledge_answer" in program.fragment_ids
    assert "examples.smalltalk" not in program.fragment_ids


def test_identity_question_selects_smalltalk_examples_without_document_context():
    gate, program = planner_program("小衍，你是谁？")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert "examples.smalltalk" in program.fragment_ids


def test_compliance_blocked_request_selects_refusal_examples():
    gate, program = planner_program("能保本吗")

    assert gate.artifact_hint == "refusal"
    assert gate.compliance_hint == "blocked"
    assert "compliance.refusal_examples" in program.fragment_ids


def test_representative_blocked_requests_select_refusal_examples():
    for message, expected in [
        ("其他管理人你们怎么看？", "其他管理人"),
        ("加你微信了，通过一下", "加你微信"),
        ("你们衍复自营盘收益多少？", "自营盘"),
        ("发我一个四级估值表吧", "四级估值"),
        ("赎回费可以免了吗？", "赎回费可以免"),
    ]:
        gate, program = planner_program(message)

        assert gate.artifact_hint == "refusal"
        assert gate.compliance_hint == "blocked"
        assert expected in gate.matched_keywords
        assert "compliance.refusal_examples" in program.fragment_ids


def test_human_support_request_selects_sales_handoff():
    gate, program = planner_program("帮我问下销售")

    assert gate.artifact_hint == "human_support"
    assert "capability.sales_handoff" in program.fragment_ids
    assert "examples.handoff" in program.fragment_ids


def test_complaint_takes_service_handoff_priority_over_product_risk_word():
    gate, program = planner_program("我要投诉你们产品安全相关的问题")

    assert gate.artifact_hint == "human_support"
    assert gate.compliance_hint == "clean"
    assert "投诉" in gate.matched_keywords
    assert "capability.sales_handoff" in program.fragment_ids


def test_smalltalk_request_selects_smalltalk_examples_without_keyword_intent():
    gate, program = planner_program("[adapter_allowed_read_capabilities: query_internal_company_info]\nhi")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert gate.matched_keywords == []
    assert "examples.smalltalk" in program.fragment_ids
    assert "capability.material_pack" not in program.fragment_ids
    assert "capability.weekly_report" not in program.fragment_ids


def test_capability_help_inquiry_selects_smalltalk_examples_without_keyword_intent():
    gate, program = planner_program("\u4f60\u80fd\u505a\u4ec0\u4e48")

    assert gate.artifact_hint == "unclear"
    assert gate.side_effect_hint is False
    assert gate.matched_keywords == []
    assert "examples.smalltalk" in program.fragment_ids


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
