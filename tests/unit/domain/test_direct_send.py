from __future__ import annotations

import time

from market_support_crewai_agent.runtime.domain.planning.direct_send import (
    match_direct_send_command,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.schemas import ReplyRequest


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "hello",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_artifacts": [
            {"type": "material_pack", "options": []},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def _match(
    message: str,
    *,
    available_artifacts: list[dict[str, str | list[str]]] | None = None,
):
    request = make_request(
        message=message,
        available_artifacts=available_artifacts
        if available_artifacts is not None
        else [
            {"type": "material_pack", "options": []},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
    )
    return match_direct_send_command(request, compile_policy(request))


def test_matches_narrow_report_send_commands():
    cases = {
        "发周报": "send_weekly_report",
        "发送周报": "send_weekly_report",
        "请发一下周报": "send_weekly_report",
        "麻烦发月报": "send_monthly_report",
        "发送月度报告": "send_monthly_report",
    }

    for message, action_type in cases.items():
        result = _match(message)

        assert result.status == "direct_action"
        assert result.plan is not None
        assert result.plan.action_intents[0].action_type == action_type


def test_matches_material_pack_aliases_without_options():
    for message in (
        "发材料包",
        "来个推介材料",
        "发一页通",
        "来个开放日历",
        "发PPT",
    ):
        result = _match(message)

        assert result.status == "direct_action"
        assert result.plan is not None
        assert result.plan.action_intents[0].action_type == "send_material_pack"


def test_matches_exact_bare_artifact_names_only():
    cases = {
        "周报": "send_weekly_report",
        "月报": "send_monthly_report",
        "材料包": "send_material_pack",
        "一页通": "send_material_pack",
        "开放日历": "send_material_pack",
    }
    for message, action_type in cases.items():
        result = _match(message)

        assert result.status == "direct_action"
        assert result.plan is not None
        assert result.plan.action_intents[0].action_type == action_type


def test_t0_is_not_a_direct_send_command():
    assert _match("T0怎么操作").status == "no_match"


def test_material_pack_with_options_requires_confirmation():
    result = _match(
        "发材料包",
        available_artifacts=[
            {
                "type": "material_pack",
                "options": [
                    "中证1000指增",
                    "中证A500指增",
                ],
            },
            {"type": "weekly_report"},
        ],
    )

    assert result.status == "needs_material_pack_option"
    assert result.plan is not None
    assert result.plan.response_mode == "clarification"
    assert result.plan.action_intents == []
    assert result.plan.ambiguity_slots == ["material_pack_option"]


def test_rejects_questions_mixed_commands_and_scoped_wording():
    for message in (
        "周报里有什么",
        "周报请求",
        "月报看看",
        "可以发周报吗",
        "发周报和月报",
        "发1000指增周报",
        "发最新净值",
        "介绍一下材料包",
        "发中证1000材料包",
        "开放日历看看",
        "材料包里有什么",
        "一页通介绍一下",
    ):
        assert _match(message).status == "no_match"


def test_policy_blocks_missing_artifact_shortcut():
    result = _match(
        "发材料包",
        available_artifacts=[{"type": "weekly_report"}],
    )

    assert result.status == "no_match"
    assert result.reason_code == "action_not_allowed_by_policy"


def test_long_unmatched_message_stays_bounded():
    request = make_request(message=("发" * 5000) + "周报和月报")
    policy = compile_policy(request)

    started = time.perf_counter()
    result = match_direct_send_command(request, policy)
    elapsed = time.perf_counter() - started

    assert result.status == "no_match"
    assert elapsed < 1.0
