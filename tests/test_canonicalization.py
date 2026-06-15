from __future__ import annotations

from market_support_crewai_agent.runtime.canonicalization import canonicalize_request
from market_support_crewai_agent.schemas import ReplyRequest


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
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def test_canonicalize_resolves_explicit_numeric_strategy_alias():
    context = canonicalize_request(make_request("1000所有号的周报我想看看"))

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "中证1000"
    assert context.strategy_candidates == ("中证1000",)
    assert context.entities[0].raw_text == "1000"
    assert context.entities[0].source == "alias_table"


def test_canonicalize_resolves_direct_catalog_strategy():
    context = canonicalize_request(
        make_request(
            "灵活对冲材料包发一下",
            available_strategies=["灵活对冲", "完全对冲"],
        )
    )

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "灵活对冲"
    assert context.entities[0].source == "request_catalog"


def test_canonicalize_marks_multiple_explicit_strategies_ambiguous():
    context = canonicalize_request(make_request("500和1000周报都发一下"))

    assert context.strategy_status == "ambiguous"
    assert context.selected_strategy is None
    assert context.strategy_candidates == ("中证500", "中证1000")
    assert context.ambiguities == ("multiple_strategy_candidates",)


def test_canonicalize_marks_generic_index_enhancement_request_ambiguous():
    context = canonicalize_request(
        make_request(
            "指增材料包发一下",
            available_strategies=["中证500指增", "中证1000指增"],
        )
    )

    assert context.strategy_status == "ambiguous"
    assert context.selected_strategy is None
    assert context.strategy_candidates == ("中证500指增", "中证1000指增")


def test_canonicalize_uses_single_available_strategy_as_default():
    context = canonicalize_request(
        make_request("材料包发一下", available_strategies=["指增"])
    )

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "指增"
    assert context.entities[0].raw_text == ""
    assert context.entities[0].source == "request_catalog_default"


def test_canonicalize_does_not_confuse_a500_with_500():
    context = canonicalize_request(
        make_request(
            "中证A500指增介绍一下",
            available_strategies=["中证500", "中证A500"],
        )
    )

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "中证A500"


def test_canonicalize_resolves_representative_full_index_typo():
    context = canonicalize_request(
        make_request(
            "介绍一下宗曾全子",
            available_strategies=["中证全指", "中证500"],
        )
    )

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "中证全指"
    assert context.entities[0].raw_text == "宗曾全子"
    assert context.entities[0].source == "alias_table"
