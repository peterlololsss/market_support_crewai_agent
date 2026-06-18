from __future__ import annotations

from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
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
        "material_pack_options": ["中证500", "中证A500", "中证500"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def test_canonicalize_projects_material_pack_options_only():
    context = canonicalize_request(make_request("发一下 A500 材料包"))

    assert context.material_pack_options == ("中证500", "中证A500")


def test_canonicalize_does_not_infer_single_material_pack_option():
    context = canonicalize_request(
        make_request("材料包发一下", material_pack_options=["中证A500"])
    )

    assert context.material_pack_options == ("中证A500",)


def test_canonicalize_does_not_resolve_query_against_domain_context():
    context = canonicalize_request(
        make_request("中证A500指增介绍一下"),
        domain_context=object(),  # type: ignore[arg-type]
    )

    assert context.to_prompt_dict() == {
        "material_pack_options": ["中证500", "中证A500"],
    }
