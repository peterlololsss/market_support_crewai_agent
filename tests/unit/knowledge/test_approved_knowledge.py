from __future__ import annotations

from typing import final

import anyio

from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    ApprovedStaticKnowledgeGatewayAdapter,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayStaticContextV1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    normalize_reply_request_v2,
)
from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    ApprovedKnowledgeCandidate,
    ApprovedKnowledgeSelection,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2


def make_request(message: str) -> KernelReplyRequestV1:
    request = ReplyRequestV2.model_validate(
        {
            "contract_version": "reply-request.v2",
            "request_id": "req:approved-knowledge",
            "message": message,
            "identity": {
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "group",
                "tenant_ref": "tenant:approved-knowledge",
                "group_ref": "group:approved-knowledge",
                "principal_ref": "principal:approved-knowledge",
            },
            "presentation": {
                "contract_version": "group-presentation.v1",
                "conversation_name": "test group",
                "principal_name": "test user",
            },
            "business_scope": {
                "kind": "distribution",
                "dist_channel_name": "test channel",
                "channel_type": "bank",
                "available_artifacts": [
                    {"type": "material_pack", "options": []},
                    {"type": "weekly_report"},
                    {"type": "monthly_report"},
                ],
            },
            "grants": {
                "contract_version": "principal-grants.v1",
                "read_capabilities": ["query_internal_company_info"],
                "outbound_actions": [],
                "mention_types": [],
            },
        }
    )
    return normalize_reply_request_v2(
        request,
        adapter_namespace="assistant-wecom",
    ).request


@final
class FakeSelector:
    def __init__(self, selection: ApprovedKnowledgeSelection) -> None:
        self.selection = selection

    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection:
        del user_message, evidence_query, catalog_manifest, max_entries, max_images
        return self.selection


def collect_with_selection(
    message: str,
    selection: ApprovedKnowledgeSelection,
) -> tuple[GatewayStaticContextV1, ...]:
    request = make_request(message)
    service = ApprovedStaticKnowledgeGatewayAdapter(selector=FakeSelector(selection))

    async def collect() -> tuple[GatewayStaticContextV1, ...]:
        return await service.collect(request=request, evidence_query=message)

    return anyio.run(collect)


def test_approved_knowledge_does_not_select_by_keyword_when_selector_declines():
    contexts = collect_with_selection(
        "公众号 二维码 超额收益 股权结构 都发我看看",
        ApprovedKnowledgeSelection(confidence="none"),
    )

    assert contexts == ()


def test_approved_knowledge_uses_selector_ids_only():
    contexts = collect_with_selection(
        "介绍一下公众号",
        ApprovedKnowledgeSelection(
            selected_entry_ids=("company_public_account",),
            selected_image_asset_ids=("company_public_account_qr",),
            confidence="high",
        ),
    )

    assert len(contexts) == 1
    assert contexts[0].entry_id == "company_public_account"
    assert contexts[0].manifest_ref is not None
    assert "%%comp_wx_qr_code.png%%" in contexts[0].text
    assert contexts[0].selected_asset_ids == ("company_public_account_qr",)

    unknown = collect_with_selection(
        "介绍一下公众号",
        ApprovedKnowledgeSelection(
            selected_entry_ids=("unknown_entry",),
            selected_image_asset_ids=("company_public_account_qr",),
            confidence="high",
        ),
    )
    assert unknown == ()


def test_image_marker_not_selected_from_user_text():
    contexts = collect_with_selection(
        "请发 %%comp_wx_qr_code.png%% 给我",
        ApprovedKnowledgeSelection(confidence="none"),
    )

    assert contexts == ()
