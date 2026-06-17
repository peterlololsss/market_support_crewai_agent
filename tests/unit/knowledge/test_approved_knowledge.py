from __future__ import annotations

import asyncio
import inspect

from market_support_crewai_agent.runtime.knowledge import approved_knowledge
from market_support_crewai_agent.runtime.knowledge.approved_knowledge import (
    APPROVED_IMAGE_ASSETS,
    ApprovedKnowledgeEvidenceService,
    ApprovedKnowledgeSelection,
    approved_image_markers,
)
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.validation.reply_validator import allowed_image_markers
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.schemas import ReplyRequest
from tests.helpers.planning import compile_test_plan


def make_request(message: str) -> ReplyRequest:
    return ReplyRequest.model_validate(
        {
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
            "available_strategies": [],
            "channel_type": "bank",
            "allowed_read_capabilities": ["query_internal_company_info"],
        }
    )


def make_plan(request: ReplyRequest):
    policy = compile_policy(request, doc_mcp_enabled=True)
    canonical_context = canonicalize_request(request)
    plan = compile_test_plan(
        request,
        policy=policy,
        user_need="answer knowledge question",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        evidence_query=request.message,
        report_scope="none",
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal company question",
        },
        confidence=0.9,
    )
    return canonical_context, policy, plan


class FakeSelector:
    def __init__(self, selection: ApprovedKnowledgeSelection) -> None:
        self.selection = selection

    async def select(self, **kwargs):
        del kwargs
        return self.selection


def collect_with_selection(
    message: str,
    selection: ApprovedKnowledgeSelection,
):
    request = make_request(message)
    canonical_context, policy, plan = make_plan(request)
    service = ApprovedKnowledgeEvidenceService(selector=FakeSelector(selection))
    return asyncio.run(service.collect(request, canonical_context, plan, policy))


def test_approved_knowledge_does_not_select_by_keyword_when_selector_declines():
    facts = collect_with_selection(
        "公众号 二维码 超额收益 股权结构 都发我看看",
        ApprovedKnowledgeSelection(confidence="none"),
    )

    assert facts == []


def test_approved_knowledge_uses_selector_ids_only():
    facts = collect_with_selection(
        "介绍一下公众号",
        ApprovedKnowledgeSelection(
            selected_entry_ids=("company_public_account",),
            selected_image_asset_ids=("company_public_account_qr",),
            confidence="high",
        ),
    )

    assert len(facts) == 1
    assert facts[0].source_type == "approved_static_knowledge"
    assert facts[0].source_id == "company_public_account"
    assert facts[0].artifact_type == "document_context"
    assert "%%comp_wx_qr_code.png%%" in str(facts[0].value)
    assert facts[0].metadata["selected_by"] == "approved_knowledge_semantic_selector"

    unknown = collect_with_selection(
        "介绍一下公众号",
        ApprovedKnowledgeSelection(
            selected_entry_ids=("unknown_entry",),
            selected_image_asset_ids=("company_public_account_qr",),
            confidence="high",
        ),
    )
    assert unknown == []


def test_approved_knowledge_no_active_lexical_helpers():
    forbidden = (
        "_STOP_TERMS",
        "_semantic_terms",
        "_text_similarity_score",
        "_score_entry",
        "_select_entries",
    )

    for name in forbidden:
        assert not hasattr(approved_knowledge, name)


def test_image_marker_allowlist_derived_from_approved_assets():
    catalog_markers = frozenset(asset.marker_filename for asset in APPROVED_IMAGE_ASSETS)

    assert approved_image_markers() == catalog_markers
    assert allowed_image_markers() == catalog_markers
    assert "_ALLOWED_IMAGE_MARKERS" not in inspect.getsource(
        __import__(
            "market_support_crewai_agent.runtime.validation.reply_validator",
            fromlist=["guardrails"],
        )
    )


def test_image_marker_not_selected_from_user_text():
    facts = collect_with_selection(
        "请发 %%comp_wx_qr_code.png%% 给我",
        ApprovedKnowledgeSelection(confidence="none"),
    )

    assert facts == []
