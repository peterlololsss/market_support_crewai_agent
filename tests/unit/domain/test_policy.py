from __future__ import annotations

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.domain.policy import (
    compile_policy,
    ledger_summary_from_action_history,
)
from market_support_crewai_agent.schemas import ActionFeedbackRequest, ReplyRequest


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
        "available_materials": ["material", "weekly", "monthly"],
        "material_pack_options": ["指增"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def test_compile_policy_preflights_reports_without_csv_material_hint():
    policy = compile_policy(make_request(available_materials=[]))

    assert policy.policy_id == "support-reply-policy:bank"
    assert policy.allowed_side_effect_actions == frozenset(
        {
            "send_weekly_report",
            "send_monthly_report",
        }
    )
    assert "send_material_pack" not in policy.allowed_side_effect_actions
    assert policy.allowed_adapter_resolves == frozenset(
        {
            "weekly_report",
            "monthly_report",
            "sales_mention",
        }
    )
    assert "material_pack" not in policy.allowed_capabilities


def test_compile_policy_allows_material_pack_from_available_materials():
    policy = compile_policy(make_request(available_materials=["material"]))

    assert policy.allowed_side_effect_actions == frozenset(
        {
            "send_material_pack",
            "send_weekly_report",
            "send_monthly_report",
        }
    )


def test_compile_policy_keeps_all_safe_reply_kinds_available():
    policy = compile_policy(make_request())

    assert policy.allowed_reply_modes == frozenset(
        {
            "action",
            "clarification",
            "handoff",
            "refusal",
            "unable",
            "knowledge_answer",
            "smalltalk",
            "no_reply",
        }
    )
    assert policy.evidence_call_limit == 4


def test_compile_policy_enables_document_capability_only_when_configured():
    default_policy = compile_policy(make_request())
    doc_policy = compile_policy(make_request(), doc_mcp_enabled=True)

    assert "document_context" not in default_policy.allowed_capabilities
    assert "query_internal_company_info" not in default_policy.allowed_read_capabilities
    assert "document_context" in doc_policy.allowed_capabilities
    assert "knowledge_answer" in doc_policy.allowed_reply_modes
    assert "query_internal_company_info" in doc_policy.allowed_read_capabilities
    assert doc_policy.evidence_call_limit == 5


def test_compile_policy_intersects_structured_adapter_read_capabilities():
    request = make_request(
        allowed_read_capabilities=[
            "resolve_weekly_report",
            "query_internal_company_info",
        ],
    )

    policy = compile_policy(request, doc_mcp_enabled=True)

    assert policy.allowed_capabilities == frozenset(
        {"weekly_report", "document_context"}
    )
    assert policy.allowed_read_capabilities == frozenset(
        {"resolve_weekly_report", "query_internal_company_info"}
    )
    assert policy.allowed_adapter_resolves == frozenset({"weekly_report"})
    assert policy.allowed_side_effect_actions == frozenset({"send_weekly_report"})


def test_compile_policy_scopes_document_capability_by_channel_type():
    bank_policy = compile_policy(
        make_request(channel_type="bank"),
        doc_mcp_enabled=True,
        doc_mcp_allowed_channel_types=("bank",),
    )
    non_bank_policy = compile_policy(
        make_request(channel_type="non_bank"),
        doc_mcp_enabled=True,
        doc_mcp_allowed_channel_types=("bank",),
    )

    assert "query_internal_company_info" in bank_policy.allowed_read_capabilities
    assert "query_internal_company_info" not in non_bank_policy.allowed_read_capabilities
    assert non_bank_policy.evidence_call_limit == 4


def test_compile_policy_includes_adapter_safe_ledger_summary():
    ledger = ActionLedger()
    ledger.record_feedback(
        ActionFeedbackRequest.model_validate(
            {
                "conversation_key": "wecom:group-1:sender-1",
                "group_id": "group-1",
                "sender_id": "sender-1",
                "context_id": "msg-1",
                "response_id": "resp-1",
                "executions": [
                    {
                        "action_type": "send_monthly_report",
                        "status": "failed",
                        "action_id": "act-failed",
                        "resolve_ref": "monthly:resolve-ref",
                        "material_type": "monthly",
                        "material_id": "monthly:opaque",
                        "version": "202605",
                    },
                    {
                        "action_type": "send_weekly_report",
                        "status": "executed",
                        "action_id": "act-weekly",
                        "resolve_ref": "weekly:resolve-ref",
                        "material_type": "weekly",
                        "material_id": "weekly:opaque",
                        "version": "20260529",
                    },
                ],
            }
        )
    )

    summary = ledger_summary_from_action_history(
        ledger.recent_executed_for_conversation("wecom:group-1:sender-1")
    )
    policy = compile_policy(make_request(), ledger_summary=summary)

    assert policy.ledger_summary.has_recent_executed_actions is True
    assert policy.ledger_summary.recent_executed_count == 1
    assert policy.ledger_summary.recent_material_types == ("weekly",)
    assert policy.ledger_summary.recent_material_pack_options == ()
    assert policy.ledger_summary.recent_versions == ("20260529",)
    assert "opaque" not in str(policy.ledger_summary.to_prompt_dict())
