from __future__ import annotations

from market_support_crewai_agent.runtime.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.policy import (
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
        "available_strategies": ["指增"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def test_compile_policy_allows_actions_from_available_materials():
    policy = compile_policy(make_request(available_materials=["weekly"]))

    assert policy.allowed_side_effect_actions == frozenset({"send_weekly_report"})
    assert "send_material_pack" not in policy.allowed_side_effect_actions
    assert "send_monthly_report" not in policy.allowed_side_effect_actions
    assert policy.required_adapter_resolves == frozenset(
        {
            "material_pack",
            "weekly_report",
            "monthly_report",
            "sales_mention",
        }
    )


def test_compile_policy_keeps_all_safe_reply_kinds_available():
    policy = compile_policy(make_request())

    assert policy.allowed_reply_kinds == frozenset(
        {
            "answer",
            "clarification",
            "human_handoff",
            "unable_to_answer",
            "no_reply",
        }
    )
    assert "unsupported_report_scope_exclusion" in policy.forbidden_claim_categories
    assert policy.evidence_call_limit == 4


def test_compile_policy_enables_document_capability_only_when_configured():
    default_policy = compile_policy(make_request())
    doc_policy = compile_policy(make_request(), doc_mcp_enabled=True)

    assert "query_internal_company_info" not in default_policy.allowed_read_capabilities
    assert "query_internal_company_info" in doc_policy.allowed_read_capabilities
    assert doc_policy.evidence_call_limit == 5


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
                        "action_type": "send_material",
                        "status": "failed",
                        "action_id": "act-failed",
                        "material_type": "monthly",
                        "material_id": "monthly:opaque",
                        "version": "202605",
                    },
                    {
                        "action_type": "send_material",
                        "status": "executed",
                        "action_id": "act-weekly",
                        "material_type": "weekly",
                        "strategy": "中证1000",
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
    assert policy.ledger_summary.recent_strategies == ("中证1000",)
    assert policy.ledger_summary.recent_versions == ("20260529",)
    assert "opaque" not in str(policy.ledger_summary.to_prompt_dict())
