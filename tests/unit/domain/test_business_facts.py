from __future__ import annotations

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.evidence import (
    EvidenceFact,
    evidence_facts_from_action_history,
)
from market_support_crewai_agent.schemas import ActionFeedbackRequest, ReplyRequest


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "请发一下周报",
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


def test_business_facts_derive_resolved_weekly_report_state():
    facts = derive_business_facts(
        [
            EvidenceFact(
                fact_type="weekly_report_resolvable",
                value=True,
                resolve_type="weekly_report",
                source_id="weekly_report",
                metadata={"status": "resolved", "reason_code": "ok"},
            ),
            EvidenceFact(
                fact_type="report_contains_strategy",
                value=True,
                resolve_type="weekly_report",
                metadata={"strategy": "中证1000", "period": "20260529"},
            ),
            EvidenceFact(
                fact_type="report_scope_status",
                value="included",
                resolve_type="weekly_report",
                metadata={"strategy": "中证1000", "period": "20260529"},
            ),
        ],
        make_request(),
    )

    assert facts.weekly_report.status == "available"
    assert facts.weekly_report.resolvable is True
    assert facts.weekly_report.contains_strategy is True
    assert facts.weekly_report.scope_status == "included"
    assert facts.weekly_report.strategy == "中证1000"
    assert facts.weekly_report.period == "20260529"
    assert facts.requested_strategy_status == "available"


def test_business_facts_preserve_ambiguous_material_candidates():
    facts = derive_business_facts(
        [
            EvidenceFact(
                fact_type="material_pack_resolvable",
                value=False,
                resolve_type="material_pack",
                metadata={
                    "status": "ambiguous",
                    "candidates": ["中证500", "中证1000"],
                    "reason_code": "multiple_packs",
                },
            )
        ],
        make_request(),
    )

    assert facts.material_pack.status == "ambiguous"
    assert facts.material_pack.resolvable is False
    assert facts.material_pack.candidates == ("中证500", "中证1000")
    assert facts.material_pack.reason_code == "multiple_packs"
    assert facts.requested_strategy_status == "ambiguous"


def test_business_facts_preserve_resolved_material_strategy():
    facts = derive_business_facts(
        [
            EvidenceFact(
                fact_type="material_pack_resolvable",
                value=True,
                resolve_type="material_pack",
                metadata={
                    "status": "resolved",
                    "reason_code": "ok",
                    "strategy": "中证1000指增",
                },
            )
        ],
        make_request(),
    )

    assert facts.material_pack.status == "available"
    assert facts.material_pack.strategy == "中证1000指增"
    assert facts.material_pack.resolvable is True


def test_business_facts_preserve_report_strategy_from_resolve_fact():
    facts = derive_business_facts(
        [
            EvidenceFact(
                fact_type="weekly_report_resolvable",
                value=True,
                resolve_type="weekly_report",
                metadata={
                    "status": "resolved",
                    "reason_code": "ok",
                    "strategy": "中证1000",
                },
            )
        ],
        make_request(),
    )

    assert facts.weekly_report.status == "available"
    assert facts.weekly_report.strategy == "中证1000"
    assert facts.weekly_report.contains_strategy is None
    assert facts.weekly_report.scope_status == "unknown"


def test_business_facts_mark_report_exclusion_as_unavailable_strategy():
    facts = derive_business_facts(
        [
            EvidenceFact(
                fact_type="report_contains_strategy",
                value=False,
                resolve_type="monthly_report",
                metadata={"strategy": "小市值", "period": "202605"},
            ),
            EvidenceFact(
                fact_type="report_scope_status",
                value="excluded",
                resolve_type="monthly_report",
                metadata={"strategy": "小市值", "period": "202605"},
            ),
        ],
        make_request(),
    )

    assert facts.monthly_report.status == "unknown"
    assert facts.monthly_report.contains_strategy is False
    assert facts.monthly_report.scope_status == "excluded"
    assert facts.requested_strategy_status == "unavailable"


def test_business_facts_default_to_unknown_when_evidence_is_absent():
    facts = derive_business_facts([], make_request())

    assert facts.material_pack.status == "unknown"
    assert facts.weekly_report.scope_status == "unknown"
    assert facts.sales_mention.resolvable is False
    assert facts.requested_strategy_status == "unknown"
    assert facts.evidence_fact_count == 0


def test_business_facts_include_only_adapter_executed_action_history():
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
                        "action_type": "send_weekly_report",
                        "status": "failed",
                        "action_id": "act-failed",
                        "resolve_ref": "weekly:failed-ref",
                        "material_type": "weekly",
                        "material_id": "weekly:failed",
                    },
                    {
                        "action_type": "send_weekly_report",
                        "status": "executed",
                        "action_id": "act-weekly",
                        "resolve_ref": "weekly:resolve-ref",
                        "material_type": "weekly",
                        "strategy": "中证1000",
                        "material_id": "weekly:opaque",
                        "version": "20260529",
                    },
                ],
            }
        )
    )
    action_history = ledger.recent_executed_for_conversation(
        "wecom:group-1:sender-1",
    )

    evidence_facts = evidence_facts_from_action_history(action_history)
    business_facts = derive_business_facts(evidence_facts, make_request())

    assert [fact.fact_type for fact in evidence_facts] == ["recent_executed_action"]
    assert business_facts.evidence_fact_count == 1
    assert len(business_facts.recent_executed_actions) == 1
    executed = business_facts.recent_executed_actions[0]
    assert executed.action_id == "act-weekly"
    assert executed.material_type == "weekly"
    assert executed.strategy == "中证1000"
    assert executed.version == "20260529"
    assert executed.resolve_ref == "weekly:resolve-ref"
    assert executed.resolve_ref_available is True
    assert executed.material_ref_available is True
    assert "weekly:resolve-ref" not in str(business_facts.to_prompt_dict())
    assert "act-failed" not in str(business_facts.to_prompt_dict())
