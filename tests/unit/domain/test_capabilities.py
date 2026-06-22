from __future__ import annotations

from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.domain.business_facts import (
    ReportState,
    ResolvableState,
    derive_business_facts,
)
from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_REGISTRY,
    adapter_resolve_types,
    capability_by_action_type,
    capability_by_name,
    capability_by_resolve_type,
    read_capabilities,
    read_capabilities_for_artifact,
    resolvable_fact_type_for_resolve,
    resolve_type_for_action,
    side_effect_action_types,
)
from market_support_crewai_agent.runtime.evidence import evidence_facts_from_preflight
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.schemas import AdapterResolveResult, ReplyRequest


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
        "available_artifacts": [
            {"type": "material_pack", "options": ["指增"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def resolved_item(resolve_type: str, resolve_ref: str) -> AdapterPreflightItem:
    return AdapterPreflightItem(
        resolve_type=resolve_type,  # type: ignore[arg-type]
        result=AdapterResolveResult.model_validate(
            {
                "contract_version": "adapter-resolve",
                "resolve_type": resolve_type,
                "status": "resolved",
                "display_name": "测试渠道",
                "reason_code": "ok",
                "candidates": [],
                "channel_type": "bank",
                "available_artifacts": [
                    {"type": "material_pack", "options": ["指增"]},
                    {"type": "weekly_report"},
                    {"type": "monthly_report"},
                ],
                "resolved_at": 1,
                "resolve_ref": resolve_ref,
                "material_pack_option": "指增"
                if resolve_type == "material_pack"
                else None,
                "period": "20260612" if resolve_type == "weekly_report" else None,
                "report_date": "2026-06-12" if resolve_type == "weekly_report" else None,
            }
        ),
    )


def test_every_side_effect_action_can_reverse_lookup_resolve_type():
    for action_type in side_effect_action_types():
        capability = capability_by_action_type(action_type)

        assert capability is not None
        assert capability.side_effect_action_type == action_type
        assert capability.resolve_type is not None
        assert resolve_type_for_action(action_type) == capability.resolve_type
        assert capability_by_resolve_type(capability.resolve_type) == capability


def test_every_adapter_resolve_can_reverse_lookup_fact_type():
    for resolve_type in adapter_resolve_types():
        capability = capability_by_resolve_type(resolve_type)

        assert capability is not None
        assert capability.resolve_type == resolve_type
        assert capability.resolvable_fact_type is not None
        assert resolvable_fact_type_for_resolve(resolve_type) == capability.resolvable_fact_type


def test_registry_has_no_duplicate_action_resolve_or_fact():
    action_types = [
        item.side_effect_action_type
        for item in CAPABILITY_REGISTRY
        if item.side_effect_action_type is not None
    ]
    resolve_types = [
        item.resolve_type for item in CAPABILITY_REGISTRY if item.resolve_type is not None
    ]
    fact_types = [
        item.resolvable_fact_type
        for item in CAPABILITY_REGISTRY
        if item.resolvable_fact_type is not None
    ]

    assert len(action_types) == len(set(action_types))
    assert len(resolve_types) == len(set(resolve_types))
    assert len(fact_types) == len(set(fact_types))


def test_registry_declares_capability_contract_mappings_without_prompt_coupling():
    material = capability_by_name("material_pack")
    weekly = capability_by_name("weekly_report")
    monthly = capability_by_name("monthly_report")
    sales = capability_by_name("sales_mention")
    document = capability_by_name("document_context")

    assert material.resolve_type == "material_pack"
    assert material.side_effect_action_type == "send_material_pack"
    assert material.read_capability == "resolve_material_pack"
    assert material.supports_material_pack_option is True
    assert weekly.resolve_type == "weekly_report"
    assert weekly.side_effect_action_type == "send_weekly_report"
    assert weekly.is_report is True
    assert monthly.resolve_type == "monthly_report"
    assert monthly.side_effect_action_type == "send_monthly_report"
    assert monthly.is_report is True
    assert sales.resolve_type == "sales_mention"
    assert sales.side_effect_action_type is None
    assert document.read_capability == "query_internal_company_info"
    assert document.resolve_type is None
    for capability in (material, weekly, monthly, sales, document):
        assert not hasattr(capability, "planner_fragment_ids")
        assert not hasattr(capability, "composer_fragment_ids")


def test_policy_uses_registry_outputs():
    policy = compile_policy(make_request(), doc_mcp_enabled=True)

    assert policy.allowed_adapter_resolves == adapter_resolve_types()
    assert policy.allowed_read_capabilities == read_capabilities()
    assert "send_material_pack" in policy.allowed_side_effect_actions
    assert "material_pack" in policy.allowed_capabilities
    assert resolve_type_for_action("send_material_pack") == "material_pack"


def test_policy_disables_document_capability_from_registry_by_default():
    policy = compile_policy(make_request())

    assert "document_context" not in policy.allowed_capabilities
    assert read_capabilities_for_artifact("knowledge_answer") == frozenset(
        {"query_internal_company_info"}
    )
    assert (
        read_capabilities_for_artifact("knowledge_answer")
        & policy.allowed_read_capabilities
    ) == frozenset()


def test_evidence_and_business_facts_use_registry_fact_and_state_fields():
    snapshot = AdapterPreflightSnapshot(
        items=[resolved_item("weekly_report", "weekly:resolve-ref")]
    )
    evidence_facts = evidence_facts_from_preflight(snapshot)
    fact_type = resolvable_fact_type_for_resolve("weekly_report")

    assert fact_type == capability_by_name("weekly_report").resolvable_fact_type
    assert any(fact.fact_type == fact_type for fact in evidence_facts)

    business_facts = derive_business_facts(evidence_facts, make_request())
    state_field = capability_by_resolve_type("weekly_report").business_state_field
    state = getattr(business_facts, state_field)

    assert isinstance(state, ReportState)
    assert state.status == "available"
    assert state.resolve_ref == "weekly:resolve-ref"
    assert state.period == "20260612"


def test_business_facts_state_fields_are_registry_driven_for_resolvables():
    evidence_facts = evidence_facts_from_preflight(
        AdapterPreflightSnapshot(
            items=[
                resolved_item("material_pack", "material:resolve-ref"),
                resolved_item("sales_mention", "sales:resolve-ref"),
            ]
        )
    )

    business_facts = derive_business_facts(evidence_facts, make_request())

    for resolve_type in ("material_pack", "sales_mention"):
        capability = capability_by_resolve_type(resolve_type)
        state = getattr(business_facts, capability.business_state_field)
        assert isinstance(state, ResolvableState)
        assert state.status == "available"
        assert state.resolve_ref == f"{resolve_type.split('_')[0]}:resolve-ref"
