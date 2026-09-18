from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
    state_admission_hash_v1,
)
from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_json import JsonInput
from tests.helpers.reply_contract_requests import make_v2_envelope


def _core(*, granted: bool):
    grants = {
        "contract_version": "principal-grants.v1",
        "read_capabilities": ["query_internal_company_info"] if granted else [],
        "outbound_actions": [],
        "mention_types": [],
    }
    request = make_v2_envelope(grants=grants).request
    return compile_policy_authority_core_v1(
        request,
        business_scope_authority_v1(request.business_scope),
    )


def test_policy_compiles_one_unified_internal_knowledge_authority() -> None:
    denied = _core(granted=False)
    granted = _core(granted=True)

    assert denied.internal_company_knowledge_enabled is False
    assert granted.internal_company_knowledge_enabled is True
    assert denied.recall_mode == "shortcut"
    assert state_admission_hash_v1(denied) != state_admission_hash_v1(granted)
    assert (
        PolicyManifestV2.from_core(denied, policy_ledger_summary_v1((), 0)).policy_id
        != PolicyManifestV2.from_core(
            granted, policy_ledger_summary_v1((), 0)
        ).policy_id
    )
    assert {ref.manifest_id for ref in denied.eligible_capabilities}.isdisjoint(
        {"channel.strategy_summary", "channel.product_summary"}
    )
    assert "answer_internal_company_knowledge" in {
        ref.manifest_id for ref in granted.eligible_capabilities
    }


def test_direct_policy_keeps_internal_knowledge_but_disables_preplanner_recall() -> (
    None
):
    request = make_v2_envelope(
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:test",
            "direct_thread_ref": "direct:thread-1",
            "principal_ref": "principal:sender-1",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "test user",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request

    policy = compile_policy_authority_core_v1(
        request,
        business_scope_authority_v1(request.business_scope),
        group_recall_mode="advisory",
    )

    assert policy.internal_company_knowledge_enabled is True
    assert policy.recall_mode == "off"
    assert policy.allowed_outbound_actions == ()
    assert policy.allowed_mention_types == ()
    assert "answer_internal_company_knowledge" in {
        ref.manifest_id for ref in policy.eligible_capabilities
    }


def test_product_list_requires_its_own_read_grant() -> None:
    # Given: weekly report availability and only the product-list grant.
    request = make_v2_envelope(
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_weekly_report_product_list"],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request

    # When: the deterministic policy is compiled.
    policy = compile_policy_authority_core_v1(
        request,
        business_scope_authority_v1(request.business_scope),
    )

    # Then: only the weekly product-list read is eligible and resolvable.
    eligible_ids = {ref.manifest_id for ref in policy.eligible_capabilities}
    assert "weekly_report.product_list" in eligible_ids
    assert "monthly_report.product_list" not in eligible_ids
    assert "weekly_report.send" not in eligible_ids
    assert policy.allowed_adapter_resolves == ("weekly_report",)


def test_report_product_lists_are_group_distribution_only_when_granted() -> None:
    # Given: both report-list grants and report artifacts in a group distribution scene.
    grants = {
        "contract_version": "principal-grants.v1",
        "read_capabilities": [
            "query_weekly_report_product_list",
            "query_monthly_report_product_list",
        ],
        "outbound_actions": [],
        "mention_types": [],
    }
    group_request = make_v2_envelope(grants=grants).request

    # When: the canonical policy compiler evaluates the group request.
    group_policy = compile_policy_authority_core_v1(
        group_request,
        business_scope_authority_v1(group_request.business_scope),
    )

    # Then: group keeps both granted report-list authorities and resolves.
    report_manifest_ids = {
        "weekly_report.product_list",
        "monthly_report.product_list",
    }
    group_ids = {ref.manifest_id for ref in group_policy.eligible_capabilities}
    assert report_manifest_ids <= group_ids
    assert group_policy.allowed_read_capabilities == (
        "query_monthly_report_product_list",
        "query_weekly_report_product_list",
    )
    assert group_policy.allowed_adapter_resolves == (
        "monthly_report",
        "weekly_report",
    )


def test_direct_internal_company_knowledge_requires_unscoped_business_scope() -> None:
    # Given: the company-knowledge grant in direct unscoped and group distribution scenes.
    grants = {
        "contract_version": "principal-grants.v1",
        "read_capabilities": ["query_internal_company_info"],
        "outbound_actions": [],
        "mention_types": [],
    }
    direct_identity = {
        "contract_version": "conversation-identity.v1",
        "surface": "wecom",
        "scene": "direct",
        "tenant_ref": "tenant:test",
        "direct_thread_ref": "direct:thread-1",
        "principal_ref": "principal:sender-1",
    }
    direct_presentation = {
        "contract_version": "direct-presentation.v1",
        "principal_name": "test user",
    }
    direct_unscoped_request = make_v2_envelope(
        identity=direct_identity,
        presentation=direct_presentation,
        business_scope={"kind": "unscoped"},
        grants=grants,
    ).request
    group_distribution_request = make_v2_envelope(grants=grants).request

    # When: each request is compiled through the same deterministic authority seam.
    direct_unscoped_policy = compile_policy_authority_core_v1(
        direct_unscoped_request,
        business_scope_authority_v1(direct_unscoped_request.business_scope),
    )
    group_distribution_policy = compile_policy_authority_core_v1(
        group_distribution_request,
        business_scope_authority_v1(group_distribution_request.business_scope),
    )

    # Then: only direct distribution loses company authority; direct unscoped and group remain eligible.
    direct_unscoped_ids = {
        ref.manifest_id for ref in direct_unscoped_policy.eligible_capabilities
    }
    group_distribution_ids = {
        ref.manifest_id for ref in group_distribution_policy.eligible_capabilities
    }
    assert direct_unscoped_policy.internal_company_knowledge_enabled is True
    assert "answer_internal_company_knowledge" in direct_unscoped_ids
    assert direct_unscoped_policy.allowed_read_capabilities == (
        "query_internal_company_info",
    )
    assert group_distribution_policy.internal_company_knowledge_enabled is True
    assert "answer_internal_company_knowledge" in group_distribution_ids
    assert group_distribution_policy.recall_mode == "shortcut"


@pytest.mark.parametrize(
    ("business_scope", "read_capabilities"),
    (
        (
            {
                "kind": "distribution",
                "dist_channel_name": "test channel",
                "channel_type": "bank",
                "available_artifacts": [],
            },
            ["query_internal_company_info"],
        ),
        ({"kind": "unscoped"}, ["resolve_weekly_report"]),
        (
            {"kind": "unscoped"},
            ["query_internal_company_info", "resolve_weekly_report"],
        ),
    ),
)
def test_direct_request_rejects_forbidden_scope_or_read_grant(
    business_scope: JsonInput,
    read_capabilities: JsonInput,
) -> None:
    with pytest.raises(ValidationError):
        _ = make_v2_envelope(
            identity={
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "direct",
                "tenant_ref": "tenant:test",
                "direct_thread_ref": "direct:thread-1",
                "principal_ref": "principal:sender-1",
            },
            presentation={
                "contract_version": "direct-presentation.v1",
                "principal_name": "test user",
            },
            business_scope=business_scope,
            grants={
                "contract_version": "principal-grants.v1",
                "read_capabilities": read_capabilities,
                "outbound_actions": [],
                "mention_types": [],
            },
        )


def test_settings_reject_invalid_group_recall_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        _ = Settings.model_validate({"group_recall_mode": "prompt-injected"})
    monkeypatch.setenv("MARKET_AGENT_GROUP_RECALL_MODE", "prompt-injected")
    with pytest.raises(ValidationError):
        _ = get_settings()


def test_request_cannot_set_group_recall_mode() -> None:
    with pytest.raises(ValidationError):
        _ = make_v2_envelope(recall_mode="off")
