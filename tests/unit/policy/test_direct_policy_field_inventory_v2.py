from __future__ import annotations

from typing import TypeAlias

import pytest

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestIdV2,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyAuthorityCoreV1,
    PolicyLedgerSummaryV1,
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from tests.helpers.reply_contract_requests import make_v2_envelope


PolicyPayloadValue: TypeAlias = (
    str
    | int
    | bool
    | tuple[str, ...]
    | tuple[ManifestRefV1, ...]
    | PolicyLedgerSummaryV1
)
_GENERAL_DIRECT_MANIFEST_IDS: tuple[CapabilityManifestIdV2, ...] = (
    "general.clarification",
    "general.abstention",
    "general.refusal",
    "general.smalltalk",
    "general.no_reply",
    "general.handoff",
)
_GENERAL_DIRECT_REPLY_MODES = (
    "clarification",
    "handoff",
    "no_reply",
    "refusal",
    "smalltalk",
    "unable",
)


def _direct_core() -> PolicyAuthorityCoreV1:
    request = make_v2_envelope(
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:direct-field-inventory",
            "direct_thread_ref": "direct:direct-field-inventory",
            "principal_ref": "principal:direct-field-inventory",
        },
        presentation={"contract_version": "direct-presentation.v1"},
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": [],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    return compile_policy_authority_core_v1(
        request,
        business_scope_authority_v1(request.business_scope),
        group_recall_mode="shortcut",
    )


def _manifest_ref(manifest_id: CapabilityManifestIdV2) -> ManifestRefV1:
    return ManifestRefV1(
        manifest_id=manifest_id,
        manifest_version="2026-07-18.1",
    )


def _direct_policy() -> PolicyManifestV2:
    return PolicyManifestV2.from_core(
        _direct_core(),
        policy_ledger_summary_v1((), 0),
    )


@pytest.mark.parametrize(
    ("field_name", "valid_value"),
    (
        ("contract_version", "policy-authority-core.v1"),
        ("scene", "direct"),
        ("scene_ceiling_version", "scene-ceilings.v1"),
        ("allowed_reply_modes", _GENERAL_DIRECT_REPLY_MODES),
        (
            "eligible_capabilities",
            tuple(_manifest_ref(value) for value in _GENERAL_DIRECT_MANIFEST_IDS),
        ),
        ("allowed_read_capabilities", ()),
        ("allowed_outbound_actions", ()),
        ("allowed_mention_types", ()),
        ("allowed_adapter_resolves", ()),
        ("internal_company_knowledge_enabled", False),
        ("recall_mode", "off"),
        ("evidence_call_limit", 0),
        ("actions_allowed", False),
        ("mentions_allowed", False),
        ("material_pack_options", ()),
        ("effective_grants_hash", _direct_policy().effective_grants_hash),
        ("business_scope_hash", _direct_policy().business_scope_hash),
    ),
)
def test_direct_policy_authority_core_field_inventory_is_exhaustive(
    field_name: str,
    valid_value: PolicyPayloadValue,
) -> None:
    assert field_name in PolicyAuthorityCoreV1.model_fields
    assert getattr(_direct_core(), field_name) == valid_value


def test_direct_policy_authority_core_field_inventory_has_no_unchecked_fields() -> None:
    assert tuple(PolicyAuthorityCoreV1.model_fields) == (
        "contract_version",
        "scene",
        "scene_ceiling_version",
        "allowed_reply_modes",
        "eligible_capabilities",
        "allowed_read_capabilities",
        "allowed_outbound_actions",
        "allowed_mention_types",
        "allowed_adapter_resolves",
        "internal_company_knowledge_enabled",
        "recall_mode",
        "evidence_call_limit",
        "actions_allowed",
        "mentions_allowed",
        "material_pack_options",
        "effective_grants_hash",
        "business_scope_hash",
    )


def test_direct_policy_manifest_field_inventory_has_no_unchecked_fields() -> None:
    assert tuple(PolicyManifestV2.model_fields) == (
        "contract_version",
        "policy_id",
        "scene",
        "scene_ceiling_version",
        "allowed_reply_modes",
        "eligible_capabilities",
        "allowed_read_capabilities",
        "allowed_outbound_actions",
        "allowed_mention_types",
        "allowed_adapter_resolves",
        "internal_company_knowledge_enabled",
        "recall_mode",
        "evidence_call_limit",
        "actions_allowed",
        "mentions_allowed",
        "material_pack_options",
        "ledger_summary",
        "effective_grants_hash",
        "business_scope_hash",
        "state_admission_hash",
    )
