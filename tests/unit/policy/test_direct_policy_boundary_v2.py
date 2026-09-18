from __future__ import annotations

from typing import TypeAlias

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
    business_scope_hash_v1,
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
    policy_manifest_id_v2,
    state_admission_hash_v1,
)
from market_support_crewai_agent.schemas.conversation import DistributionScopeV1
from tests.helpers.reply_contract_requests import make_v2_envelope

PolicyPayloadValue: TypeAlias = (
    str
    | int
    | bool
    | tuple[str, ...]
    | tuple[ManifestRefV1, ...]
    | PolicyLedgerSummaryV1
)
PolicyPayload: TypeAlias = dict[str, PolicyPayloadValue]
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
_GENERAL_DIRECT_REPLY_MODES_WITH_ACTION = (
    "action",
    *_GENERAL_DIRECT_REPLY_MODES,
)


def _direct_core(read_capabilities: list[str] | None = None) -> PolicyAuthorityCoreV1:
    request = make_v2_envelope(
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:direct-boundary",
            "direct_thread_ref": "direct:direct-boundary",
            "principal_ref": "principal:direct-boundary",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "direct boundary",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": read_capabilities or [],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    return compile_policy_authority_core_v1(
        request,
        business_scope_authority_v1(request.business_scope),
        group_recall_mode="shortcut",
    )


def _direct_policy(read_capabilities: list[str] | None = None) -> PolicyManifestV2:
    return PolicyManifestV2.from_core(
        _direct_core(read_capabilities),
        policy_ledger_summary_v1((), 0),
    )


def _manifest_ref(manifest_id: CapabilityManifestIdV2) -> ManifestRefV1:
    return ManifestRefV1(
        manifest_id=manifest_id,
        manifest_version="2026-07-18.1",
    )


def _distribution_scope_hash() -> str:
    return business_scope_hash_v1(
        DistributionScopeV1(
            kind="distribution",
            dist_channel_name="manifest boundary distribution",
            channel_type="non_bank",
            available_artifacts=[],
        )
    )


def _direct_core_payload() -> PolicyPayload:
    core = _direct_core()
    return {
        field_name: getattr(core, field_name)
        for field_name in PolicyAuthorityCoreV1.model_fields
    }


def _direct_manifest_payload_with_coherent_hashes(
    **updates: PolicyPayloadValue,
) -> PolicyPayload:
    policy = _direct_policy()
    payload: PolicyPayload = {
        field_name: getattr(policy, field_name)
        for field_name in PolicyManifestV2.model_fields
    }
    payload.update(updates)
    core_payload = {
        field_name: payload[field_name]
        for field_name in PolicyAuthorityCoreV1.model_fields
    }
    core_payload["contract_version"] = "policy-authority-core.v1"
    core = PolicyAuthorityCoreV1.model_construct(_fields_set=None, **core_payload)
    payload["state_admission_hash"] = state_admission_hash_v1(core)
    draft = PolicyManifestV2.model_construct(_fields_set=None, **payload)
    payload["policy_id"] = policy_manifest_id_v2(draft)
    return payload


@pytest.mark.parametrize(
    ("field_name", "widened_value"),
    (
        ("allowed_reply_modes", _GENERAL_DIRECT_REPLY_MODES_WITH_ACTION),
        (
            "eligible_capabilities",
            (
                *(_manifest_ref(value) for value in _GENERAL_DIRECT_MANIFEST_IDS),
                _manifest_ref("weekly_report.send"),
            ),
        ),
        ("allowed_read_capabilities", ("resolve_weekly_report",)),
        ("allowed_outbound_actions", ("send_weekly_report",)),
        ("allowed_mention_types", ("sales",)),
        ("allowed_adapter_resolves", ("weekly_report",)),
        ("internal_company_knowledge_enabled", True),
        ("recall_mode", "advisory"),
        ("evidence_call_limit", 8),
        ("actions_allowed", True),
        ("mentions_allowed", True),
        ("material_pack_options", ("material-a",)),
        ("business_scope_hash", _distribution_scope_hash()),
    ),
)
def test_direct_policy_authority_core_rejects_every_widened_authority_field(
    field_name: str,
    widened_value: PolicyPayloadValue,
) -> None:
    payload = _direct_core_payload()
    payload[field_name] = widened_value

    with pytest.raises(ValidationError):
        _ = PolicyAuthorityCoreV1.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "widened_value"),
    (
        ("allowed_reply_modes", _GENERAL_DIRECT_REPLY_MODES_WITH_ACTION),
        (
            "eligible_capabilities",
            (
                *(_manifest_ref(value) for value in _GENERAL_DIRECT_MANIFEST_IDS),
                _manifest_ref("weekly_report.send"),
            ),
        ),
        ("allowed_read_capabilities", ("resolve_weekly_report",)),
        ("allowed_outbound_actions", ("send_weekly_report",)),
        ("allowed_mention_types", ("sales",)),
        ("allowed_adapter_resolves", ("weekly_report",)),
        ("internal_company_knowledge_enabled", True),
        ("recall_mode", "shortcut"),
        ("evidence_call_limit", 8),
        ("actions_allowed", True),
        ("mentions_allowed", True),
        ("material_pack_options", ("material-a",)),
        ("business_scope_hash", _distribution_scope_hash()),
        (
            "ledger_summary",
            policy_ledger_summary_v1(("weekly_report",), 1),
        ),
    ),
)
def test_direct_policy_manifest_rejects_every_valid_hash_widened_authority_field(
    field_name: str,
    widened_value: PolicyPayloadValue,
) -> None:
    payload = _direct_manifest_payload_with_coherent_hashes(
        **{field_name: widened_value}
    )

    with pytest.raises(ValidationError):
        _ = PolicyManifestV2.model_validate(payload)


@pytest.mark.parametrize("read_capabilities", ([], ["query_internal_company_info"]))
def test_direct_policy_manifest_from_core_round_trips_valid_direct_policy(
    read_capabilities: list[str],
) -> None:
    policy = _direct_policy(read_capabilities)

    parsed = PolicyManifestV2.model_validate(policy.model_dump(mode="json"))

    assert parsed == policy
    assert parsed.evidence_call_limit == 0
    assert parsed.ledger_summary == policy_ledger_summary_v1((), 0)
