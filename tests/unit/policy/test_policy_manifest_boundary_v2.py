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
)
from tests.helpers.reply_contract_requests import make_v2_envelope


_ZERO_POL1 = "pol1:" + "0" * 64
_ZERO_PAR1 = "par1:" + "0" * 64


def _direct_policy(read_capabilities: list[str] | None = None) -> PolicyManifestV2:
    request = make_v2_envelope(
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:manifest-boundary",
            "direct_thread_ref": "direct:manifest-boundary",
            "principal_ref": "principal:manifest-boundary",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "manifest boundary",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": read_capabilities or [],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    return PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(
            request,
            business_scope_authority_v1(request.business_scope),
            group_recall_mode="shortcut",
        ),
        policy_ledger_summary_v1((), 0),
    )


def _group_policy() -> PolicyManifestV2:
    request = make_v2_envelope().request
    return PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(
            request,
            business_scope_authority_v1(request.business_scope),
            group_recall_mode="shortcut",
        ),
        policy_ledger_summary_v1(("weekly_report",), 2),
    )


@pytest.mark.parametrize("read_capabilities", ([], ["query_internal_company_info"]))
def test_policy_manifest_round_trips_generated_direct_policy(
    read_capabilities: list[str],
) -> None:
    policy = _direct_policy(read_capabilities)

    parsed = PolicyManifestV2.model_validate(policy.model_dump(mode="json"))

    assert parsed == policy
    assert parsed.scene == "direct"
    assert parsed.recall_mode == "off"
    assert parsed.allowed_outbound_actions == ()
    assert parsed.allowed_adapter_resolves == ()


def test_policy_manifest_round_trips_generated_group_policy() -> None:
    policy = _group_policy()

    parsed = PolicyManifestV2.model_validate(policy.model_dump(mode="json"))

    assert parsed == policy
    assert parsed.scene == "group"
    assert parsed.recall_mode == "shortcut"


def test_policy_manifest_rejects_forged_direct_policy_that_widens_ceiling() -> None:
    payload = _direct_policy(["query_internal_company_info"]).model_dump(mode="json")
    payload["policy_id"] = _ZERO_POL1
    payload["recall_mode"] = "shortcut"
    payload["allowed_outbound_actions"] = ["send_weekly_report"]
    payload["allowed_mention_types"] = ["sales"]
    payload["allowed_adapter_resolves"] = ["sales_mention"]
    payload["actions_allowed"] = True
    payload["mentions_allowed"] = True

    with pytest.raises(ValidationError):
        _ = PolicyManifestV2.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ("policy_id", "state_admission_hash"),
)
def test_policy_manifest_rejects_mismatched_canonical_hashes(field_name: str) -> None:
    payload = _group_policy().model_dump(mode="json")
    payload[field_name] = _ZERO_POL1 if field_name == "policy_id" else _ZERO_PAR1

    with pytest.raises(ValidationError):
        _ = PolicyManifestV2.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ("allowed_reply_modes", "eligible_capabilities"),
)
def test_policy_manifest_rejects_empty_required_authority_sets(field_name: str) -> None:
    payload = _group_policy().model_dump(mode="json")
    payload[field_name] = []

    with pytest.raises(ValidationError):
        _ = PolicyManifestV2.model_validate(payload)
