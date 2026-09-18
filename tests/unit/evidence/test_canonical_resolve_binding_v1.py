from __future__ import annotations

import anyio
import pytest

from market_support_crewai_agent.runtime.decisions.business_facts import (
    derive_unit_business_facts_v1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import normalize_reply_request_v2
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.direct_send import (
    match_direct_send_command,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.schemas.adapter import AdapterResolveResult
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2


class _Preflight:
    async def collect(
        self, request, resolve_types=None, resolve_material_pack_options=None
    ):
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot(
            items=[
                AdapterPreflightItem(
                    resolve_type="weekly_report",
                    result=AdapterResolveResult.model_validate(
                        {
                            "contract_version": "adapter-resolve",
                            "resolve_type": "weekly_report",
                            "status": "resolved",
                            "display_name": "d",
                            "reason_code": "ok",
                            "candidates": [],
                            "channel_type": "bank",
                            "available_artifacts": [{"type": "weekly_report"}],
                            "resolved_at": 1,
                            "resolve_ref": "weekly.ref",
                            "period": "20260529",
                            "report_date": "2026-05-29",
                        }
                    ),
                )
            ]
        )


def _plan_and_policy():
    request = ReplyRequestV2.model_validate(
        {
            "contract_version": "reply-request.v2",
            "request_id": "req:resolve-binding-001",
            "message": "发送周报",
            "context_id": "ctx:resolve-binding-001",
            "identity": {
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "group",
                "tenant_ref": "tenant:test",
                "group_ref": "group:test",
                "principal_ref": "principal:test",
            },
            "presentation": {
                "contract_version": "group-presentation.v1",
                "conversation_name": "g",
                "principal_name": "p",
            },
            "business_scope": {
                "kind": "distribution",
                "dist_channel_name": "d",
                "channel_type": "bank",
                "available_artifacts": [{"type": "weekly_report"}],
            },
            "grants": {
                "contract_version": "principal-grants.v1",
                "read_capabilities": ["resolve_weekly_report"],
                "outbound_actions": ["send_weekly_report"],
                "mention_types": [],
            },
        }
    )
    envelope = normalize_reply_request_v2(request, adapter_namespace="test")
    scope = business_scope_authority_v1(envelope.request.business_scope)
    policy = PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(
            envelope.request,
            scope,
            group_recall_mode="off",
        ),
        policy_ledger_summary_v1(recent_artifact_types=(), recent_executed_count=0),
    )
    plan = match_direct_send_command(envelope.request, policy, scope).plan
    assert plan is not None
    return envelope.request, scope, policy, plan


def test_canonical_resolve_binding_enables_exact_action_state() -> None:
    request, scope, policy, plan = _plan_and_policy()

    async def run():
        return await EvidenceExecutor(_Preflight()).execute_v2(
            request,
            plan,
            policy,
            scope_authority=scope,
        )

    result = anyio.run(run)

    state = result.groundings[0].business_facts.weekly_report
    assert state.resolve_ref == "weekly.ref"
    assert state.period == "20260529"
    assert state.report_date is not None


def test_canonical_resolve_binding_rejects_provenance_mismatch() -> None:
    request, scope, policy, plan = _plan_and_policy()

    async def run():
        return await EvidenceExecutor(_Preflight()).execute_v2(
            request,
            plan,
            policy,
            scope_authority=scope,
        )

    result = anyio.run(run)
    grounding = result.groundings[0]
    binding = result.resolve_bindings[0].model_copy(
        update={"source_record_ref": "esr1:" + "0" * 64}
    )
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(grounding.manifest_ref.manifest_id)

    with pytest.raises(
        ValueError, match="canonical_resolve_binding_provenance_mismatch"
    ):
        derive_unit_business_facts_v1(
            plan.units[0],
            manifest,
            policy,
            grounding.allowed_evidence,
            (binding,),
        )


def test_unit_business_facts_keeps_missing_report_unknown_and_denies_ineligible() -> (
    None
):
    # Given: a declared weekly-report unit whose policy no longer admits its manifest.
    _, _, policy, plan = _plan_and_policy()
    unit = plan.units[0]
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(unit.manifest_ref.manifest_id)
    denied_policy = policy.model_copy(update={"eligible_capabilities": ()})

    # When: no canonical evidence is admitted for the unit.
    facts = derive_unit_business_facts_v1(
        unit,
        manifest,
        denied_policy,
        (),
    )

    # Then: absence is preserved and policy cannot be widened by derivation.
    assert facts.weekly_report.availability == "unknown"
    assert facts.user_permission == "denied"
