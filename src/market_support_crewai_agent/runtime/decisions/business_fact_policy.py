from __future__ import annotations

from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    CanonicalExecutedActionStateV1,
    UserPermissionStatus,
    strict_iso_date_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    DistributionEvidenceScopeIdentityV1,
    UnscopedEvidenceScopeIdentityV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.planning.models import (
    DistributionExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    UnscopedExecutionDomainScopeV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestV2,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2

_MANIFEST_READ_REQUIREMENTS_V1: dict[str, str] = {
    "material_pack.send": "resolve_material_pack",
    "weekly_report.send": "resolve_weekly_report",
    "monthly_report.send": "resolve_monthly_report",
    "sales.handoff": "resolve_sales_mention",
    "weekly_report.product_list": "query_weekly_report_product_list",
    "monthly_report.product_list": "query_monthly_report_product_list",
    "answer_internal_company_knowledge": "query_internal_company_info",
}
_MANIFEST_ACTION_REQUIREMENTS_V1: dict[str, str] = {
    "material_pack.send": "send_material_pack",
    "weekly_report.send": "send_weekly_report",
    "monthly_report.send": "send_monthly_report",
}


def unit_recent_actions_v1(
    unit: ExecutionPlanUnitV2,
    manifest: CapabilityManifestV2,
    facts: tuple[CanonicalEvidenceFactV1, ...],
) -> tuple[CanonicalExecutedActionStateV1, ...]:
    if not _manifest_admits_fact_type_v1(manifest, "recent_executed_action"):
        return ()
    actions = tuple(
        fact
        for fact in facts
        if fact.fact_type == "recent_executed_action"
        and fact.recent_executed_action is not None
        and _fact_scope_matches_unit_v1(fact, unit)
    )
    ordered = tuple(
        sorted(
            actions,
            key=lambda fact: (
                fact.recent_executed_action.received_at_epoch_seconds
                if fact.recent_executed_action is not None
                else -1,
                fact.evidence_id,
            ),
            reverse=True,
        )
    )[:20]
    return tuple(
        CanonicalExecutedActionStateV1(
            action_type=payload.action_type,
            artifact_type=payload.artifact_type,
            artifact_ref=payload.artifact_ref,
            material_pack_option=payload.material_option,
            period=payload.period,
            report_date=strict_iso_date_v1(payload.report_date),
            status_revision=payload.status_revision,
            received_at_epoch_seconds=payload.received_at_epoch_seconds,
            response_id=payload.response_id,
            action_id=payload.action_id,
            source_record_ref=fact.provenance.source_record_ref,
        )
        for fact in ordered
        for payload in (fact.recent_executed_action,)
        if payload is not None
    )


def _manifest_admits_fact_type_v1(
    manifest: CapabilityManifestV2,
    fact_type: str,
) -> bool:
    contract = manifest.evidence_contract
    return (
        fact_type in contract.allowed_fact_types
        and fact_type not in contract.forbidden_fact_types
    )


def _fact_scope_matches_unit_v1(
    fact: CanonicalEvidenceFactV1,
    unit: ExecutionPlanUnitV2,
) -> bool:
    match unit.scope, fact.scope:
        case (
            DistributionExecutionDomainScopeV2() as unit_scope,
            DistributionEvidenceScopeIdentityV1() as fact_scope,
        ):
            return (
                unit_scope.business_scope_ref == fact_scope.business_scope_ref
                and unit_scope.channel_kind == fact_scope.channel_kind
                and unit_scope.product_ids == fact_scope.product_ids
            )
        case (UnscopedExecutionDomainScopeV2(), UnscopedEvidenceScopeIdentityV1()):
            return True
        case (
            DistributionExecutionDomainScopeV2(),
            UnscopedEvidenceScopeIdentityV1(),
        ) | (UnscopedExecutionDomainScopeV2(), DistributionEvidenceScopeIdentityV1()):
            return False


def unit_permission_v1(
    unit: ExecutionPlanUnitV2,
    policy: PolicyManifestV2,
) -> UserPermissionStatus:
    if (
        not unit.runtime_capabilities
        and not unit.answer_capabilities
        and not unit.adapter_resolves
        and not unit.action_intents
    ):
        return "unknown"
    manifest_id = unit.manifest_ref.manifest_id
    is_eligible = any(ref == unit.manifest_ref for ref in policy.eligible_capabilities)
    read_requirement = _MANIFEST_READ_REQUIREMENTS_V1.get(manifest_id)
    action_requirement = _MANIFEST_ACTION_REQUIREMENTS_V1.get(manifest_id)
    read_allowed = (
        read_requirement is None or read_requirement in policy.allowed_read_capabilities
    )
    action_allowed = (
        action_requirement is None
        or action_requirement in policy.allowed_outbound_actions
    )
    mention_allowed = manifest_id != "sales.handoff" or (
        policy.mentions_allowed and "sales" in policy.allowed_mention_types
    )
    resolves_allowed = all(
        resolve.resolve_type in policy.allowed_adapter_resolves
        for resolve in unit.adapter_resolves
    )
    intents_allowed = all(
        intent.action_type in policy.allowed_outbound_actions
        for intent in unit.action_intents
    )
    return (
        "allowed"
        if is_eligible
        and read_allowed
        and action_allowed
        and mention_allowed
        and resolves_allowed
        and intents_allowed
        else "denied"
    )
