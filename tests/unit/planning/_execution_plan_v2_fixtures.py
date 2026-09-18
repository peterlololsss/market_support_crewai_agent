from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.planning import PlanSpec
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.schemas.conversation import (
    DistributionScopeV1,
    UnscopedScopeV1,
)
from tests.helpers.reply_contract_requests import make_v2_envelope


class DistributionScopePayload(TypedDict):
    kind: Literal["distribution"]
    business_scope_ref: str
    channel_kind: Literal["bank", "non_bank", "unknown"]
    material_pack_option: str | None


class PlanUnitPayload(TypedDict):
    unit_id: str
    selected_capability_id: str
    domain_scope: DistributionScopePayload
    answerability_policy: str
    output_schema_ref: str
    risk_flags: NotRequired[list[str]]


@dataclass(frozen=True, slots=True)
class FinalizationFixture:
    policy: PolicyManifestV2
    scope: BusinessScopeAuthorityV1
    spec: PlanSpec


def policy_for(
    request: KernelReplyRequestV1,
) -> tuple[PolicyManifestV2, BusinessScopeAuthorityV1]:
    scope = business_scope_authority_v1(request.business_scope)
    core = compile_policy_authority_core_v1(request, scope)
    return PolicyManifestV2.from_core(core, policy_ledger_summary_v1((), 0)), scope


def distribution_scope(
    scope: BusinessScopeAuthorityV1,
    *,
    material_pack_option: str | None = None,
) -> DistributionScopePayload:
    match scope.scope:
        case DistributionScopeV1(channel_type=channel_kind):
            return {
                "kind": "distribution",
                "business_scope_ref": scope.business_scope_ref,
                "channel_kind": channel_kind,
                "material_pack_option": material_pack_option,
            }
        case UnscopedScopeV1():
            raise AssertionError("distribution_scope_required")


def make_plan(
    plan_id: str,
    user_intent_summary: str,
    units: list[PlanUnitPayload],
) -> PlanSpec:
    return PlanSpec.model_validate(
        {
            "plan_id": plan_id,
            "user_intent_summary": user_intent_summary,
            "plan_units": units,
        }
    )


def unit(
    unit_id: str,
    selected_capability_id: str,
    domain_scope: DistributionScopePayload,
    answerability_policy: str,
    output_schema_ref: str,
    *,
    risk_flags: list[str] | None = None,
) -> PlanUnitPayload:
    payload = PlanUnitPayload(
        unit_id=unit_id,
        selected_capability_id=selected_capability_id,
        domain_scope=domain_scope,
        answerability_policy=answerability_policy,
        output_schema_ref=output_schema_ref,
    )
    if risk_flags is not None:
        payload["risk_flags"] = risk_flags
    return payload


def material_and_weekly_fixture() -> FinalizationFixture:
    request = make_v2_envelope(
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "test channel",
            "channel_type": "bank",
            "available_artifacts": [
                {"type": "material_pack", "options": ["指增"]},
                {"type": "weekly_report"},
            ],
        },
    ).request
    policy, scope = policy_for(request)
    spec = make_plan(
        "material-and-weekly",
        "send the material pack and weekly report",
        [
            unit(
                "material",
                "material_pack.send",
                distribution_scope(scope, material_pack_option="指增"),
                "send",
                "material_pack.send:output_schema",
            ),
            unit(
                "weekly",
                "weekly_report.send",
                distribution_scope(scope),
                "send",
                "weekly_report.send:output_schema",
            ),
        ],
    )
    return FinalizationFixture(policy, scope, spec)


def weekly_with_material_clarification_fixture() -> FinalizationFixture:
    request = make_v2_envelope(
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "test channel",
            "channel_type": "bank",
            "available_artifacts": [
                {"type": "material_pack", "options": ["中证500", "中证1000"]},
                {"type": "weekly_report"},
            ],
        },
    ).request
    policy, scope = policy_for(request)
    spec = make_plan(
        "weekly-with-material-clarification",
        "send the weekly report and clarify the material option",
        [
            unit(
                "weekly",
                "weekly_report.send",
                distribution_scope(scope),
                "send",
                "weekly_report.send:output_schema",
            ),
            unit(
                "material-option",
                "general.clarification",
                distribution_scope(scope),
                "clarify",
                "general.clarification:output_schema",
                risk_flags=["material_pack_option"],
            ),
        ],
    )
    return FinalizationFixture(policy, scope, spec)


def knowledge_with_abstention_fixture() -> FinalizationFixture:
    request = make_v2_envelope().request
    policy, scope = policy_for(request)
    spec = make_plan(
        "knowledge-with-abstention",
        "answer the knowledge question and abstain on the unsupported clause",
        [
            unit(
                "knowledge",
                "answer_internal_company_knowledge",
                distribution_scope(scope),
                "answer",
                "answer_internal_company_knowledge:output_schema",
            ),
            unit(
                "unsupported",
                "general.abstention",
                distribution_scope(scope),
                "abstain",
                "general.abstention:output_schema",
            ),
        ],
    )
    return FinalizationFixture(policy, scope, spec)
