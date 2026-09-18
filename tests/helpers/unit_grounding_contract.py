from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.context.models import (
    GroundingProjectionContextV1,
    ReportScopeProductViewV1,
    ReportScopeSectionViewV1,
)
from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    UnitBusinessFactsV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    ExecutionUnitGroundingV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.planning.models import (
    CanonicalActionIntentV1,
    ComplianceDecisionV1,
    DistributionExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
    execution_plan_id_v2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.conversation import DistributionScopeV1


@dataclass(frozen=True, slots=True)
class UnitGroundingContractScenario:
    plan: ExecutionPlanV2
    groundings: tuple[ExecutionUnitGroundingV1, ...]
    context: GroundingProjectionContextV1


def make_unit_grounding_contract_scenario() -> UnitGroundingContractScenario:
    authority = business_scope_authority_v1(
        DistributionScopeV1(
            kind="distribution",
            dist_channel_name="Sanitized Channel",
            channel_type="bank",
            available_artifacts=[],
        )
    )
    manifest = CAPABILITY_MANIFEST_REGISTRY.get("material_pack.send")
    manifest_ref = ManifestRefV1(
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.manifest_version,
    )
    scopes = (
        DistributionExecutionDomainScopeV2(
            business_scope_ref=authority.business_scope_ref,
            channel_kind="bank",
            product_ids=("private-product-a",),
        ),
        DistributionExecutionDomainScopeV2(
            business_scope_ref=authority.business_scope_ref,
            channel_kind="bank",
            product_ids=("private-product-b", "private-product-c"),
        ),
    )
    intents = (
        CanonicalActionIntentV1(
            action_type="send_material_pack",
            capability="material_pack",
            material_pack_option="Option A",
        ),
        CanonicalActionIntentV1(
            action_type="send_material_pack",
            capability="material_pack",
            material_pack_option="Option B",
        ),
    )
    units = tuple(
        ExecutionPlanUnitV2(
            unit_id=f"unit-{index}",
            manifest_ref=manifest_ref,
            answerability_policy="send",
            artifact_kind="material_pack",
            action_intents=(intents[index - 1],),
            scope=scopes[index - 1],
            evidence_query=f"query-{index}",
        )
        for index in (1, 2)
    )
    draft = ExecutionPlanV2.model_construct(
        execution_plan_id="epl1:" + "0" * 64,
        origin="deterministic",
        user_need="send two independently scoped material packs",
        artifact_kind="multi_action",
        response_mode="action",
        compliance=ComplianceDecisionV1(
            is_compliant=True,
            reason_code="compliant_product_request",
        ),
        units=units,
        selected_manifest_refs=(manifest_ref,),
        adapter_resolves=(),
        action_intents=intents,
        guardrail_decisions=(),
        confidence=0.9,
        plan_spec=None,
    )
    plan = ExecutionPlanV2.model_validate(
        draft.model_dump(mode="python")
        | {"execution_plan_id": execution_plan_id_v2(draft)}
    )
    groundings = tuple(
        ExecutionUnitGroundingV1(
            unit_id=unit.unit_id,
            manifest_ref=unit.manifest_ref,
            answerability=unit.answerability_policy,
            scope=unit.scope,
            evidence_query=unit.evidence_query,
            action_intents=unit.action_intents,
            business_facts=UnitBusinessFactsV1(evidence_fact_count=0),
        )
        for unit in plan.units
    )
    context = GroundingProjectionContextV1(
        business_scope_authority=authority,
        locator_safety=LocatorSafetyClassifierV1(
            internal_origins=frozenset(),
            secrets=(),
        ),
        evaluation_epoch_seconds=1_000,
    )
    return UnitGroundingContractScenario(
        plan=plan,
        groundings=groundings,
        context=context,
    )


def grounding_with_fake_evidence(
    grounding: ExecutionUnitGroundingV1,
    index: int,
) -> ExecutionUnitGroundingV1:
    fact = CanonicalEvidenceFactV1.model_construct(evidence_id=f"eid1:{index:064x}")
    return grounding.model_copy(
        update={
            "allowed_evidence_ids": (fact.evidence_id,),
            "allowed_evidence": (fact,),
            "business_facts": UnitBusinessFactsV1(evidence_fact_count=1),
        }
    )


def report_product(index: int) -> ReportScopeProductViewV1:
    return ReportScopeProductViewV1(
        product_name=f"Product {index}",
        portfolio_type="market-neutral",
        report_section="Section A",
        source_pdf_status="found",
        final_report_status="generated",
    )


def report_section(index: int) -> ReportScopeSectionViewV1:
    return ReportScopeSectionViewV1(
        name=f"Section {index}",
        source_pdf_count=1,
        final_report_count=1,
        missing_product_count=0,
    )
