from __future__ import annotations

from typing import Literal

from market_support_crewai_agent.runtime.context.business_view_models import (
    BusinessFactsViewV1,
    ReportStateViewV1,
    ResolvableStateViewV1,
)
from market_support_crewai_agent.runtime.context.models import RecallPlannerViewV1
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestIdV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.prompt_projection import (
    CapabilityViewAuthorityV1,
    project_capability_view_set,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    ComposerCapabilityViewSetV1,
    PlannerCapabilityViewSetV1,
    VerifierCapabilityViewSetV1,
)
from market_support_crewai_agent.runtime.recall.outcome_models import (
    RecallBranchOutcomeV1,
)


def _manifest_ref(manifest_id: CapabilityManifestIdV2) -> ManifestRefV1:
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(manifest_id)
    return ManifestRefV1(
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.manifest_version,
    )


KNOWLEDGE_REF = _manifest_ref("answer_internal_company_knowledge")
SMALLTALK_REF = _manifest_ref("general.smalltalk")


def planner_capabilities() -> PlannerCapabilityViewSetV1:
    value = project_capability_view_set(
        stage="planner_intent",
        authority=CapabilityViewAuthorityV1(
            eligible_manifest_refs=(KNOWLEDGE_REF,),
            selected_manifest_refs=(),
        ),
    )
    assert isinstance(value, PlannerCapabilityViewSetV1)
    return value


def composer_capabilities(
    stage: Literal["knowledge_composer", "smalltalk_composer"],
    manifest_ref: ManifestRefV1,
) -> ComposerCapabilityViewSetV1:
    value = project_capability_view_set(
        stage=stage,
        authority=CapabilityViewAuthorityV1(
            eligible_manifest_refs=(manifest_ref,),
            selected_manifest_refs=(manifest_ref,),
        ),
    )
    assert isinstance(value, ComposerCapabilityViewSetV1)
    return value


def verifier_capabilities() -> VerifierCapabilityViewSetV1:
    value = project_capability_view_set(
        stage="alignment_verifier",
        authority=CapabilityViewAuthorityV1(
            eligible_manifest_refs=(KNOWLEDGE_REF,),
            selected_manifest_refs=(KNOWLEDGE_REF,),
        ),
    )
    assert isinstance(value, VerifierCapabilityViewSetV1)
    return value


def business_facts() -> BusinessFactsViewV1:
    resolvable = ResolvableStateViewV1(
        availability="unknown",
        source_available=False,
        resolve_ref_available=False,
    )
    report = ReportStateViewV1(
        availability="unknown",
        source_available=False,
        resolve_ref_available=False,
    )
    return BusinessFactsViewV1(
        material_pack=resolvable,
        weekly_report=report,
        monthly_report=report,
        sales_mention=resolvable,
        requested_material_pack_option_status="unknown",
        user_permission="unknown",
        evidence_fact_count=0,
    )


def recall() -> RecallPlannerViewV1:
    return RecallPlannerViewV1(
        mode="off",
        decision="disabled",
        candidates=(),
        branch_outcomes=(
            RecallBranchOutcomeV1(
                source_class="approved_static",
                status="not_called",
                accepted_count=0,
                rejected_count=0,
                reason_code="disabled",
            ),
            RecallBranchOutcomeV1(
                source_class="document_mcp",
                status="not_called",
                accepted_count=0,
                rejected_count=0,
                reason_code="disabled",
            ),
        ),
        shortcut_summary=None,
        trace_hash="rch1:" + "c" * 64,
    )
