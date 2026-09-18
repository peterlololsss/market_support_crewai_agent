from __future__ import annotations

from typing import TYPE_CHECKING

from market_support_crewai_agent.runtime.context.evidence_view_models import (
    EvidenceFactViewV1,
)
from market_support_crewai_agent.runtime.context.evidence_view_projection import (
    project_evidence_fact_view_v1 as _project_evidence_fact_view_v1,
)
from market_support_crewai_agent.runtime.context.grounding_projection_context import (
    GroundingProjectionContextV1,
)
from market_support_crewai_agent.runtime.context.grounding_view_projection import (
    project_unit_grounding_views_v1 as _project_unit_grounding_views_v1,
)
from market_support_crewai_agent.runtime.context.models import UnitGroundingViewV1
from market_support_crewai_agent.runtime.context.plan_view_models import (
    PlanScopeViewV1,
    ValidatedPlanViewV1,
)
from market_support_crewai_agent.runtime.context.plan_view_projection import (
    project_plan_scope_view_v1 as _project_plan_scope_view_v1,
)
from market_support_crewai_agent.runtime.context.plan_view_projection import (
    project_validated_plan_view_v1 as _project_validated_plan_view_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.planning.models import (
    ExecutionDomainScopeV2,
    ExecutionPlanV2,
)

if TYPE_CHECKING:
    from market_support_crewai_agent.runtime.evidence.grounding import (
        ExecutionUnitGroundingV1,
    )


def project_plan_scope_view_v1(
    scope: ExecutionDomainScopeV2,
    authority: BusinessScopeAuthorityV1,
) -> PlanScopeViewV1:
    return _project_plan_scope_view_v1(scope, authority)


def project_evidence_fact_view_v1(
    fact: CanonicalEvidenceFactV1,
    context: GroundingProjectionContextV1,
) -> EvidenceFactViewV1:
    return _project_evidence_fact_view_v1(fact, context)


def project_validated_plan_view_v1(
    plan: ExecutionPlanV2,
    authority: BusinessScopeAuthorityV1,
) -> ValidatedPlanViewV1:
    return _project_validated_plan_view_v1(plan, authority)


def project_unit_grounding_views_v1(
    plan: ExecutionPlanV2,
    groundings: tuple[ExecutionUnitGroundingV1, ...],
    context: GroundingProjectionContextV1,
) -> tuple[UnitGroundingViewV1, ...]:
    return _project_unit_grounding_views_v1(plan, groundings, context)
