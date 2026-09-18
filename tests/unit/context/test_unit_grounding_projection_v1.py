from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.context import models as context_models
from market_support_crewai_agent.runtime.context import projection
from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    UnitBusinessFactsV1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    ExecutionUnitGroundingV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.planning.models import (
    ComplianceDecisionV1,
    DistributionExecutionDomainScopeV2,
    ExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
    UnscopedExecutionDomainScopeV2,
    execution_plan_id_v2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.conversation import (
    DistributionScopeV1,
    UnscopedScopeV1,
)


def _manifest_ref() -> ManifestRefV1:
    manifest = CAPABILITY_MANIFEST_REGISTRY.get("answer_internal_company_knowledge")
    return ManifestRefV1(
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.manifest_version,
    )


def _plan(
    scopes: tuple[ExecutionDomainScopeV2, ...],
) -> ExecutionPlanV2:
    manifest_ref = _manifest_ref()
    units = tuple(
        ExecutionPlanUnitV2(
            unit_id=f"unit-{index}",
            manifest_ref=manifest_ref,
            answerability_policy="answer",
            artifact_kind="knowledge_answer",
            scope=scope,
            evidence_query=f"query-{index}",
        )
        for index, scope in enumerate(scopes, start=1)
    )
    draft = ExecutionPlanV2.model_construct(
        execution_plan_id="epl1:" + "0" * 64,
        origin="deterministic",
        user_need="answer two bounded questions",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance=ComplianceDecisionV1(
            is_compliant=True,
            reason_code="compliant_product_request",
        ),
        units=units,
        selected_manifest_refs=(manifest_ref,),
        adapter_resolves=(),
        action_intents=(),
        guardrail_decisions=(),
        confidence=0.9,
        plan_spec=None,
    )
    return ExecutionPlanV2(
        **draft.model_dump(mode="python")
        | {"execution_plan_id": execution_plan_id_v2(draft)}
    )


def _groundings(
    plan: ExecutionPlanV2,
) -> tuple[ExecutionUnitGroundingV1, ...]:
    return tuple(
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


def _unscoped_authority() -> BusinessScopeAuthorityV1:
    return business_scope_authority_v1(UnscopedScopeV1(kind="unscoped"))


def _distribution_authority() -> BusinessScopeAuthorityV1:
    return business_scope_authority_v1(
        DistributionScopeV1(
            kind="distribution",
            dist_channel_name="Sanitized Channel",
            channel_type="bank",
            available_artifacts=[],
        )
    )


def _projection_context(
    authority: BusinessScopeAuthorityV1,
) -> context_models.GroundingProjectionContextV1:
    return context_models.GroundingProjectionContextV1(
        business_scope_authority=authority,
        locator_safety=LocatorSafetyClassifierV1(
            internal_origins=frozenset(),
            secrets=(),
        ),
        evaluation_epoch_seconds=1_000,
    )


def test_projection_preserves_repeated_manifest_refs_as_distinct_units() -> None:
    # Given: two plan units intentionally selecting the same manifest reference.
    plan = _plan(
        (
            UnscopedExecutionDomainScopeV2(),
            UnscopedExecutionDomainScopeV2(),
        )
    )

    # When: canonical groundings are projected in plan order.
    views = projection.project_unit_grounding_views_v1(
        plan,
        _groundings(plan),
        _projection_context(_unscoped_authority()),
    )

    # Then: no manifest-level deduplication loses either positional unit.
    assert tuple(view.unit_id for view in views) == ("unit-1", "unit-2")
    assert tuple(view.evidence_query for view in views) == ("query-1", "query-2")


def test_projection_rejects_cross_unit_grounding_swap() -> None:
    # Given: two valid groundings supplied in the opposite positional order.
    plan = _plan(
        (
            UnscopedExecutionDomainScopeV2(),
            UnscopedExecutionDomainScopeV2(),
        )
    )
    groundings = _groundings(plan)

    # When/Then: projection rejects before constructing model-visible views.
    with pytest.raises(ValueError, match="unit_grounding_unit_mismatch"):
        projection.project_unit_grounding_views_v1(
            plan,
            tuple(reversed(groundings)),
            _projection_context(_unscoped_authority()),
        )


def test_projection_rejects_grounding_scope_mismatch() -> None:
    # Given: a distribution plan and a grounding whose canonical scope was replaced.
    authority = _distribution_authority()
    scope = DistributionExecutionDomainScopeV2(
        business_scope_ref=authority.business_scope_ref,
        channel_kind="bank",
        product_ids=("private-product-id",),
    )
    plan = _plan((scope,))
    grounding = _groundings(plan)[0].model_copy(
        update={"scope": UnscopedExecutionDomainScopeV2()}
    )

    # When/Then: canonical scope equality is enforced before sanitization.
    with pytest.raises(ValueError, match="unit_grounding_scope_mismatch"):
        projection.project_unit_grounding_views_v1(
            plan,
            (grounding,),
            _projection_context(authority),
        )


def test_projection_rechecks_canonical_evidence_id_binding() -> None:
    # Given: model_copy bypasses the canonical grounding constructor's ID binding.
    plan = _plan((UnscopedExecutionDomainScopeV2(),))
    grounding = _groundings(plan)[0].model_copy(
        update={"allowed_evidence_ids": ("eid1:" + "a" * 64,)}
    )

    # When/Then: the projection boundary independently rejects the rebinding.
    with pytest.raises(ValueError, match="unit_grounding_evidence_binding_mismatch"):
        projection.project_unit_grounding_views_v1(
            plan,
            (grounding,),
            _projection_context(_unscoped_authority()),
        )


def test_plan_scope_projection_replaces_private_ids_with_exact_count() -> None:
    # Given: a canonical distribution scope carrying private refs and product IDs.
    authority = _distribution_authority()
    scope = DistributionExecutionDomainScopeV2(
        business_scope_ref=authority.business_scope_ref,
        channel_kind="bank",
        product_ids=("product-private-a", "product-private-b"),
    )

    # When: the sole scope projector creates the model-safe view.
    view = projection.project_plan_scope_view_v1(scope, authority)

    # Then: only the authority label and exact cardinality survive.
    assert view.model_dump(mode="json") == {
        "kind": "distribution",
        "channel_type": "bank",
        "dist_channel_name": "Sanitized Channel",
        "material_pack_option": None,
        "time_range": None,
        "product_count": 2,
    }
