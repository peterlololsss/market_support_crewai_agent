from __future__ import annotations

from typing import Literal, assert_never

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.context import models as context_models
from market_support_crewai_agent.runtime.context import projection
from tests.helpers.unit_grounding_contract import (
    grounding_with_fake_evidence,
    make_unit_grounding_contract_scenario,
    report_product,
    report_section,
)


MismatchKind = Literal["position", "scope", "query", "intents"]


def test_projection_preserves_repeated_refs_and_positional_unit_fields() -> None:
    # Given: two units sharing one manifest while carrying distinct scope/query/intents.
    scenario = make_unit_grounding_contract_scenario()

    # When: canonical groundings cross the public model-visible projection.
    views = projection.project_unit_grounding_views_v1(
        scenario.plan,
        scenario.groundings,
        scenario.context,
    )

    # Then: both positions and every unit-bound field remain distinct.
    assert tuple(view.unit_id for view in views) == ("unit-1", "unit-2")
    assert views[0].manifest_ref == views[1].manifest_ref
    assert tuple(view.evidence_query for view in views) == ("query-1", "query-2")
    assert tuple(
        view.scope.model_dump(mode="json").get("product_count") for view in views
    ) == (1, 2)
    assert tuple(view.action_intents[0].material_pack_option for view in views) == (
        "Option A",
        "Option B",
    )


@pytest.mark.parametrize(
    ("kind", "error_code"),
    [
        ("position", "unit_grounding_unit_mismatch"),
        ("scope", "unit_grounding_scope_mismatch"),
        ("query", "unit_grounding_query_mismatch"),
        ("intents", "unit_grounding_action_intents_mismatch"),
    ],
)
def test_projection_rejects_cross_unit_field_swaps(
    kind: MismatchKind,
    error_code: str,
) -> None:
    # Given: one canonical two-unit scenario with a field copied across positions.
    scenario = make_unit_grounding_contract_scenario()
    first, second = scenario.groundings
    match kind:
        case "position":
            groundings = (second, first)
        case "scope":
            groundings = (first.model_copy(update={"scope": second.scope}), second)
        case "query":
            groundings = (
                first.model_copy(update={"evidence_query": second.evidence_query}),
                second,
            )
        case "intents":
            groundings = (
                first.model_copy(update={"action_intents": second.action_intents}),
                second,
            )
        case unreachable:
            assert_never(unreachable)

    # When/Then: positional validation rejects before producing any view.
    with pytest.raises(ValueError, match=error_code):
        projection.project_unit_grounding_views_v1(
            scenario.plan,
            groundings,
            scenario.context,
        )


def test_projection_rejects_cross_unit_evidence_id_swap() -> None:
    # Given: distinct evidence rows whose ID tuple is swapped across unit boundaries.
    scenario = make_unit_grounding_contract_scenario()
    first = grounding_with_fake_evidence(scenario.groundings[0], 1)
    second = grounding_with_fake_evidence(scenario.groundings[1], 2)
    swapped = first.model_copy(
        update={"allowed_evidence_ids": second.allowed_evidence_ids}
    )

    # When/Then: the projection independently rechecks evidence identity binding.
    with pytest.raises(ValueError, match="unit_grounding_evidence_binding_mismatch"):
        projection.project_unit_grounding_views_v1(
            scenario.plan,
            (swapped, second),
            scenario.context,
        )


def test_report_summary_enforces_sixty_four_section_bound() -> None:
    # Given: report summaries at and just beyond the section ceiling.
    sections = tuple(report_section(index) for index in range(64))

    # When: the exact-boundary summary is constructed.
    view = context_models.ReportScopeSummaryViewV1(
        material_type="weekly",
        available=True,
        reason_code="ok",
        sections=sections,
    )

    # Then: 64 survive and a 65th rejects without clipping.
    assert len(view.sections) == 64
    with pytest.raises(ValidationError):
        context_models.ReportScopeSummaryViewV1(
            material_type="weekly",
            available=True,
            reason_code="ok",
            sections=(*sections, report_section(64)),
        )


def test_report_match_enforces_fifty_product_bound() -> None:
    # Given: report match products at and just beyond the match ceiling.
    products = tuple(report_product(index) for index in range(50))

    # When: the exact-boundary match is constructed.
    view = context_models.ReportScopeMatchViewV1(
        status="ambiguous",
        query="Product",
        candidate_count=50,
        products=products,
        page=1,
        page_size=50,
    )

    # Then: 50 survive and a 51st rejects without clipping.
    assert len(view.products) == 50
    with pytest.raises(ValidationError):
        context_models.ReportScopeMatchViewV1(
            status="ambiguous",
            query="Product",
            candidate_count=51,
            products=(*products, report_product(50)),
            page=1,
            page_size=50,
        )


def test_report_products_preserves_bounded_incomplete_projection() -> None:
    # Given: the maximum 200 projected rows from a 201-product report.
    products = tuple(report_product(index) for index in range(200))

    # When: the bounded product-list view records its completeness.
    view = context_models.ReportScopeProductsViewV1(
        available=True,
        reason_code="ok",
        products=products,
        page=1,
        page_size=50,
        total_count=201,
        returned_count=200,
        products_are_paginated=True,
        full_product_list_in_projection=False,
    )

    # Then: every projected row survives and completeness stays explicitly false.
    assert len(view.products) == 200
    assert view.full_product_list_in_projection is False


def test_report_products_rejects_false_completeness_claim() -> None:
    # Given: only 200 projected rows from a 201-product report.
    products = tuple(report_product(index) for index in range(200))

    # When/Then: claiming completeness rejects deterministically.
    with pytest.raises(ValidationError, match="report_products_completeness_mismatch"):
        context_models.ReportScopeProductsViewV1(
            available=True,
            reason_code="ok",
            products=products,
            page=1,
            page_size=50,
            total_count=201,
            returned_count=200,
            products_are_paginated=True,
            full_product_list_in_projection=True,
        )
