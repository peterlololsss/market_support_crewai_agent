from __future__ import annotations

import anyio
import pytest

from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    DistributionEvidenceScopeIdentityV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    EvidenceStringValueV1,
    ReportScopeProductsCanonicalV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.hashing import evidence_scope_ref
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.adapter.report_scope import (
    ReportScopeEvidenceService,
)
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from tests.unit.evidence._report_scope_evidence_fixtures import (
    FakeReportScopeClient,
    report_inputs,
)


def test_collect_returns_typed_canonical_report_facts_with_bound_provenance() -> None:
    # Given: a V2 product-list plan and authoritative adapter preflight metadata.
    source = report_inputs()
    client = FakeReportScopeClient()
    service = ReportScopeEvidenceService(adapter_client=client)

    # When: report-scope evidence is collected through the integration boundary.
    async def collect():
        return await service.collect(
            source.request,
            source.plan,
            source.policy,
            source.preflight,
        )

    facts = anyio.run(collect)

    # Then: transport payloads become canonical facts without metadata dictionaries.
    assert [call.command for call in client.calls] == ["summary", "list_products"]
    assert tuple(fact.fact_type for fact in facts) == (
        "report_scope_summary",
        "report_scope_products",
    )
    assert all(fact.contract_version == "canonical-evidence-fact.v1" for fact in facts)
    assert all(
        fact.provenance.scope_ref == evidence_scope_ref(fact.scope) for fact in facts
    )
    for fact in facts:
        assert isinstance(fact.scope, DistributionEvidenceScopeIdentityV1)
        assert fact.scope.business_scope_ref == source.scope.business_scope_ref
    payload = facts[1].report_payload
    assert isinstance(payload, ReportScopeProductsCanonicalV1)
    assert payload.products[0].product_name == "Product1"


def test_collect_keeps_product_paging_bounded_and_marks_partial_projection() -> None:
    # Given: more products than the canonical 200-item projection ceiling.
    source = report_inputs()
    pages = {
        page: tuple(
            f"Product{index:03d}" for index in range((page - 1) * 50, page * 50)
        )
        for page in range(1, 5)
    }
    client = FakeReportScopeClient(product_pages=pages, product_total_count=201)
    service = ReportScopeEvidenceService(adapter_client=client)

    # When: all bounded pages are collected.
    async def collect():
        return await service.collect(
            source.request, source.plan, source.policy, source.preflight
        )

    facts = anyio.run(collect)

    # Then: only four pages are fetched and the typed payload declares incompleteness.
    assert [(call.command, call.page) for call in client.calls] == [
        ("summary", None),
        ("list_products", 1),
        ("list_products", 2),
        ("list_products", 3),
        ("list_products", 4),
    ]
    payload = facts[1].report_payload
    assert isinstance(payload, ReportScopeProductsCanonicalV1)
    assert payload.returned_count == 200
    assert payload.total_count == 201
    assert payload.full_product_list_in_projection is False


def test_execute_v2_admits_report_products_into_the_matching_unit_grounding() -> None:
    # Given: the real canonical executor with fake adapter transports.
    source = report_inputs()
    report_service = ReportScopeEvidenceService(adapter_client=FakeReportScopeClient())

    class Preflight:
        async def collect(
            self,
            request: KernelReplyRequestV1,
            resolve_types: list[AdapterResolveType] | None = None,
            resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
        ) -> AdapterPreflightSnapshot:
            del request, resolve_types, resolve_material_pack_options
            return source.preflight

    executor = EvidenceExecutor(Preflight(), report_scope_service=report_service)

    # When: the V2 plan executes end to end without a legacy fact adapter.
    async def execute():
        return await executor.execute_v2(
            source.request,
            source.plan,
            source.policy,
            scope_authority=source.scope,
        )

    result = anyio.run(execute)

    # Then: the report product fact is the only evidence admitted by that manifest.
    grounding = result.groundings[0]
    assert {fact.fact_type for fact in result.canonical_facts} == {
        "weekly_report_resolvable",
        "report_scope_summary",
        "report_scope_products",
    }
    assert tuple(fact.fact_type for fact in grounding.allowed_evidence) == (
        "report_scope_products",
    )
    assert grounding.business_facts.evidence_fact_count == 1
    assert grounding.business_facts.user_permission == "allowed"


@pytest.mark.parametrize(
    ("response_overrides", "reason_code"),
    [
        (
            {"material_type": "monthly"},
            "adapter_report_scope_material_type_mismatch",
        ),
        ({"dist_name": "OtherDist"}, "adapter_report_scope_distribution_mismatch"),
        ({"period": "19990101"}, "adapter_report_scope_period_mismatch"),
    ],
    ids=["wrong-material-type", "wrong-distribution", "wrong-period"],
)
def test_execute_v2_rejects_report_scope_response_outside_issued_target(
    response_overrides: dict[str, str],
    reason_code: str,
) -> None:
    # Given: a valid weekly report plan and a parsed adapter response for another scope.
    source = report_inputs()
    report_service = ReportScopeEvidenceService(
        adapter_client=FakeReportScopeClient(response_overrides=response_overrides)
    )

    class Preflight:
        async def collect(
            self,
            request: KernelReplyRequestV1,
            resolve_types: list[AdapterResolveType] | None = None,
            resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
        ) -> AdapterPreflightSnapshot:
            del request, resolve_types, resolve_material_pack_options
            return source.preflight

    executor = EvidenceExecutor(Preflight(), report_scope_service=report_service)

    # When: the response crosses the real report fact and evidence-admission seam.
    async def execute():
        return await executor.execute_v2(
            source.request,
            source.plan,
            source.policy,
            scope_authority=source.scope,
        )

    result = anyio.run(execute)

    # Then: the mismatch is typed unavailable and cannot enter unit grounding.
    unavailable = next(
        fact
        for fact in result.canonical_facts
        if fact.fact_type == "report_scope_unavailable"
    )
    assert isinstance(unavailable.value, EvidenceStringValueV1)
    assert unavailable.value.value == reason_code
    assert "report_scope_products" not in {
        fact.fact_type for fact in result.canonical_facts
    }
    assert result.groundings[0].allowed_evidence == ()
