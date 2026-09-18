from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.context import models as context_models
from market_support_crewai_agent.runtime.context import projection
from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    DistributionEvidenceScopeIdentityV1,
    EvidenceProvenanceCanonicalV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
    build_canonical_evidence_fact_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    EvidenceContentValueV1,
    EvidenceNullValueV1,
    ReportProductsEvidencePayloadCanonicalV1,
    ReportScopeProductCanonicalV1,
    ReportScopeProductsCanonicalV1,
    ScalarEvidencePayloadCanonicalV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.hashing import (
    evidence_scope_ref,
    public_evidence_url_hash,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.conversation import DistributionScopeV1


def _authority() -> BusinessScopeAuthorityV1:
    return business_scope_authority_v1(
        DistributionScopeV1(
            kind="distribution",
            dist_channel_name="Safe Channel",
            channel_type="bank",
            available_artifacts=[],
        )
    )


def _context(
    authority: BusinessScopeAuthorityV1,
) -> context_models.GroundingProjectionContextV1:
    return context_models.GroundingProjectionContextV1(
        business_scope_authority=authority,
        locator_safety=LocatorSafetyClassifierV1(
            internal_origins=frozenset(),
            secrets=("never-project-this-secret",),
        ),
        evaluation_epoch_seconds=1_000,
    )


def _provenance(
    scope: DistributionEvidenceScopeIdentityV1,
    public_urls: tuple[str, ...],
    observed_at: int,
) -> EvidenceProvenanceCanonicalV1:
    return EvidenceProvenanceCanonicalV1(
        source_class="document_mcp",
        source_record_ref="esr1:" + "c" * 64,
        producer_contract_version="provider-contract.v1",
        retrieval_operation="private-provider-operation",
        scope_ref=evidence_scope_ref(scope),
        as_of_epoch_seconds=observed_at,
        public_url_hashes=tuple(
            sorted(public_evidence_url_hash(url) for url in public_urls)
        ),
    )


def _document_fact(
    authority: BusinessScopeAuthorityV1,
    public_url: str,
) -> CanonicalEvidenceFactV1:
    scope = DistributionEvidenceScopeIdentityV1(
        business_scope_ref=authority.business_scope_ref,
        channel_kind="bank",
        product_ids=("private-product-id",),
        artifact_type="document_context",
    )
    return build_canonical_evidence_fact_v1(
        fact_type="document_context",
        source_type="document_mcp",
        artifact_type="document_context",
        resolve_type=None,
        payload=ScalarEvidencePayloadCanonicalV1(
            value=EvidenceContentValueV1(
                media_type="text/markdown",
                text="bounded selected answer",
            )
        ),
        scope=scope,
        provenance=_provenance(scope, (public_url,), 900),
        observed_at_epoch_seconds=900,
        public_urls=(public_url,),
    )


def _report_fact(
    authority: BusinessScopeAuthorityV1,
) -> CanonicalEvidenceFactV1:
    scope = DistributionEvidenceScopeIdentityV1(
        business_scope_ref=authority.business_scope_ref,
        channel_kind="bank",
        product_ids=("report-private-product-id",),
        artifact_type="weekly_report",
        period="2026-W28",
        report_date="2026-07-17",
    )
    product = ReportScopeProductCanonicalV1(
        product_name="Visible Product",
        portfolio_type="market-neutral",
        report_section="Section A",
        source_pdf_status="found",
        final_report_status="generated",
    )
    return build_canonical_evidence_fact_v1(
        fact_type="report_scope_products",
        source_type="adapter_report_scope",
        artifact_type="weekly_report",
        resolve_type="weekly_report",
        payload=ReportProductsEvidencePayloadCanonicalV1(
            value=EvidenceNullValueV1(),
            report_payload=ReportScopeProductsCanonicalV1(
                available=True,
                reason_code="ok",
                period="2026-W28",
                report_date="2026-07-17",
                products=(product,),
                page=1,
                page_size=50,
                total_count=2,
                returned_count=1,
                products_are_paginated=True,
                full_product_list_in_projection=False,
            ),
        ),
        scope=scope,
        provenance=_provenance(scope, (), 950),
        observed_at_epoch_seconds=950,
    )


def test_scalar_evidence_projection_removes_private_provenance_and_scope_ids() -> None:
    # Given: canonical evidence contains private scope, product, and provider refs.
    authority = _authority()
    fact = _document_fact(authority, "https://example.com/public/report")

    # When: the canonical fact crosses the model-visible projection boundary.
    view = projection.project_evidence_fact_view_v1(fact, _context(authority))
    payload = view.model_dump(mode="json")
    rendered = str(payload)

    # Then: safe evidence remains while every private identity/locator is absent.
    assert payload["scope"]["product_count"] == 1
    assert payload["observed_age_seconds"] == 100
    assert payload["public_urls"] == ["https://example.com/public/report"]
    assert authority.business_scope_ref not in rendered
    assert "private-product-id" not in rendered
    assert "esr1:" not in rendered
    assert "private-provider-operation" not in rendered


def test_report_evidence_projection_uses_typed_payload_and_keeps_incomplete_flag() -> (
    None
):
    # Given: a canonical report fact whose projected product list is incomplete.
    authority = _authority()
    fact = _report_fact(authority)

    # When: the report fact is projected.
    view = projection.project_evidence_fact_view_v1(fact, _context(authority))

    # Then: no scalar value is exposed and report completeness stays false.
    assert view.value is None
    assert isinstance(
        view.report_payload,
        context_models.ReportScopeProductsViewV1,
    )
    assert view.report_payload.full_product_list_in_projection is False
    assert tuple(product.product_name for product in view.report_payload.products) == (
        "Visible Product",
    )


def test_evidence_projection_rejects_internal_public_url_canary() -> None:
    # Given: a canonical URL field containing a loopback locator canary.
    authority = _authority()
    fact = _document_fact(authority, "http://127.0.0.1/private")

    # When/Then: projection fails closed instead of leaking the locator.
    with pytest.raises(ValueError, match="evidence_public_url_not_safe"):
        projection.project_evidence_fact_view_v1(fact, _context(authority))


def test_evidence_view_rejects_raw_provider_field() -> None:
    # Given: a valid projected evidence payload plus a raw provider locator field.
    authority = _authority()
    payload = projection.project_evidence_fact_view_v1(
        _document_fact(authority, "https://example.com/public/report"),
        _context(authority),
    ).model_dump(mode="python")
    payload["source_record_ref"] = "esr1:" + "d" * 64

    # When/Then: the strict DTO rejects the unowned privacy field.
    with pytest.raises(ValidationError):
        context_models.EvidenceFactViewV1.model_validate(payload)
