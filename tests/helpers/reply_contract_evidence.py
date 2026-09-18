from __future__ import annotations

from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    DistributionEvidenceScopeIdentityV1,
    EvidenceProvenanceCanonicalV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
    build_canonical_evidence_fact_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    EvidenceNullValueV1,
    ReportProductsEvidencePayloadCanonicalV1,
    ReportScopeProductCanonicalV1,
    ReportScopeProductsCanonicalV1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
    ground_execution_plan_v2,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_admission import (
    document_fact,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.hashing import evidence_scope_ref
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.models import (
    DistributionExecutionDomainScopeV2,
    ExecutionPlanV2,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder


def v2_evidence(
    request: KernelReplyRequestV1,
    plan: ExecutionPlanV2,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
    *,
    preflight: AdapterPreflightSnapshot | None = None,
    facts: tuple[CanonicalEvidenceFactV1, ...] = (),
) -> CanonicalEvidenceExecutionResultV1:
    canonical_facts = tuple(facts)
    return CanonicalEvidenceExecutionResultV1(
        preflight=preflight or AdapterPreflightSnapshot.empty(),
        canonical_facts=canonical_facts,
        resolve_bindings=(),
        groundings=ground_execution_plan_v2(plan, policy, canonical_facts),
        domain_context=DomainContextV1Builder().build(
            request, scope_authority=scope_authority
        ),
    )


def document_evidence_fact(plan: ExecutionPlanV2, text: str) -> CanonicalEvidenceFactV1:
    fact = document_fact(
        GatewayDocumentContextV1(document_id="doc-1", text=text), plan.units[0]
    )
    if fact is None:
        raise AssertionError("document fixture must produce evidence")
    return fact


def report_scope_products_fact(plan: ExecutionPlanV2) -> CanonicalEvidenceFactV1:
    unit = plan.units[0]
    assert isinstance(unit.scope, DistributionExecutionDomainScopeV2)
    scope = DistributionEvidenceScopeIdentityV1(
        business_scope_ref=unit.scope.business_scope_ref,
        channel_kind=unit.scope.channel_kind,
        product_ids=unit.scope.product_ids,
        artifact_type="weekly_report",
    )
    payload = ReportScopeProductsCanonicalV1(
        available=True,
        reason_code="ok",
        period="20260605",
        report_date="2026-06-05",
        products=(
            ReportScopeProductCanonicalV1(
                product_name="Product1",
                report_section="IndexPlus",
                source_pdf_status="found",
                final_report_status="generated",
            ),
            ReportScopeProductCanonicalV1(
                product_name="Product2",
                report_section="IndexPlus",
                source_pdf_status="found",
                final_report_status="generated",
            ),
        ),
        page=1,
        page_size=50,
        total_count=2,
        returned_count=2,
        full_product_list_in_projection=True,
    )
    provenance = EvidenceProvenanceCanonicalV1(
        source_class="adapter_report_scope",
        source_record_ref="esr1:" + "a" * 64,
        producer_contract_version="adapter-resolve.v1",
        retrieval_operation="report_scope_products",
        scope_ref=evidence_scope_ref(scope),
        as_of_epoch_seconds=1,
        public_url_hashes=(),
    )
    return build_canonical_evidence_fact_v1(
        fact_type="report_scope_products",
        source_type="adapter_report_scope",
        artifact_type="weekly_report",
        resolve_type="weekly_report",
        payload=ReportProductsEvidencePayloadCanonicalV1(
            value=EvidenceNullValueV1(), report_payload=payload
        ),
        scope=scope,
        provenance=provenance,
        observed_at_epoch_seconds=1,
    )
