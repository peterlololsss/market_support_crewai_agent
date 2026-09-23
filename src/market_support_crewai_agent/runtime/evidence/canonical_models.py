from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from market_support_crewai_agent.runtime.evidence import (
    canonical_identity,
    canonical_values,
)
from market_support_crewai_agent.runtime.hashing import (
    evidence_fact_id,
    evidence_provenance_hash,
    evidence_scope_ref,
    evidence_value_hash,
    public_evidence_url_hash,
)

CanonicalEvidenceFactTypeV1 = Literal[
    "material_pack_resolvable",
    "weekly_report_resolvable",
    "monthly_report_resolvable",
    "sales_mention_resolvable",
    "report_period",
    "report_scope_summary",
    "report_scope_match",
    "report_scope_products",
    "report_scope_unavailable",
    "recent_executed_action",
    "document_context",
    "document_context_unavailable",
]
CanonicalEvidenceSourceTypeV1 = Literal[
    "adapter_resolve",
    "adapter_report_scope",
    "action_ledger",
    "document_mcp",
    "approved_static_knowledge",
    "conversation_history",
    "user_upload",
    "current_artifact",
    "adapter_context",
    "user_message",
    "assistant_message",
    "history_summary",
    "retrieved_doc",
    "tool_result",
]
CanonicalEvidenceArtifactTypeV1 = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "document_context",
    "adapter_context",
    "history",
    "user_upload",
    "unknown",
]


class CanonicalEvidenceFactV1(canonical_values.CanonicalModelV1):
    contract_version: Literal["canonical-evidence-fact.v1"] = (
        "canonical-evidence-fact.v1"
    )
    evidence_id: str = Field(pattern=r"^eid1:[0-9a-f]{64}$")
    fact_type: CanonicalEvidenceFactTypeV1
    source_type: CanonicalEvidenceSourceTypeV1
    artifact_type: CanonicalEvidenceArtifactTypeV1
    resolve_type: (
        Literal["material_pack", "weekly_report", "monthly_report", "sales_mention"]
        | None
    ) = None
    value: canonical_values.EvidenceValueCanonicalV1
    scope: canonical_identity.EvidenceScopeIdentityV1
    provenance: canonical_identity.EvidenceProvenanceCanonicalV1
    observed_at_epoch_seconds: int | None = Field(default=None, ge=0)
    public_urls: tuple[str, ...] = Field(default=(), max_length=32)
    report_payload: canonical_values.CanonicalEvidenceReportPayloadV1 | None = None
    recent_executed_action: (
        canonical_values.RecentExecutedActionEvidencePayloadCanonicalV1 | None
    ) = None

    @model_validator(mode="after")
    def _validate_payload_shape(self) -> CanonicalEvidenceFactV1:
        if self.artifact_type != self.scope.artifact_type:
            raise ValueError("canonical_evidence_artifact_scope_mismatch")
        report_fact_types = {
            "report_scope_summary": canonical_values.ReportScopeSummaryCanonicalV1,
            "report_scope_match": canonical_values.ReportScopeMatchCanonicalV1,
            "report_scope_products": canonical_values.ReportScopeProductsCanonicalV1,
        }
        expected = report_fact_types.get(self.fact_type)
        if expected is None:
            if self.report_payload is not None:
                raise ValueError("canonical_evidence_unexpected_report_payload")
        else:
            if not isinstance(self.value, canonical_values.EvidenceNullValueV1):
                raise ValueError("canonical_evidence_report_value_must_be_null")
            if not isinstance(self.report_payload, expected):
                raise ValueError("canonical_evidence_report_payload_mismatch")
        if self.fact_type == "recent_executed_action":
            if not isinstance(self.value, canonical_values.EvidenceNullValueV1):
                raise ValueError("canonical_evidence_action_value_must_be_null")
            if self.recent_executed_action is None:
                raise ValueError("canonical_evidence_action_payload_required")
        elif self.recent_executed_action is not None:
            raise ValueError("canonical_evidence_unexpected_action_payload")
        return self


class CanonicalResolveBindingV1(canonical_values.CanonicalModelV1):
    contract_version: Literal["canonical-resolve-binding.v1"] = (
        "canonical-resolve-binding.v1"
    )
    evidence_id: str = Field(pattern=r"^eid1:[0-9a-f]{64}$")
    resolve_type: Literal[
        "material_pack", "weekly_report", "monthly_report", "sales_mention"
    ]
    source_record_ref: str = Field(pattern=r"^esr1:[0-9a-f]{64}$")
    scope_ref: str = Field(pattern=r"^escope1:[0-9a-f]{64}$")
    resolve_ref: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:~-]{0,159}$",
    )
    period: str | None = Field(default=None, min_length=1, max_length=40)
    report_date: str | None = Field(default=None, min_length=1, max_length=40)

    @field_validator("resolve_ref")
    @classmethod
    def _validate_resolve_ref(cls, value: str) -> str:
        if "/" in value or "\\" in value or "://" in value or ".." in value:
            raise ValueError("canonical_resolve_binding_invalid_ref")
        return value

    @model_validator(mode="after")
    def _validate_report_binding(self) -> CanonicalResolveBindingV1:
        if self.resolve_type in {
            "material_pack",
            "sales_mention",
        } and (self.period is not None or self.report_date is not None):
            raise ValueError("canonical_resolve_binding_non_report_metadata")
        return self


def build_canonical_evidence_fact_v1(
    *,
    fact_type: CanonicalEvidenceFactTypeV1,
    source_type: CanonicalEvidenceSourceTypeV1,
    artifact_type: CanonicalEvidenceArtifactTypeV1,
    resolve_type: Literal[
        "material_pack", "weekly_report", "monthly_report", "sales_mention"
    ]
    | None,
    payload: canonical_values.EvidencePayloadCanonicalV1,
    scope: canonical_identity.EvidenceScopeIdentityV1,
    provenance: canonical_identity.EvidenceProvenanceCanonicalV1,
    observed_at_epoch_seconds: int | None,
    public_urls: tuple[str, ...] = (),
) -> CanonicalEvidenceFactV1:
    scope_ref = evidence_scope_ref(scope)
    if provenance.scope_ref != scope_ref:
        raise ValueError("canonical_evidence_provenance_scope_mismatch")
    if observed_at_epoch_seconds != provenance.as_of_epoch_seconds:
        raise ValueError("canonical_evidence_observed_at_mismatch")

    hashed_urls = tuple((public_evidence_url_hash(url), url) for url in public_urls)
    canonical_urls = tuple(sorted(hashed_urls, key=lambda item: item[0]))
    canonical_url_hashes = tuple(item[0] for item in canonical_urls)
    if len(set(canonical_url_hashes)) != len(canonical_url_hashes):
        raise ValueError("canonical_evidence_duplicate_public_url")
    if provenance.public_url_hashes != canonical_url_hashes:
        raise ValueError("canonical_evidence_provenance_public_urls_mismatch")

    value, report_payload, recent_executed_action = _canonical_fact_payload_fields_v1(
        fact_type=fact_type,
        payload=payload,
    )
    identity = canonical_identity.EvidenceFactIdentityV1(
        source_class=provenance.source_class,
        source_record_ref=provenance.source_record_ref,
        fact_type=fact_type,
        artifact_type=artifact_type,
        scope_ref=scope_ref,
        as_of_epoch_seconds=provenance.as_of_epoch_seconds,
        value_hash=evidence_value_hash(payload),
        provenance_hash=evidence_provenance_hash(provenance),
    )
    return CanonicalEvidenceFactV1(
        evidence_id=evidence_fact_id(identity),
        fact_type=fact_type,
        source_type=source_type,
        artifact_type=artifact_type,
        resolve_type=resolve_type,
        value=value,
        scope=scope,
        provenance=provenance,
        observed_at_epoch_seconds=observed_at_epoch_seconds,
        public_urls=tuple(item[1] for item in canonical_urls),
        report_payload=report_payload,
        recent_executed_action=recent_executed_action,
    )


def _canonical_fact_payload_fields_v1(
    *,
    fact_type: CanonicalEvidenceFactTypeV1,
    payload: canonical_values.EvidencePayloadCanonicalV1,
) -> tuple[
    canonical_values.EvidenceValueCanonicalV1,
    canonical_values.CanonicalEvidenceReportPayloadV1 | None,
    canonical_values.RecentExecutedActionEvidencePayloadCanonicalV1 | None,
]:
    match payload:
        case canonical_values.ScalarEvidencePayloadCanonicalV1(value=value):
            return value, None, None
        case canonical_values.ReportSummaryEvidencePayloadCanonicalV1(
            value=value, report_payload=report_payload
        ):
            if fact_type != "report_scope_summary":
                raise ValueError("canonical_evidence_report_payload_fact_type_mismatch")
            return value, report_payload, None
        case canonical_values.ReportMatchEvidencePayloadCanonicalV1(
            value=value, report_payload=report_payload
        ):
            if fact_type != "report_scope_match":
                raise ValueError("canonical_evidence_report_payload_fact_type_mismatch")
            return value, report_payload, None
        case canonical_values.ReportProductsEvidencePayloadCanonicalV1(
            value=value, report_payload=report_payload
        ):
            if fact_type != "report_scope_products":
                raise ValueError("canonical_evidence_report_payload_fact_type_mismatch")
            return value, report_payload, None
        case canonical_values.RecentExecutedActionEvidencePayloadCanonicalV1(
            value=value
        ):
            if fact_type != "recent_executed_action":
                raise ValueError("canonical_evidence_action_payload_fact_type_mismatch")
            return value, None, payload
