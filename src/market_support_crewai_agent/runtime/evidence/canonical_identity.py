from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, field_validator

from market_support_crewai_agent.runtime.evidence import canonical_values


class DistributionEvidenceScopeIdentityV1(canonical_values.CanonicalModelV1):
    kind: Literal["distribution"] = "distribution"
    business_scope_ref: str = Field(pattern=r"^bsr:[0-9a-f]{32}$")
    channel_kind: Literal["bank", "non_bank", "unknown"]
    product_ids: tuple[str, ...] = Field(max_length=1_000)
    artifact_type: str = Field(min_length=1, max_length=80)
    material_option: str | None = Field(default=None, max_length=80)
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=40)


class UnscopedEvidenceScopeIdentityV1(canonical_values.CanonicalModelV1):
    kind: Literal["unscoped"] = "unscoped"
    artifact_type: str = Field(min_length=1, max_length=80)
    material_option: str | None = Field(default=None, max_length=80)
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=40)


EvidenceScopeIdentityV1: TypeAlias = Annotated[
    DistributionEvidenceScopeIdentityV1 | UnscopedEvidenceScopeIdentityV1,
    Field(discriminator="kind"),
]


class EvidenceProvenanceCanonicalV1(canonical_values.CanonicalModelV1):
    source_class: str = Field(min_length=1, max_length=80)
    source_record_ref: str = Field(pattern=r"^esr1:[0-9a-f]{64}$")
    producer_contract_version: str = Field(min_length=1, max_length=120)
    retrieval_operation: str = Field(min_length=1, max_length=120)
    scope_ref: str = Field(pattern=r"^escope1:[0-9a-f]{64}$")
    as_of_epoch_seconds: int | None = Field(default=None, ge=0)
    public_url_hashes: tuple[str, ...] = Field(max_length=32)

    @field_validator("public_url_hashes")
    @classmethod
    def canonical_public_urls(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.startswith("puh1:") or len(value) != 69 for value in values):
            raise ValueError("invalid_public_url_hash")
        if len(set(values)) != len(values):
            raise ValueError("duplicate_public_url_hash")
        return tuple(sorted(values))


class ApprovedStaticEvidenceSourceRecordKeyV1(canonical_values.CanonicalModelV1):
    kind: Literal["approved_static"] = "approved_static"
    catalog_version: str = Field(min_length=1, max_length=120)
    canonical_id: str = Field(min_length=1, max_length=160)


class DocumentMcpEvidenceSourceRecordKeyV1(canonical_values.CanonicalModelV1):
    kind: Literal["document_mcp"] = "document_mcp"
    client_contract_version: str = Field(min_length=1, max_length=120)
    corpus_version: str = Field(min_length=1, max_length=120)
    document_id: str = Field(min_length=1, max_length=160)
    entry_id: str = Field(min_length=1, max_length=160)


class AdapterEvidenceSourceRecordKeyV1(canonical_values.CanonicalModelV1):
    kind: Literal["adapter"] = "adapter"
    adapter_service_id: str = Field(min_length=1, max_length=120)
    adapter_contract_version: str = Field(min_length=1, max_length=120)
    adapter_build_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    resolve_type: Literal[
        "material_pack", "weekly_report", "monthly_report", "sales_mention"
    ]
    opaque_resolve_ref: str = Field(min_length=1, max_length=160)


class ActionLedgerEvidenceSourceRecordKeyV1(canonical_values.CanonicalModelV1):
    kind: Literal["action_ledger"] = "action_ledger"
    state_key_ref: str = Field(pattern=r"^csk1:[0-9a-f]{64}$")
    action_id: str = Field(min_length=1, max_length=160)
    status_revision: int = Field(ge=1)


class ConversationEvidenceSourceRecordKeyV1(canonical_values.CanonicalModelV1):
    kind: Literal["conversation"] = "conversation"
    state_key_ref: str = Field(pattern=r"^csk1:[0-9a-f]{64}$")
    turn_ordinal: int = Field(ge=0)
    role: Literal["user", "assistant"]


class StateOpaqueEvidenceSourceRecordKeyV1(canonical_values.CanonicalModelV1):
    kind: Literal["user_upload", "current_artifact"]
    state_key_ref: str = Field(pattern=r"^csk1:[0-9a-f]{64}$")
    opaque_record_ref: str = Field(min_length=1, max_length=160)


class ToolResultEvidenceSourceRecordKeyV1(canonical_values.CanonicalModelV1):
    kind: Literal["tool_result"] = "tool_result"
    evidence_command_hash: str = Field(pattern=r"^ech1:[0-9a-f]{64}$")
    item_ordinal: int = Field(ge=0)


EvidenceSourceRecordKeyV1: TypeAlias = Annotated[
    ApprovedStaticEvidenceSourceRecordKeyV1
    | DocumentMcpEvidenceSourceRecordKeyV1
    | AdapterEvidenceSourceRecordKeyV1
    | ActionLedgerEvidenceSourceRecordKeyV1
    | ConversationEvidenceSourceRecordKeyV1
    | StateOpaqueEvidenceSourceRecordKeyV1
    | ToolResultEvidenceSourceRecordKeyV1,
    Field(discriminator="kind"),
]


class EvidenceFactIdentityV1(canonical_values.CanonicalModelV1):
    source_class: str = Field(min_length=1, max_length=80)
    source_record_ref: str = Field(pattern=r"^esr1:[0-9a-f]{64}$")
    fact_type: str = Field(min_length=1, max_length=120)
    artifact_type: str = Field(min_length=1, max_length=80)
    scope_ref: str = Field(pattern=r"^escope1:[0-9a-f]{64}$")
    as_of_epoch_seconds: int | None = Field(default=None, ge=0)
    value_hash: str = Field(pattern=r"^evh1:[0-9a-f]{64}$")
    provenance_hash: str = Field(pattern=r"^eph1:[0-9a-f]{64}$")
