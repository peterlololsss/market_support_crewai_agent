from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, field_validator

from market_support_crewai_agent.runtime.evidence import canonical_values


class DocumentQueryEvidenceCommandV1(canonical_values.CanonicalModelV1):
    source_type: Literal["document_mcp"] = "document_mcp"
    source_version: str = Field(min_length=1, max_length=120)
    operation: Literal["document_query"] = "document_query"
    query: str = Field(min_length=1, max_length=200)
    document_ids: tuple[Annotated[str, Field(min_length=1, max_length=160)], ...] = (
        Field(
            min_length=1,
            max_length=50,
        )
    )
    page: int = Field(ge=1, le=65_535)
    page_size: int = Field(ge=1, le=50)
    max_results: int = Field(ge=1, le=50)

    @field_validator("document_ids")
    @classmethod
    def canonical_document_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("duplicate_document_id")
        return tuple(sorted(values))


class ApprovedStaticQueryEvidenceCommandV1(canonical_values.CanonicalModelV1):
    source_type: Literal["approved_static"] = "approved_static"
    source_version: str = Field(min_length=1, max_length=120)
    operation: Literal["approved_static_query"] = "approved_static_query"
    query: str = Field(min_length=1, max_length=200)
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=50)
    catalog_version: str = Field(min_length=1, max_length=120)

    @field_validator("candidate_ids")
    @classmethod
    def canonical_candidate_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("duplicate_candidate_id")
        return tuple(sorted(values))


class AdapterResolveEvidenceCommandV1(canonical_values.CanonicalModelV1):
    source_type: Literal["adapter"] = "adapter"
    source_version: str = Field(min_length=1, max_length=120)
    operation: Literal["adapter_resolve"] = "adapter_resolve"
    resolve_type: Literal[
        "material_pack", "weekly_report", "monthly_report", "sales_mention"
    ]
    material_option: str | None = Field(default=None, max_length=80)


class ReportScopeEvidenceCommandV1(canonical_values.CanonicalModelV1):
    source_type: Literal["adapter"] = "adapter"
    source_version: str = Field(min_length=1, max_length=120)
    operation: Literal["report_scope"] = "report_scope"
    material_type: Literal["weekly", "monthly"]
    command: Literal["summary", "match", "list_products"]
    period: str | None = Field(default=None, max_length=40)
    query: str | None = Field(default=None, max_length=200)
    page: int = Field(ge=1, le=65_535)
    page_size: int = Field(ge=1, le=50)


EvidenceCommandKeyV1: TypeAlias = Annotated[
    DocumentQueryEvidenceCommandV1
    | ApprovedStaticQueryEvidenceCommandV1
    | AdapterResolveEvidenceCommandV1
    | ReportScopeEvidenceCommandV1,
    Field(discriminator="operation"),
]


class ApprovedStaticCacheConfigV1(canonical_values.CanonicalModelV1):
    source_class: Literal["approved_static"] = "approved_static"
    catalog_version: str = Field(min_length=1, max_length=120)
    selector_program_id: str = Field(min_length=1, max_length=160)
    selector_program_version: str = Field(min_length=1, max_length=80)
    selector_output_schema_hash: str = Field(pattern=r"^osh1:[0-9a-f]{64}$")
    max_candidates: int = Field(ge=1, le=50)
    ttl_seconds: int = Field(gt=0)
    capacity: int = Field(gt=0)


class DocumentMcpCacheConfigV1(canonical_values.CanonicalModelV1):
    source_class: Literal["document_mcp"] = "document_mcp"
    client_contract_version: str = Field(min_length=1, max_length=120)
    corpus_version: str = Field(min_length=1, max_length=120)
    qa_endpoint_path: Literal["/qa"] = "/qa"
    request_schema_hash: str = Field(pattern=r"^osh1:[0-9a-f]{64}$")
    response_schema_hash: str = Field(pattern=r"^osh1:[0-9a-f]{64}$")
    timeout_milliseconds: int = Field(gt=0)
    max_candidates: int = Field(ge=1, le=50)
    ttl_seconds: int = Field(gt=0)
    capacity: int = Field(gt=0)


SourceCacheConfigV1: TypeAlias = Annotated[
    ApprovedStaticCacheConfigV1 | DocumentMcpCacheConfigV1,
    Field(discriminator="source_class"),
]


class EvidenceCommandCacheKeyV1(canonical_values.CanonicalModelV1):
    contract_version: Literal["evidence-command-cache-key.v1"] = (
        "evidence-command-cache-key.v1"
    )
    state_key_ref: str = Field(pattern=r"^csk1:[0-9a-f]{64}$")
    policy_id: str = Field(pattern=r"^pol1:[0-9a-f]{64}$")
    manifest_ref: str = Field(min_length=1, max_length=200)
    business_scope_hash: str = Field(pattern=r"^bsh1:[0-9a-f]{64}$")
    evidence_command_hash: str = Field(pattern=r"^ech1:[0-9a-f]{64}$")
    source_cache_config: SourceCacheConfigV1
