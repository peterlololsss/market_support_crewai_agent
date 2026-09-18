from __future__ import annotations

import unicodedata
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from market_support_crewai_agent.schemas.base import MetricCount, StrictModel
from market_support_crewai_agent.schemas.conversation import (
    validate_canonical_tenant_ref,
)
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveStatus,
    AdapterResolveType,
)


class AdapterCapabilityEndpoints(StrictModel):
    health: str
    capabilities: str
    metrics: str
    resolve: str
    batch_resolve: str
    report_scope: str | None = None


class AdapterCapabilityAuth(StrictModel):
    header_schemes: list[str] = Field(default_factory=list)
    protected_endpoints: list[str] = Field(default_factory=list)


class _AdapterCapabilityValidationError(ValueError):
    def __init__(self, field_name: str, reason: str) -> None:
        self.field_name: str = field_name
        self.reason: str = reason
        super().__init__(f"{field_name} {reason}")


class AdapterCapabilities(StrictModel):
    service: Literal["assistant-wecom-market-agent-adapter"]
    contract_version: Literal["adapter-resolve"]
    batch_contract_version: Literal["adapter-resolve-batch"]
    action_contract_version: Literal["adapter-action"]
    endpoints: AdapterCapabilityEndpoints
    resolve_types: list[AdapterResolveType]
    statuses: list[AdapterResolveStatus]
    max_batch_requests: int = Field(gt=0)
    max_request_body_bytes: int = Field(gt=0)
    cache_ttl_seconds: float = Field(default=0, ge=0)
    cache_max_entries: int = Field(default=0, ge=0)
    auth: AdapterCapabilityAuth | None = None
    supported_scenes: list[Literal["direct", "group"]] | None = Field(
        default=None,
        max_length=2,
    )
    reply_request_contract_versions: list[str] | None = Field(
        default=None,
        max_length=16,
    )
    action_feedback_contract_versions: list[str] | None = Field(
        default=None,
        max_length=16,
    )
    conversation_identity_contract_versions: list[str] | None = Field(
        default=None,
        max_length=16,
    )
    deployment_tenant_ref: str | None = None

    @field_validator("supported_scenes")
    @classmethod
    def canonicalize_supported_scenes(
        cls,
        values: list[Literal["direct", "group"]] | None,
    ) -> list[Literal["direct", "group"]] | None:
        if values is None:
            return None
        if len(set(values)) != len(values):
            raise _AdapterCapabilityValidationError(
                "supported_scenes",
                "must not contain duplicates",
            )
        return sorted(values)

    @field_validator(
        "reply_request_contract_versions",
        "action_feedback_contract_versions",
        "conversation_identity_contract_versions",
    )
    @classmethod
    def canonicalize_adapter_contract_versions(
        cls,
        values: list[str] | None,
        info: ValidationInfo,
    ) -> list[str] | None:
        if values is None:
            return None
        field_name = info.field_name or "adapter_contract_versions"
        for value in values:
            if (
                not value
                or len(value) > 80
                or value != value.strip()
                or any(
                    char.isspace() or unicodedata.category(char) == "Cc"
                    for char in value
                )
            ):
                raise _AdapterCapabilityValidationError(
                    field_name,
                    "entries must be canonical strings of 1-80 chars",
                )
        if len(set(values)) != len(values):
            raise _AdapterCapabilityValidationError(
                field_name,
                "must not contain duplicates",
            )
        return sorted(values)

    @field_validator("deployment_tenant_ref")
    @classmethod
    def validate_deployment_tenant_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_canonical_tenant_ref(
            value,
            field_name="deployment_tenant_ref",
        )


class AdapterCacheMetrics(StrictModel):
    ttl_seconds: float = Field(ge=0)
    max_entries: int = Field(ge=0)
    entries: int = Field(ge=0)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    sets: int = Field(ge=0)
    expired: int = Field(ge=0)
    evictions: int = Field(ge=0)


class AdapterResolverMetrics(StrictModel):
    cache: AdapterCacheMetrics


class AdapterDurationMetrics(StrictModel):
    count: MetricCount
    total: float = Field(ge=0)
    max: float = Field(ge=0)


class AdapterTransportRouteMetrics(StrictModel):
    requests: MetricCount
    errors: MetricCount
    status_codes: dict[str, MetricCount] = Field(default_factory=dict)
    duration_ms: AdapterDurationMetrics


class AdapterTransportMetrics(StrictModel):
    requests_total: MetricCount
    inflight_requests: MetricCount
    errors_total: MetricCount
    status_codes: dict[str, MetricCount] = Field(default_factory=dict)
    duration_ms: AdapterDurationMetrics
    routes: dict[str, AdapterTransportRouteMetrics] = Field(default_factory=dict)


class AdapterMetrics(StrictModel):
    service: Literal["assistant-wecom-market-agent-adapter"]
    uptime_seconds: int = Field(ge=0)
    resolver: AdapterResolverMetrics
    transport: AdapterTransportMetrics
