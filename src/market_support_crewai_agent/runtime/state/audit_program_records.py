from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.runtime.state.audit_types import (
    AuditErrorCodeV1,
    AuditInvocationStatusV1,
    AuditStageV1,
    AuditTargetSlotV1,
)
from market_support_crewai_agent.runtime.state.audit_validation import (
    validate_direct_audit_digest,
    validate_prefixed_digest,
    validate_program_audit_labels,
)


@dataclass(frozen=True, slots=True)
class LlmInvocationAuditProposalV1:
    ordinal: int
    stage: AuditStageV1
    program_id: str
    target_slot: AuditTargetSlotV1
    status: AuditInvocationStatusV1
    latency_ms: int
    input_bytes: int
    output_bytes: int | None
    error_code: AuditErrorCodeV1 | None
    logical_attempt: int = 1
    transport_attempt: int = 1
    program_version: str | None = None
    scene_key: (
        Literal["wecom_group.v1", "wecom_direct.v1", "scene_neutral.v1"] | None
    ) = None
    scene_contract_ref: str | None = None
    model_family: str | None = None
    provider_id: str | None = None
    transport_id: str | None = None
    input_schema_version: str | None = None
    output_schema_version: str | None = None
    osh1: str | None = None
    poh1: str | None = None
    hph1: str | None = None
    prh1: str | None = None
    input_digest: str | None = None
    output_digest: str | None = None
    contract_version: Literal["llm-invocation-audit-proposal.v1"] = (
        "llm-invocation-audit-proposal.v1"
    )

    def __post_init__(self) -> None:
        validate_program_audit_labels(
            ordinal=self.ordinal,
            stage=self.stage,
            program_id=self.program_id,
            target_slot=self.target_slot,
            status=self.status,
            latency_ms=self.latency_ms,
            input_bytes=self.input_bytes,
            output_bytes=self.output_bytes,
            error_code=self.error_code,
        )
        if self.logical_attempt < 1 or self.transport_attempt != 1:
            raise ValueError("audit invocation attempts are invalid")


@dataclass(frozen=True, slots=True)
class GroupProgramAuditV2:
    ordinal: int
    stage: AuditStageV1
    program_id: str
    program_version: str
    target_slot: AuditTargetSlotV1
    scene_key: Literal["wecom_group.v1", "scene_neutral.v1"]
    scene_contract_ref: str | None
    model_family: str
    provider_id: str
    transport_id: str
    input_schema_version: str
    output_schema_version: str
    status: AuditInvocationStatusV1
    latency_ms: int
    input_bytes: int
    output_bytes: int | None
    error_code: AuditErrorCodeV1 | None
    osh1: str
    poh1: str
    hph1: str
    prh1: str
    input_digest: str
    output_digest: str | None
    logical_attempt: int = 1
    transport_attempt: int = 1
    contract_version: Literal["group-program-audit.v2"] = "group-program-audit.v2"

    def __post_init__(self) -> None:
        validate_program_audit_labels(
            ordinal=self.ordinal,
            stage=self.stage,
            program_id=self.program_id,
            target_slot=self.target_slot,
            status=self.status,
            latency_ms=self.latency_ms,
            input_bytes=self.input_bytes,
            output_bytes=self.output_bytes,
            error_code=self.error_code,
        )
        if self.logical_attempt < 1 or self.transport_attempt != 1:
            raise ValueError("group program audit attempts are invalid")
        if self.scene_key == "wecom_group.v1" and self.scene_contract_ref is None:
            raise ValueError("group user-facing program requires scene contract ref")
        if self.scene_key == "scene_neutral.v1" and self.scene_contract_ref is not None:
            raise ValueError("group neutral program forbids scene contract ref")
        for value, prefix in (
            (self.osh1, "osh1"),
            (self.poh1, "poh1"),
            (self.hph1, "hph1"),
            (self.prh1, "prh1"),
            (self.input_digest, "mch1"),
        ):
            validate_prefixed_digest(value, prefix)
        if self.output_digest is not None:
            validate_prefixed_digest(self.output_digest, "out1")
        match self.status:
            case "success" | "validation_error":
                if self.output_digest is None or self.output_bytes is None:
                    raise ValueError(
                        "group successful audit requires output digest and count"
                    )
            case "output_contract_error" | "transport_error":
                if self.output_digest is not None or self.output_bytes is not None:
                    raise ValueError(
                        "group failed audit forbids output digest and count"
                    )


@dataclass(frozen=True, slots=True)
class DirectProgramAuditV2:
    ordinal: int
    logical_attempt: int
    transport_attempt: int
    stage: AuditStageV1
    program_id: str
    program_version: str
    target_slot: AuditTargetSlotV1
    scene_key: Literal["wecom_direct.v1", "scene_neutral.v1"]
    scene_contract_ref: str | None
    model_family: str
    provider_id: str
    transport_id: str
    input_schema_version: str
    output_schema_version: str
    status: AuditInvocationStatusV1
    latency_ms: int
    input_bytes: int
    output_bytes: int | None
    error_code: AuditErrorCodeV1 | None
    input_digest: str
    output_digest: str | None
    redacted: Literal[True] = True
    contract_version: Literal["direct-program-audit.v2"] = "direct-program-audit.v2"

    def __post_init__(self) -> None:
        validate_program_audit_labels(
            ordinal=self.ordinal,
            stage=self.stage,
            program_id=self.program_id,
            target_slot=self.target_slot,
            status=self.status,
            latency_ms=self.latency_ms,
            input_bytes=self.input_bytes,
            output_bytes=self.output_bytes or 0,
            error_code=self.error_code,
        )
        if self.redacted is not True:
            raise ValueError("direct program audit must be redacted")
        if self.logical_attempt < 1 or self.transport_attempt < 1:
            raise ValueError("direct program audit attempts must be positive")
        if self.scene_key == "wecom_direct.v1" and self.scene_contract_ref is None:
            raise ValueError("direct user-facing program requires scene contract ref")
        if self.scene_key == "scene_neutral.v1" and self.scene_contract_ref is not None:
            raise ValueError("neutral direct program forbids scene contract ref")
        validate_direct_audit_digest(self.input_digest)
        if self.output_digest is not None:
            validate_direct_audit_digest(self.output_digest)
        match self.status:
            case "success" | "validation_error":
                if self.output_digest is None or self.output_bytes is None:
                    raise ValueError(
                        "direct successful audit requires output digest and count"
                    )
            case "output_contract_error" | "transport_error":
                if self.output_digest is not None or self.output_bytes is not None:
                    raise ValueError(
                        "direct failed audit forbids output digest and count"
                    )
