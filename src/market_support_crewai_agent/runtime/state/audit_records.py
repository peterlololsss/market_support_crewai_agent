from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.audit_program_records import (
    DirectProgramAuditV2,
    GroupProgramAuditV2,
    LlmInvocationAuditProposalV1,
)
from market_support_crewai_agent.runtime.state.audit_types import (
    DirectAuditLengthPurposeV1,
    DirectAuditPurposeV1,
    DirectAuditReasonCodeV1,
    DirectDependencyNameV1,
    DirectRedactionFlagV1,
    ReasonCodeV1,
)
from market_support_crewai_agent.runtime.state.audit_validation import (
    COMMITTED_RESPONSE_ID,
    DIRECT_AUDIT_LENGTH_PURPOSES,
    DIRECT_AUDIT_PROGRAM_IDS,
    DIRECT_AUDIT_PURPOSES,
    DIRECT_DEPENDENCY_NAMES,
    DIRECT_REDACTION_FLAGS,
    GROUP_AUDIT_PROGRAM_IDS,
    validate_audit_manifest_refs,
    validate_audit_program_scene,
    validate_direct_audit_digest,
    validate_direct_audit_reason_code,
    validate_registered_reason_code,
    validate_registered_reply_kind,
)
from market_support_crewai_agent.schemas.type_ids import ReplyKind


@dataclass(frozen=True, slots=True)
class DirectAuditLengthV1:
    purpose: DirectAuditLengthPurposeV1
    count: int

    def __post_init__(self) -> None:
        if self.purpose not in DIRECT_AUDIT_LENGTH_PURPOSES or self.count < 0:
            raise ValueError("direct audit length is invalid")


@dataclass(frozen=True, slots=True)
class DirectDependencyCountV1:
    name: DirectDependencyNameV1
    count: int

    def __post_init__(self) -> None:
        if self.name not in DIRECT_DEPENDENCY_NAMES or self.count < 0:
            raise ValueError("direct audit dependency count is invalid")


@dataclass(frozen=True, slots=True)
class GroupAuditCommitProposalV1:
    reason_code: ReasonCodeV1
    manifest_refs: tuple[str, ...]
    program_attempts: tuple[LlmInvocationAuditProposalV1, ...]
    contract_version: Literal["group-audit-commit-proposal.v1"] = (
        "group-audit-commit-proposal.v1"
    )
    kind: Literal["group"] = "group"
    scene_key: Literal["wecom_group.v1"] = "wecom_group.v1"

    def __post_init__(self) -> None:
        validate_registered_reason_code(self.reason_code)
        validate_audit_manifest_refs(self.manifest_refs)
        if len(self.program_attempts) > 18:
            raise ValueError("group audit proposal accepts at most 18 program attempts")


@dataclass(frozen=True, slots=True)
class DirectAuditCommitProposalV1:
    state_key_digest: str
    request_id_digest: str
    reason_code: DirectAuditReasonCodeV1
    manifest_refs: tuple[str, ...]
    content_digests: tuple[tuple[DirectAuditPurposeV1, str], ...]
    lengths: tuple[DirectAuditLengthV1, ...]
    dependency_counts: tuple[DirectDependencyCountV1, ...]
    static_registry_id: str
    static_registry_version: str
    program_attempts: tuple[LlmInvocationAuditProposalV1, ...] = ()
    contract_version: Literal["direct-audit-commit-proposal.v1"] = (
        "direct-audit-commit-proposal.v1"
    )
    kind: Literal["direct"] = "direct"
    scene_key: Literal["wecom_direct.v1"] = "wecom_direct.v1"

    def __post_init__(self) -> None:
        validate_direct_audit_digest(self.state_key_digest)
        validate_direct_audit_digest(self.request_id_digest)
        validate_direct_audit_reason_code(self.reason_code)
        validate_audit_manifest_refs(self.manifest_refs)
        if len(self.program_attempts) > 18:
            raise ValueError(
                "direct audit proposal accepts at most 18 program attempts"
            )
        if len(self.content_digests) > 36:
            raise ValueError("direct audit proposal accepts at most 36 content digests")
        if len(self.lengths) > 7:
            raise ValueError("direct audit proposal accepts at most 7 lengths")
        if len(self.dependency_counts) > 12:
            raise ValueError(
                "direct audit proposal accepts at most 12 dependency counts"
            )
        for purpose, digest in self.content_digests:
            if purpose not in DIRECT_AUDIT_PURPOSES:
                raise ValueError(
                    "direct audit content digest purpose is not registered"
                )
            validate_direct_audit_digest(digest)


SanitizedAuditCommitProposalV1 = (
    GroupAuditCommitProposalV1 | DirectAuditCommitProposalV1
)


@dataclass(frozen=True, slots=True)
class GroupAuditRecordV1:
    state_key: ConversationStateKey
    response_id: str
    request_id: str
    reply_kind: ReplyKind
    reason_code: ReasonCodeV1
    manifest_refs: tuple[str, ...]
    programs: tuple[GroupProgramAuditV2, ...]
    created_at_epoch_ms: int
    expires_at_monotonic_ns: int
    contract_version: Literal["group-audit-record.v1"] = "group-audit-record.v1"
    scene: Literal["group"] = "group"
    scene_key: Literal["wecom_group.v1"] = "wecom_group.v1"

    def __post_init__(self) -> None:
        validate_registered_reply_kind(self.reply_kind)
        validate_registered_reason_code(self.reason_code)
        validate_audit_manifest_refs(self.manifest_refs)
        if len(self.programs) > 18:
            raise ValueError("group audit accepts at most 18 program records")
        validate_audit_program_scene(self.programs, GROUP_AUDIT_PROGRAM_IDS)


@dataclass(frozen=True, slots=True)
class DirectAuditRecordV1:
    state_key_digest: str
    response_id: str
    request_id_digest: str
    reply_kind: ReplyKind
    reason_code: DirectAuditReasonCodeV1
    programs: tuple[DirectProgramAuditV2, ...]
    content_digests: tuple[tuple[DirectAuditPurposeV1, str], ...]
    lengths: tuple[DirectAuditLengthV1, ...]
    manifest_refs: tuple[str, ...]
    dependency_counts: tuple[DirectDependencyCountV1, ...]
    static_registry_id: str
    static_registry_version: str
    created_at_epoch_ms: int
    expires_at_monotonic_ns: int
    redaction_flags: tuple[DirectRedactionFlagV1, ...]
    contract_version: Literal["direct-audit-record.v1"] = "direct-audit-record.v1"
    scene: Literal["direct"] = "direct"
    scene_key: Literal["wecom_direct.v1"] = "wecom_direct.v1"

    def __post_init__(self) -> None:
        validate_registered_reply_kind(self.reply_kind)
        validate_direct_audit_reason_code(self.reason_code)
        validate_direct_audit_digest(self.state_key_digest)
        if COMMITTED_RESPONSE_ID.fullmatch(self.response_id) is None:
            raise ValueError("direct audit response_id must be server-issued")
        if len(self.programs) > 18:
            raise ValueError("direct audit accepts at most 18 program records")
        if len(self.content_digests) > 36:
            raise ValueError("direct audit accepts at most 36 content digests")
        if len(self.lengths) > 7:
            raise ValueError("direct audit accepts at most 7 lengths")
        if len(self.manifest_refs) > 4:
            raise ValueError("direct audit accepts at most 4 manifest refs")
        if len(self.dependency_counts) > 12:
            raise ValueError("direct audit accepts at most 12 dependency counts")
        validate_audit_program_scene(self.programs, DIRECT_AUDIT_PROGRAM_IDS)
        validate_direct_audit_digest(self.request_id_digest)
        for purpose, digest in self.content_digests:
            if purpose not in DIRECT_AUDIT_PURPOSES:
                raise ValueError(
                    "direct audit content digest purpose is not registered"
                )
            validate_direct_audit_digest(digest)
        if self.redaction_flags != DIRECT_REDACTION_FLAGS:
            raise ValueError("direct audit redaction_flags must be canonical")


AuditRecordV1 = GroupAuditRecordV1 | DirectAuditRecordV1
