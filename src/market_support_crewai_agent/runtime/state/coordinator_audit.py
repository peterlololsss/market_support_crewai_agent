from secrets import token_bytes
from typing import Literal

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.audit_program_records import (
    DirectProgramAuditV2,
    GroupProgramAuditV2,
    LlmInvocationAuditProposalV1,
)
from market_support_crewai_agent.runtime.state.audit_records import (
    DirectAuditCommitProposalV1,
    DirectAuditRecordV1,
    GroupAuditCommitProposalV1,
    GroupAuditRecordV1,
)
from market_support_crewai_agent.runtime.state.audit_types import (
    ReplyStateJournalOperationV1,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
    ReplyStateJournalV2,
)
from market_support_crewai_agent.schemas.type_ids import ReplyKind


def build_journal(
    *,
    operation: ReplyStateJournalOperationV1,
    initiating_state_key: ConversationStateKey | None,
    affected_state_keys: tuple[ConversationStateKey, ...],
    prior_root: CoordinatorStateRootV1,
    candidate_root: CoordinatorStateRootV1,
) -> ReplyStateJournalV2:
    return ReplyStateJournalV2(
        transaction_id=token_bytes(16),
        operation=operation,
        initiating_state_key=initiating_state_key,
        affected_state_keys=affected_state_keys,
        prior_root=prior_root,
        candidate_root=candidate_root,
        outcome="staging",
    )


def build_audit_record(
    proposal: GroupAuditCommitProposalV1 | DirectAuditCommitProposalV1,
    *,
    state_key: ConversationStateKey,
    response_id: str,
    request_id: str,
    reply_kind: ReplyKind,
    created_at_epoch_ms: int,
    expires_at_monotonic_ns: int,
) -> GroupAuditRecordV1 | DirectAuditRecordV1:
    match proposal:
        case GroupAuditCommitProposalV1():
            programs = tuple(
                GroupProgramAuditV2(
                    ordinal=item.ordinal,
                    stage=item.stage,
                    program_id=item.program_id,
                    program_version=_required_group_field(
                        item.program_version, "program_version"
                    ),
                    target_slot=item.target_slot,
                    scene_key=_required_group_scene(item.scene_key),
                    scene_contract_ref=item.scene_contract_ref,
                    model_family=_required_group_field(
                        item.model_family, "model_family"
                    ),
                    provider_id=_required_group_field(item.provider_id, "provider_id"),
                    transport_id=_required_group_field(
                        item.transport_id, "transport_id"
                    ),
                    input_schema_version=_required_group_field(
                        item.input_schema_version, "input_schema_version"
                    ),
                    output_schema_version=_required_group_field(
                        item.output_schema_version, "output_schema_version"
                    ),
                    status=item.status,
                    latency_ms=item.latency_ms,
                    input_bytes=item.input_bytes,
                    output_bytes=item.output_bytes,
                    error_code=item.error_code,
                    osh1=_required_group_field(item.osh1, "osh1"),
                    poh1=_required_group_field(item.poh1, "poh1"),
                    hph1=_required_group_field(item.hph1, "hph1"),
                    prh1=_required_group_field(item.prh1, "prh1"),
                    input_digest=_required_group_field(
                        item.input_digest, "input_digest"
                    ),
                    output_digest=item.output_digest,
                    logical_attempt=item.logical_attempt,
                    transport_attempt=item.transport_attempt,
                )
                for item in proposal.program_attempts
            )
            return GroupAuditRecordV1(
                state_key=state_key,
                response_id=response_id,
                request_id=request_id,
                reply_kind=reply_kind,
                reason_code=proposal.reason_code,
                manifest_refs=proposal.manifest_refs,
                programs=programs,
                created_at_epoch_ms=created_at_epoch_ms,
                expires_at_monotonic_ns=expires_at_monotonic_ns,
            )
        case DirectAuditCommitProposalV1():
            programs = tuple(
                _direct_program_audit(item) for item in proposal.program_attempts
            )
            return DirectAuditRecordV1(
                state_key_digest=proposal.state_key_digest,
                response_id=response_id,
                request_id_digest=proposal.request_id_digest,
                reply_kind=reply_kind,
                reason_code=proposal.reason_code,
                programs=programs,
                content_digests=proposal.content_digests,
                lengths=proposal.lengths,
                manifest_refs=proposal.manifest_refs,
                dependency_counts=proposal.dependency_counts,
                static_registry_id=proposal.static_registry_id,
                static_registry_version=proposal.static_registry_version,
                created_at_epoch_ms=created_at_epoch_ms,
                expires_at_monotonic_ns=expires_at_monotonic_ns,
                redaction_flags=(
                    "raw_request_omitted",
                    "raw_identity_omitted",
                    "raw_content_omitted",
                    "unkeyed_hashes_omitted",
                    "locators_omitted",
                    "provider_errors_omitted",
                    "marker_names_omitted",
                ),
            )


def _required_group_field(value: str | None, field_name: str) -> str:
    if value is None:
        raise CoordinatorError(f"group_audit_{field_name}_required")
    return value


def _required_group_scene(
    value: Literal["wecom_group.v1", "wecom_direct.v1", "scene_neutral.v1"] | None,
) -> Literal["wecom_group.v1", "scene_neutral.v1"]:
    match value:
        case "wecom_group.v1" | "scene_neutral.v1":
            return value
        case "wecom_direct.v1" | None:
            raise CoordinatorError("group_audit_program_scene_invalid")


def _required_direct_field(value: str | None) -> str:
    if value is None:
        raise CoordinatorError("direct_audit_program_metadata_required")
    return value


def _required_direct_scene(
    value: Literal["wecom_group.v1", "wecom_direct.v1", "scene_neutral.v1"] | None,
) -> Literal["wecom_direct.v1", "scene_neutral.v1"]:
    match value:
        case "wecom_direct.v1" | "scene_neutral.v1":
            return value
        case "wecom_group.v1" | None:
            raise CoordinatorError("direct_audit_program_scene_invalid")


def _direct_program_audit(item: LlmInvocationAuditProposalV1) -> DirectProgramAuditV2:
    return DirectProgramAuditV2(
        ordinal=item.ordinal,
        logical_attempt=item.logical_attempt,
        transport_attempt=item.transport_attempt,
        stage=item.stage,
        program_id=item.program_id,
        program_version=_required_direct_field(item.program_version),
        target_slot=item.target_slot,
        scene_key=_required_direct_scene(item.scene_key),
        scene_contract_ref=item.scene_contract_ref,
        model_family=_required_direct_field(item.model_family),
        provider_id=_required_direct_field(item.provider_id),
        transport_id=_required_direct_field(item.transport_id),
        input_schema_version=_required_direct_field(item.input_schema_version),
        output_schema_version=_required_direct_field(item.output_schema_version),
        status=item.status,
        latency_ms=item.latency_ms,
        input_bytes=item.input_bytes,
        output_bytes=(
            item.output_bytes
            if item.status in {"success", "validation_error"}
            else None
        ),
        error_code=item.error_code,
        input_digest=_required_direct_field(item.input_digest),
        output_digest=item.output_digest,
    )
