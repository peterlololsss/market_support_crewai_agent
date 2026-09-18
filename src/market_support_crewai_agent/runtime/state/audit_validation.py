import re
from collections.abc import Sequence
from typing import Final, Protocol, get_args

from market_support_crewai_agent.runtime.state.audit_types import (
    AuditInvocationStatusV1,
    AuditStageV1,
    AuditTargetSlotV1,
    DirectAuditLengthPurposeV1,
    DirectAuditPurposeV1,
    DirectAuditReasonCodeV1,
    DirectDependencyNameV1,
    DirectRedactionFlagV1,
    ReasonCodeV1,
)
from market_support_crewai_agent.schemas.type_ids import ReplyKind


class _ProgramAudit(Protocol):
    @property
    def program_id(self) -> str: ...


REGISTERED_REASON_CODES: Final = frozenset(get_args(ReasonCodeV1))
DIRECT_AUDIT_REASON_CODES: Final = frozenset(get_args(DirectAuditReasonCodeV1))
REGISTERED_REPLY_KINDS: Final = frozenset(get_args(ReplyKind))
REGISTERED_AUDIT_STAGES: Final = frozenset(get_args(AuditStageV1))
REGISTERED_TARGET_SLOTS: Final = frozenset(get_args(AuditTargetSlotV1))
REGISTERED_AUDIT_STATUSES: Final = frozenset(get_args(AuditInvocationStatusV1))
VALIDATION_ERROR_BY_STAGE: Final = {
    "planner_intent": "planner_plan_validation_failed",
    "knowledge_composer": "knowledge_composer_reply_validation_failed",
    "smalltalk_composer": "smalltalk_composer_reply_validation_failed",
    "alignment_verifier": "alignment_verifier_verdict_validation_failed",
    "approved_knowledge_selector": "approved_selector_validation_failed",
    "document_product_selector": "document_selector_validation_failed",
}
OUTPUT_CONTRACT_ERRORS: Final = frozenset(
    {
        "provider_output_missing",
        "provider_output_type",
        "provider_output_encoding",
        "provider_output_too_large",
        "provider_output_contract",
    }
)
TRANSPORT_ERRORS: Final = frozenset(
    {
        "provider_timeout",
        "provider_auth_failed",
        "provider_rate_limited",
        "provider_http_error",
        "provider_transport_unavailable",
        "provider_internal_error",
        "direct_provider_stdio_violation",
    }
)
PROGRAM_STAGE_BY_ID: Final = {
    "planner_intent.wecom_group.v1@1": "planner_intent",
    "knowledge_composer.wecom_group.v1@1": "knowledge_composer",
    "smalltalk_composer.wecom_group.v1@1": "smalltalk_composer",
    "alignment_verifier.wecom_group.v1@1": "alignment_verifier",
    "planner_intent.wecom_direct.v1@1": "planner_intent",
    "knowledge_composer.wecom_direct.v1@1": "knowledge_composer",
    "smalltalk_composer.wecom_direct.v1@1": "smalltalk_composer",
    "alignment_verifier.wecom_direct.v1@1": "alignment_verifier",
    "approved_knowledge_selector.scene_neutral.v1@1": "approved_knowledge_selector",
    "document_product_selector.scene_neutral.v1@1": "document_product_selector",
}
GROUP_AUDIT_PROGRAM_IDS: Final = frozenset(
    {
        key
        for key in PROGRAM_STAGE_BY_ID
        if "wecom_group" in key or "scene_neutral" in key
    }
)
DIRECT_AUDIT_PROGRAM_IDS: Final = frozenset(
    {
        key
        for key in PROGRAM_STAGE_BY_ID
        if "wecom_direct" in key or "scene_neutral" in key
    }
)
DIRECT_REDACTION_FLAGS: Final[tuple[DirectRedactionFlagV1, ...]] = (
    "raw_request_omitted",
    "raw_identity_omitted",
    "raw_content_omitted",
    "unkeyed_hashes_omitted",
    "locators_omitted",
    "provider_errors_omitted",
    "marker_names_omitted",
)
DIRECT_AUDIT_PURPOSES: Final = frozenset(get_args(DirectAuditPurposeV1))
DIRECT_AUDIT_LENGTH_PURPOSES: Final = frozenset(get_args(DirectAuditLengthPurposeV1))
DIRECT_DEPENDENCY_NAMES: Final = frozenset(get_args(DirectDependencyNameV1))
COMMITTED_RESPONSE_ID: Final = re.compile(r"^resp-[0-9a-f]{32}$")


def validate_registered_reason_code(reason_code: str) -> None:
    if reason_code not in REGISTERED_REASON_CODES:
        raise ValueError("audit reason_code is not registered")


def validate_direct_audit_reason_code(reason_code: str) -> None:
    if reason_code not in DIRECT_AUDIT_REASON_CODES:
        raise ValueError("direct audit reason_code is not registered")


def validate_registered_reply_kind(reply_kind: str) -> None:
    if reply_kind not in REGISTERED_REPLY_KINDS:
        raise ValueError("audit reply_kind is not registered")


def validate_program_audit_labels(
    *,
    ordinal: int,
    stage: str,
    program_id: str,
    target_slot: str,
    status: str,
    latency_ms: int,
    input_bytes: int,
    output_bytes: int | None,
    error_code: str | None,
) -> None:
    if ordinal < 1:
        raise ValueError("audit ordinal must be positive")
    if stage not in REGISTERED_AUDIT_STAGES:
        raise ValueError("audit stage is not registered")
    if PROGRAM_STAGE_BY_ID.get(program_id) != stage:
        raise ValueError("audit program_id does not match a registered stage")
    if target_slot not in REGISTERED_TARGET_SLOTS:
        raise ValueError("audit target_slot is not registered")
    expected_target_slot = (
        "planner"
        if stage == "planner_intent"
        else (
            "selector"
            if stage in {"approved_knowledge_selector", "document_product_selector"}
            else "composer"
        )
    )
    if target_slot != expected_target_slot:
        raise ValueError("audit target_slot is not allowed for stage")
    if status not in REGISTERED_AUDIT_STATUSES:
        raise ValueError("audit status is not registered")
    if (
        latency_ms < 0
        or input_bytes < 0
        or (output_bytes is not None and output_bytes < 0)
    ):
        raise ValueError("audit counts must be non-negative")
    match status:
        case "success":
            if error_code is not None:
                raise ValueError(
                    "successful audit invocation cannot have an error_code"
                )
        case "validation_error":
            if error_code != VALIDATION_ERROR_BY_STAGE[stage]:
                raise ValueError("audit validation error_code does not match stage")
        case "output_contract_error":
            if error_code not in OUTPUT_CONTRACT_ERRORS:
                raise ValueError("audit output contract error_code is not registered")
        case "transport_error":
            if error_code not in TRANSPORT_ERRORS:
                raise ValueError("audit transport error_code is not registered")
        case unreachable:
            raise AssertionError(f"unreachable audit status: {unreachable}")


def validate_audit_program_scene(
    programs: Sequence[_ProgramAudit], allowed_program_ids: frozenset[str]
) -> None:
    if any(program.program_id not in allowed_program_ids for program in programs):
        raise ValueError("audit program_id is not registered for the audit scene")


def validate_audit_manifest_refs(manifest_refs: tuple[str, ...]) -> None:
    if len(manifest_refs) > 4 or len(set(manifest_refs)) != len(manifest_refs):
        raise ValueError("audit manifest refs are not canonical")
    if any(not ref or len(ref) > 160 for ref in manifest_refs):
        raise ValueError("audit manifest ref is invalid")


def validate_direct_audit_digest(value: str) -> None:
    if len(value) != 69 or not value.startswith("dah1:"):
        raise ValueError("direct audit digest must be dah1")
    if any(character not in "0123456789abcdef" for character in value[5:]):
        raise ValueError("direct audit digest must be dah1")


def validate_prefixed_digest(value: str, prefix: str) -> None:
    expected_prefix = f"{prefix}:"
    if len(value) != len(expected_prefix) + 64 or not value.startswith(expected_prefix):
        raise ValueError("group audit digest prefix is invalid")
    if any(
        character not in "0123456789abcdef"
        for character in value[len(expected_prefix) :]
    ):
        raise ValueError("group audit digest encoding is invalid")
