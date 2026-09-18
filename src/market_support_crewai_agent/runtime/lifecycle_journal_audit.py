from __future__ import annotations

from collections.abc import Callable
from typing import Final

from market_support_crewai_agent.runtime.observability.direct_audit import (
    DirectAuditPurposeV1,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
    TurnLlmInvocationRowV1,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    resolve_active_prompt_program_v2,
)
from market_support_crewai_agent.runtime.state.audit_program_records import (
    LlmInvocationAuditProposalV1,
)
from market_support_crewai_agent.runtime.state.audit_types import (
    AuditErrorCodeV1,
    AuditInvocationStatusV1,
    AuditTargetSlotV1,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)

_AUDIT_ERROR_CODES: Final[dict[str, AuditErrorCodeV1]] = {
    "planner_plan_validation_failed": "planner_plan_validation_failed",
    "knowledge_composer_reply_validation_failed": (
        "knowledge_composer_reply_validation_failed"
    ),
    "smalltalk_composer_reply_validation_failed": (
        "smalltalk_composer_reply_validation_failed"
    ),
    "alignment_verifier_verdict_validation_failed": (
        "alignment_verifier_verdict_validation_failed"
    ),
    "approved_selector_validation_failed": "approved_selector_validation_failed",
    "document_selector_validation_failed": "document_selector_validation_failed",
    "provider_output_missing": "provider_output_missing",
    "provider_output_type": "provider_output_type",
    "provider_output_encoding": "provider_output_encoding",
    "provider_output_too_large": "provider_output_too_large",
    "provider_output_contract": "provider_output_contract",
    "provider_timeout": "provider_timeout",
    "provider_auth_failed": "provider_auth_failed",
    "provider_rate_limited": "provider_rate_limited",
    "provider_http_error": "provider_http_error",
    "provider_transport_unavailable": "provider_transport_unavailable",
    "provider_internal_error": "provider_internal_error",
    "direct_provider_stdio_violation": "direct_provider_stdio_violation",
}

_AUDIT_TARGET_SLOTS: Final[dict[str, AuditTargetSlotV1]] = {
    "planner": "planner",
    "composer": "composer",
    "selector": "selector",
}


def journal_audit_attempts(
    journal: TurnLlmInvocationJournalV1,
    *,
    scene: str,
    direct_digest: Callable[[DirectAuditPurposeV1, str], str] | None = None,
) -> tuple[LlmInvocationAuditProposalV1, ...]:
    attempts: list[LlmInvocationAuditProposalV1] = []
    for row in journal.rows:
        if row.stage_kind == "llm_health_probe":
            continue
        if row.status == "reserved":
            raise CoordinatorError("llm_invocation_journal_row_unclosed")
        scene_key = (
            "scene_neutral.v1"
            if row.stage_kind
            in {"approved_knowledge_selector", "document_product_selector"}
            else ("wecom_group.v1" if scene == "group" else "wecom_direct.v1")
        )
        program, _execution_spec = resolve_active_prompt_program_v2(
            stage=row.stage_kind,
            scene_key=scene_key,
        )
        if program.program_id != row.program_id:
            raise CoordinatorError("llm_invocation_program_registry_mismatch")
        scene_contract_ref = (
            f"{program.scene_contract_id}@{program.scene_contract_version}"
            if program.scene_contract_id is not None
            and program.scene_contract_version is not None
            else None
        )
        if (
            row.program_version != program.program_version
            or row.scene_key != program.scene_key
            or row.scene_contract_ref != scene_contract_ref
        ):
            raise CoordinatorError("llm_invocation_journal_program_metadata_mismatch")
        status, error_code = _audit_status(row)
        metadata = _journal_transport_metadata(row)
        input_digest = None
        output_digest = None
        osh1 = None
        poh1 = None
        hph1 = None
        prh1 = None
        if direct_digest is not None:
            input_purpose, output_purpose = _direct_attempt_purposes(row)
            input_hash = row.prh1 or row.poh1 or row.osh1
            if input_hash is None:
                raise CoordinatorError("direct_audit_program_input_hash_required")
            input_digest = direct_digest(
                input_purpose,
                f"{row.invocation_ordinal}:{input_hash}",
            )
            if status == "success":
                if row.output_digest is None:
                    raise CoordinatorError("direct_audit_program_output_hash_required")
                output_digest = direct_digest(
                    output_purpose,
                    f"{row.invocation_ordinal}:{row.output_digest}",
                )
        else:
            input_digest = row.input_digest
            output_digest = row.output_digest
            osh1 = row.osh1
            poh1 = row.poh1
            hph1 = row.hph1
            prh1 = row.prh1
        attempts.append(
            LlmInvocationAuditProposalV1(
                ordinal=row.invocation_ordinal,
                logical_attempt=row.logical_attempt,
                transport_attempt=row.transport_attempt,
                stage=row.stage_kind,
                program_id=row.program_id,
                program_version=program.program_version,
                target_slot=_audit_target_slot(row.target_slot),
                scene_key=program.scene_key,
                scene_contract_ref=scene_contract_ref,
                model_family=metadata["model_family"],
                provider_id=metadata["provider_id"],
                transport_id=metadata["transport_id"],
                input_schema_version=metadata["input_schema_version"],
                output_schema_version=metadata["output_schema_version"],
                osh1=osh1,
                poh1=poh1,
                hph1=hph1,
                prh1=prh1,
                status=status,
                latency_ms=row.latency_ms,
                input_bytes=row.input_bytes,
                output_bytes=row.output_bytes,
                error_code=error_code,
                input_digest=input_digest,
                output_digest=output_digest,
            )
        )
    return tuple(attempts)


def _journal_transport_metadata(row: TurnLlmInvocationRowV1) -> dict[str, str]:
    model_family = row.model_family
    provider_id = row.provider_id
    transport_id = row.transport_id
    input_schema_version = row.input_schema_version
    output_schema_version = row.output_schema_version
    if (
        model_family is None
        or provider_id is None
        or transport_id is None
        or input_schema_version is None
        or output_schema_version is None
    ):
        raise CoordinatorError("llm_invocation_transport_metadata_required")
    return {
        "model_family": model_family,
        "provider_id": provider_id,
        "transport_id": transport_id,
        "input_schema_version": input_schema_version,
        "output_schema_version": output_schema_version,
    }


def _audit_target_slot(target_slot: str) -> AuditTargetSlotV1:
    try:
        return _AUDIT_TARGET_SLOTS[target_slot]
    except KeyError:
        raise CoordinatorError("llm_invocation_audit_target_slot_forbidden") from None


def _audit_error_code(error_code: str | None) -> AuditErrorCodeV1:
    if error_code is None:
        raise CoordinatorError("llm_invocation_error_code_required")
    try:
        return _AUDIT_ERROR_CODES[error_code]
    except KeyError:
        raise CoordinatorError("llm_invocation_error_code_unregistered") from None


def _audit_status(
    row: TurnLlmInvocationRowV1,
) -> tuple[AuditInvocationStatusV1, AuditErrorCodeV1 | None]:
    if row.status == "success":
        return "success", None
    if row.status == "output_contract_error":
        return "output_contract_error", _audit_error_code(row.error_code)
    if row.status == "transport_error" or row.status == "uncaptured_network_error":
        return "transport_error", _audit_error_code(row.error_code)
    if row.status == "reserved":
        raise CoordinatorError("llm_invocation_journal_row_unclosed")
    raise CoordinatorError("llm_invocation_status_unregistered")


def _direct_attempt_purposes(
    row: TurnLlmInvocationRowV1,
) -> tuple[DirectAuditPurposeV1, DirectAuditPurposeV1]:
    if row.stage_kind == "planner_intent":
        return "planner_input", "planner_output"
    if row.stage_kind == "knowledge_composer" or row.stage_kind == "smalltalk_composer":
        return "composer_input", "composer_output"
    if row.stage_kind == "alignment_verifier":
        return "verifier_input", "verifier_output"
    if row.stage_kind == "approved_knowledge_selector":
        return "approved_selector_input", "approved_selector_output"
    if row.stage_kind == "document_product_selector":
        return "document_selector_input", "document_selector_output"
    if row.stage_kind == "llm_health_probe":
        raise CoordinatorError("health_probe_direct_audit_forbidden")
    raise CoordinatorError("direct_audit_program_stage_unregistered")
