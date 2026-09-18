from __future__ import annotations

import json
from typing import Final

from market_support_crewai_agent.runtime import lifecycle_journal_audit
from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
    request_kernel_hash,
)
from market_support_crewai_agent.runtime.observability.direct_audit import (
    DirectAuditPurposeV1,
    decode_direct_audit_hmac_key,
    direct_audit_digest,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.state.audit_records import (
    DirectAuditCommitProposalV1,
    DirectAuditLengthV1,
    DirectDependencyCountV1,
    GroupAuditCommitProposalV1,
)
from market_support_crewai_agent.runtime.state.audit_types import (
    DirectAuditReasonCodeV1,
    ReasonCodeV1,
)
from market_support_crewai_agent.runtime.state.conversation_records import (
    AssistantTurnProposalV1,
    PendingClarificationProposalV1,
    UserTurnProposalV1,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    NewReplyReservationV1,
    ReplyCommitProposalV1,
    TurnAdmissionSnapshotV1,
)
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult

_DIRECT_AUDIT_REASON_CODES: Final[dict[str, DirectAuditReasonCodeV1]] = {
    "compliant_product_request": "compliant_product_request",
    "customer_service_request": "customer_service_request",
    "expected_or_target_return": "expected_or_target_return",
    "principal_or_risk_guarantee": "principal_or_risk_guarantee",
    "peer_or_competitor_comparison": "peer_or_competitor_comparison",
    "private_contact_request": "private_contact_request",
    "contract_or_restricted_document": "contract_or_restricted_document",
    "restricted_internal_document": "restricted_internal_document",
    "fee_waiver_request": "fee_waiver_request",
    "qualified_investor_or_threshold": "qualified_investor_or_threshold",
    "unrelated_request": "unrelated_request",
    "unknown": "unknown",
    "ambiguous_request": "ambiguous_request",
    "direct_human_handoff": "direct_human_handoff",
    "no_reply": "no_reply",
    "knowledge_answer_composer": "knowledge_answer_composer",
    "smalltalk_composer": "smalltalk_composer",
    "composer_not_available": "composer_not_available",
    "composer_output_rejected": "composer_output_rejected",
    "insufficient_evidence": "insufficient_evidence",
}


def reply_commit_proposal(
    candidate: V2AttemptResult,
    *,
    envelope: VerifiedRequestEnvelopeV1,
    reservation: NewReplyReservationV1,
    snapshot: TurnAdmissionSnapshotV1,
    policy: PolicyManifestV2,
    direct_audit_hmac_key: str | None,
    llm_journal: TurnLlmInvocationJournalV1,
) -> ReplyCommitProposalV1:
    request = envelope.request
    state_key = envelope.state_key
    clarification = None
    if candidate.response.reply.kind == "clarification":
        clarification = PendingClarificationProposalV1(
            kind="other",
            slots=tuple(
                slot for unit in candidate.plan.units for slot in unit.ambiguity_slots
            ),
            question=candidate.response.reply.text,
            topic=None,
        )
    return ReplyCommitProposalV1(
        state_key=state_key,
        request_id=request.request_id,
        request_hash=request_kernel_hash(envelope),
        replay_eligible=request.replay_eligible,
        owner_token=reservation.owner_token,
        expected_revision_epoch=snapshot.revision_epoch,
        expected_state_revision=snapshot.state_revision,
        response=candidate.response,
        pol1=policy.policy_id,
        par1=policy.state_admission_hash,
        grh1=policy.effective_grants_hash,
        bsh1=policy.business_scope_hash,
        user_turn=UserTurnProposalV1(text=request.message),
        assistant_turn=AssistantTurnProposalV1(
            text=candidate.response.reply.text,
            reply_kind=candidate.response.reply.kind,
            clarification_requested=clarification is not None,
        ),
        clarification=clarification,
        audit=audit_commit_proposal(
            candidate,
            request=request,
            state_key=state_key,
            direct_audit_hmac_key=direct_audit_hmac_key,
            llm_journal=llm_journal,
        ),
    )


def audit_commit_proposal(
    candidate: V2AttemptResult,
    *,
    request: KernelReplyRequestV1,
    state_key: ConversationStateKey,
    direct_audit_hmac_key: str | None,
    llm_journal: TurnLlmInvocationJournalV1,
) -> GroupAuditCommitProposalV1 | DirectAuditCommitProposalV1:
    reason_code = candidate.reason_code
    manifest_refs = tuple(
        f"{ref.manifest_id}@{ref.manifest_version}"
        for ref in candidate.plan.selected_manifest_refs
    )
    if state_key.scene == "group":
        return GroupAuditCommitProposalV1(
            reason_code=reason_code,
            manifest_refs=manifest_refs,
            program_attempts=lifecycle_journal_audit.journal_audit_attempts(
                llm_journal,
                scene="group",
            ),
        )
    if direct_audit_hmac_key is None:
        raise CoordinatorError("direct_audit_hmac_key_required")
    key = decode_direct_audit_hmac_key(direct_audit_hmac_key)

    def digest(purpose: DirectAuditPurposeV1, value: str) -> str:
        return direct_audit_digest(
            key=key,
            purpose=purpose,
            state_key=state_key,
            content=value,
        )

    program_attempts = lifecycle_journal_audit.journal_audit_attempts(
        llm_journal,
        scene="direct",
        direct_digest=digest,
    )
    reply_payload = json.dumps(
        candidate.response.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    plan_payload = json.dumps(
        candidate.plan.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    evidence_ids = ",".join(
        fact.evidence_id for fact in candidate.evidence.canonical_facts
    )
    return DirectAuditCommitProposalV1(
        state_key_digest=digest("request", "state"),
        request_id_digest=digest("request_id", request.request_id),
        reason_code=_direct_audit_reason_code(reason_code),
        manifest_refs=manifest_refs,
        content_digests=(
            ("message", digest("message", request.message)),
            ("reply", digest("reply", reply_payload)),
            ("plan", digest("plan", plan_payload)),
            ("evidence", digest("evidence", evidence_ids)),
        ),
        lengths=(
            DirectAuditLengthV1("message_chars", len(request.message)),
            DirectAuditLengthV1("message_bytes", len(request.message.encode("utf-8"))),
            DirectAuditLengthV1("reply_chars", len(candidate.response.reply.text)),
            DirectAuditLengthV1(
                "reply_bytes", len(candidate.response.reply.text.encode("utf-8"))
            ),
            DirectAuditLengthV1(
                "evidence_count", len(candidate.evidence.canonical_facts)
            ),
            DirectAuditLengthV1("manifest_count", len(manifest_refs)),
            DirectAuditLengthV1("invocation_count", len(program_attempts)),
        ),
        dependency_counts=(
            DirectDependencyCountV1("evidence", len(candidate.evidence.groundings)),
        ),
        static_registry_id="capability-registry.v2",
        static_registry_version="2026-07-18.1",
        program_attempts=program_attempts,
    )


def _direct_audit_reason_code(reason_code: ReasonCodeV1) -> DirectAuditReasonCodeV1:
    try:
        return _DIRECT_AUDIT_REASON_CODES[reason_code]
    except KeyError:
        raise CoordinatorError("direct_audit_reason_code_forbidden") from None
