from __future__ import annotations

from typing import Literal

from pydantic import JsonValue

from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
)
from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.audit_records import (
    DirectAuditCommitProposalV1,
    DirectAuditLengthV1,
    DirectDependencyCountV1,
    GroupAuditCommitProposalV1,
    SanitizedAuditCommitProposalV1,
)
from market_support_crewai_agent.runtime.state.audit_types import (
    DirectAuditReasonCodeV1,
    ReasonCodeV1,
)
from market_support_crewai_agent.runtime.state.conversation_records import (
    AssistantTurnProposalV1,
    UserTurnProposalV1,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    NewReplyReservationV1,
    ReplyCommitProposalV1,
    ReservationResultV1,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_envelope

GROUP_CANDIDATE_REASON_CODES: tuple[ReasonCodeV1, ...] = (
    "knowledge_answer_composer",
    "smalltalk_composer",
    "composer_not_available",
    "composer_output_rejected",
    "expected_or_target_return",
    "qualified_investor_or_threshold",
    "unknown",
)
DIRECT_AUDIT_REASON_CODES: tuple[DirectAuditReasonCodeV1, ...] = (
    "compliant_product_request",
    "customer_service_request",
    "expected_or_target_return",
    "principal_or_risk_guarantee",
    "peer_or_competitor_comparison",
    "private_contact_request",
    "contract_or_restricted_document",
    "restricted_internal_document",
    "fee_waiver_request",
    "qualified_investor_or_threshold",
    "unrelated_request",
    "unknown",
    "ambiguous_request",
    "direct_human_handoff",
    "no_reply",
    "knowledge_answer_composer",
    "smalltalk_composer",
    "composer_not_available",
    "composer_output_rejected",
    "insufficient_evidence",
)
GROUP_ONLY_REASON_CODES: tuple[ReasonCodeV1, ...] = (
    "action_ready",
    "direct_send_command_matched",
)
_DIGEST = "dah1:" + "0" * 64


class _EmptyDocumentProvider:
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[GatewayDocumentContextV1, ...]:
        del request, evidence_query, cache_authority
        return ()


class _EmptyStaticProvider:
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
    ) -> tuple[GatewayStaticContextV1, ...]:
        del request, evidence_query
        return ()


def state_key(scene: Literal["group", "direct"]) -> ConversationStateKey:
    return ConversationStateKey(
        surface="wecom",
        adapter_namespace="test-adapter",
        tenant_ref="tenant:test",
        scene=scene,
        subject_ref=f"{scene}:subject",
        principal_ref="principal:test",
    )


def reply(action_payloads: tuple[JsonValue, ...] = ()) -> ReplyResponse:
    return ReplyResponse.model_validate(
        {
            "reply": {"kind": "answer", "text": "safe reply"},
            "actions": list(action_payloads),
        }
    )


def action_reply_variants() -> tuple[ReplyResponse, ...]:
    return (
        reply(),
        reply(
            (
                {
                    "type": "send_weekly_report",
                    "resolve_type": "weekly_report",
                    "resolve_ref": "weekly:one",
                    "period": "2026-W29",
                    "report_date": "2026-07-21",
                },
            )
        ),
        reply(
            (
                {
                    "type": "send_weekly_report",
                    "resolve_type": "weekly_report",
                    "resolve_ref": "weekly:one",
                    "period": "2026-W29",
                    "report_date": "2026-07-21",
                },
                {
                    "type": "send_monthly_report",
                    "resolve_type": "monthly_report",
                    "resolve_ref": "monthly:two",
                    "period": "2026-07",
                    "report_date": "2026-07-01",
                },
            )
        ),
    )


def direct_audit(reason_code: DirectAuditReasonCodeV1) -> DirectAuditCommitProposalV1:
    return DirectAuditCommitProposalV1(
        state_key_digest=_DIGEST,
        request_id_digest=_DIGEST,
        reason_code=reason_code,
        manifest_refs=("general.handoff@2026-07-15.1",),
        content_digests=(("reply", _DIGEST),),
        lengths=(DirectAuditLengthV1("reply_chars", 10),),
        dependency_counts=(DirectDependencyCountV1("evidence", 0),),
        static_registry_id="capability-registry.v2",
        static_registry_version="2026-07-15.1",
    )


def group_audit(reason_code: ReasonCodeV1) -> GroupAuditCommitProposalV1:
    return GroupAuditCommitProposalV1(
        reason_code=reason_code,
        manifest_refs=("general.smalltalk@2026-07-15.1",),
        program_attempts=(),
    )


def commit_reply(
    coordinator: ReplyStateTransactionCoordinatorV1,
    state: ConversationStateKey,
    request_id: str,
    request_hash: str,
    response: ReplyResponse,
    audit: SanitizedAuditCommitProposalV1,
    reservation: ReservationResultV1 | None = None,
) -> ReplyResponse:
    accepted_reservation = reservation or coordinator.reserve_reply(
        state, request_id, request_hash, True
    )
    assert isinstance(accepted_reservation, NewReplyReservationV1)
    return coordinator.commit_reply(
        ReplyCommitProposalV1(
            state_key=accepted_reservation.state_key,
            request_id=accepted_reservation.request_id,
            request_hash=accepted_reservation.request_hash,
            replay_eligible=accepted_reservation.replay_eligible,
            owner_token=accepted_reservation.owner_token,
            expected_revision_epoch=accepted_reservation.revision_epoch,
            expected_state_revision=accepted_reservation.state_revision,
            response=response,
            pol1="pol1:test",
            par1="par1:test",
            grh1="grh1:test",
            bsh1="bsh1:test",
            user_turn=UserTurnProposalV1(text="audit request"),
            assistant_turn=AssistantTurnProposalV1(
                text=response.reply.text,
                reply_kind=response.reply.kind,
                clarification_requested=False,
            ),
            clarification=None,
            audit=audit,
        )
    )


def direct_runtime(
    settings: Settings,
    coordinator: ReplyStateTransactionCoordinatorV1,
) -> CrewAIReplyRuntime:
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=_EmptyDocumentProvider(),
        static_provider=_EmptyStaticProvider(),
    )
    return CrewAIReplyRuntime(
        settings,
        coordinator=coordinator,
        internal_company_knowledge_gateway=gateway,
    )


def direct_envelope(
    message: str,
    thread_ref: str,
    principal_ref: str,
    presentation_name: str | None = None,
) -> VerifiedRequestEnvelopeV1:
    presentation: dict[str, str] = {"contract_version": "direct-presentation.v1"}
    if presentation_name is not None:
        presentation["principal_name"] = presentation_name
    return make_v2_envelope(
        message,
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:test",
            "direct_thread_ref": thread_ref,
            "principal_ref": principal_ref,
        },
        presentation=presentation,
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
