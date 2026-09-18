from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import JsonValue

from market_support_crewai_agent.runtime import (
    lifecycle_admission,
    lifecycle_audit,
    lifecycle_contracts,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import (
    VerifiedRequestEnvelopeV1,
    request_kernel_hash,
    state_key_ref,
)
from market_support_crewai_agent.runtime.observability.runtime_trace import (
    RuntimeTrace,
    trace_event,
    trace_span,
    use_runtime_trace,
)
from market_support_crewai_agent.runtime.observability.runtime_trace_direct import (
    TraceAttributes,
)
from market_support_crewai_agent.runtime.policy.ontology import (
    DomainContextV1Builder,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
    TurnSelectorOutcomeCacheV1,
    use_turn_llm_invocation_journal,
    use_turn_selector_outcome_cache,
)
from market_support_crewai_agent.runtime.prompts.router import (
    model_family_from_settings,
    route_intent,
)
from market_support_crewai_agent.runtime.rendering.composer import CrewAIV2Composer
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    CompletedReplyReplayV1,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.v2_attempt import (
    reply_validation_error_summary,
)
from market_support_crewai_agent.runtime.validation.alignment_loop import (
    alignment_limits_from_settings,
    ensure_aligned_v2_response,
)
from market_support_crewai_agent.runtime.validation.alignment_runtime import (
    RuntimeAlignmentContextV1,
    RuntimeAlignmentRemediatorV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.validation.reply_validator import (
    ReplyContractError,
)
from market_support_crewai_agent.runtime.validation.request_input_guard import (
    validate_reply_request_input,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse

logger = logging.getLogger(__name__)


async def run_reply_turn(
    runtime: lifecycle_contracts.ReplyTurnRuntimeV1,
    envelope: VerifiedRequestEnvelopeV1,
) -> ReplyResponse:
    request = envelope.request
    state_key = envelope.state_key
    state_ref = state_key_ref(state_key)
    scope_authority = business_scope_authority_v1(request.business_scope)
    is_direct = state_key.scene == "direct"
    audit_context_id = request.context_id if state_key.scene == "group" else None
    trace = RuntimeTrace(
        context={
            "context_id": audit_context_id,
            "conversation_key": state_ref,
        },
        direct=is_direct,
    )
    reservation = runtime.coordinator.reserve_reply(
        state_key,
        request.request_id,
        request_kernel_hash(envelope),
        request.replay_eligible,
    )
    if isinstance(reservation, CompletedReplyReplayV1):
        return reservation.response

    committed = False
    llm_journal = TurnLlmInvocationJournalV1(
        max_rows=runtime.settings.maximum_llm_invocation_rows
    )
    selector_cache = TurnSelectorOutcomeCacheV1()
    if runtime.v2_composer is None:
        runtime.v2_composer = CrewAIV2Composer(runtime)
    with (
        use_runtime_trace(trace),
        use_turn_llm_invocation_journal(llm_journal),
        use_turn_selector_outcome_cache(selector_cache),
    ):
        try:
            with trace_span("request.validate"):
                validate_reply_request_input(request, runtime.settings)
                if not runtime.settings.llm_api_key:
                    raise AgentRuntimeError("YANFU_LLM_API_KEY is not configured")

            admission = lifecycle_admission.admit_turn(
                runtime.coordinator,
                request=request,
                state_key=state_key,
                reservation=reservation,
                scope_authority=scope_authority,
                group_recall_mode=runtime.settings.group_recall_mode,
            )
            policy_v2 = admission.policy
            history = admission.history
            action_history = admission.action_history

            with trace_span("domain.build_context"):
                domain_context = DomainContextV1Builder().build(
                    request,
                    conversation_metadata={
                        "context_id": request.context_id,
                        "conversation_key": state_ref,
                    },
                    scope_authority=scope_authority,
                )

            model_family = model_family_from_settings(runtime.settings)
            policy_trace_attrs: TraceAttributes = {
                "eligible_manifest_ids": [
                    ref.manifest_id for ref in policy_v2.eligible_capabilities
                ],
                "allowed_actions": list(policy_v2.allowed_outbound_actions),
            }
            if state_key.scene == "group":
                policy_trace_attrs["business_scope_hash"] = (
                    scope_authority.business_scope_hash
                )
                policy_trace_attrs["business_scope_ref"] = (
                    scope_authority.business_scope_ref
                )
            trace_event("state.policy_compiled", **policy_trace_attrs)

            with trace_span("intent.route"):
                intent_gate = route_intent(
                    request,
                    policy_v2,
                    history=history,
                )
            llm_executions: list[dict[str, JsonValue]] = []
            prompt_programs: list[PromptProgram] = []
            alignment_verdicts: list[ReplyAlignmentVerdict] = []
            alignment_remediations: list[dict[str, JsonValue]] = []
            with trace_span("candidate.build"):
                candidate_builder = runtime.candidate_response_builder_v1
                candidate = await candidate_builder(
                    request=request,
                    domain_context=domain_context,
                    policy=policy_v2,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    scope_authority=scope_authority,
                    state_key_ref=state_ref,
                    llm_journal=llm_journal,
                )
            if (
                candidate.reply_validation.valid
                and runtime.settings.reply_alignment_verifier_enabled
            ):
                with trace_span("alignment.ensure"):
                    candidate = await ensure_aligned_v2_response(
                        candidate,
                        RuntimeAlignmentRemediatorV1(
                            runtime,
                            RuntimeAlignmentContextV1(
                                request=request,
                                policy=policy_v2,
                                model_family=model_family,
                                intent_gate=intent_gate,
                                history=history,
                                action_history=action_history,
                                prompt_programs=prompt_programs,
                                llm_executions=llm_executions,
                                alignment_verdicts=alignment_verdicts,
                                alignment_remediations=alignment_remediations,
                                scope_authority=scope_authority,
                                state_key_ref=state_ref,
                                now=datetime.now(UTC),
                            ),
                        ),
                        alignment_limits_from_settings(runtime.settings),
                    )
            trace_event(
                "state.candidate_ready",
                reply_kind=candidate.response.reply.kind,
                action_count=len(candidate.response.actions),
                reply_valid=candidate.reply_validation.valid,
                alignment_verdict_count=len(alignment_verdicts),
            )

            if not candidate.reply_validation.valid:
                raise ReplyContractError(
                    f"rendered reply failed validation: {reply_validation_error_summary(candidate.reply_validation)}"
                )

            with trace_span("state.commit_turn"):
                response = runtime.coordinator.commit_reply(
                    lifecycle_audit.reply_commit_proposal(
                        candidate,
                        envelope=envelope,
                        reservation=reservation,
                        snapshot=admission.snapshot,
                        policy=policy_v2,
                        direct_audit_hmac_key=runtime.settings.direct_audit_hmac_key,
                        llm_journal=llm_journal,
                    )
                )
            committed = True
            return response
        finally:
            if not committed:
                try:
                    runtime.coordinator.abort_reply(
                        state_key, request.request_id, reservation.owner_token
                    )
                except CoordinatorError:
                    trace_event("state.abort_reply_conflict")
            trace.log_trace(
                logger,
                context_id=audit_context_id,
                conversation_key=state_ref,
            )
    raise AssertionError("unreachable reply runtime path")
