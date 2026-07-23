from __future__ import annotations

import hashlib
from dataclasses import replace

from pydantic import ValidationError

from market_support_crewai_agent.runtime.context.pending import (
    PendingOutboundConfirmation,
    pending_outbound_confirmation,
)
from market_support_crewai_agent.runtime.domain.business_facts import (
    derive_business_facts,
)
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
)
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.evidence.models import EvidenceFact
from market_support_crewai_agent.runtime.llm.prompting.context import (
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
    DirectOutboundDraft,
)
from market_support_crewai_agent.runtime.llm.prompting.router import (
    select_prompt_program,
)
from market_support_crewai_agent.runtime.orchestration.composer import (
    run_composer_kickoff_with_retry,
)
from market_support_crewai_agent.runtime.orchestration.decision import (
    ResponseDirective,
)
from market_support_crewai_agent.runtime.orchestration.direct_candidate_support import (
    build_direct_knowledge_plan,
    coerce_direct_output,
    direct_answerability,
    materialize_allowed_direct_output,
    plan_for_direct_materialization,
)
from market_support_crewai_agent.runtime.orchestration.response_ids import (
    ensure_response_ids,
)
from market_support_crewai_agent.runtime.orchestration.reply_history import (
    pending_direct_outbound_draft,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.runtime_trace import trace_event
from market_support_crewai_agent.runtime.turn import AgentRuntimeError, AttemptResult
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.validation.direct_pending_confirmation import (
    pending_confirmation_resolution_issue,
)
from market_support_crewai_agent.runtime.validation.reply_validator import (
    validate_reply,
)
from market_support_crewai_agent.schemas import PrimaryReply, ReplyRequest


async def direct_outbound_ready(runtime) -> bool:
    try:
        capabilities = (
            await runtime.preflight_service.adapter_client.capabilities_async()
        )
    except AdapterClientError:
        return False
    outbound = capabilities.outbound_messaging
    return (
        outbound is not None
        and outbound.ready
        and "outbound_message_target" in capabilities.resolve_types
        and {"channel", "group"}.issubset(capabilities.outbound_target_kinds)
    )


async def build_direct_message_candidate(
    runtime,
    *,
    request: ReplyRequest,
    domain_context: DomainContext,
    policy: PolicyManifest,
    model_family,
    intent_gate,
    history,
    action_history: list[ActionLedgerRecord],
    prompt_programs,
    llm_executions,
    **_unused,
) -> AttemptResult:
    knowledge_plan = build_direct_knowledge_plan(request, policy)
    pending_outbound_draft = pending_direct_outbound_draft(history)
    pending_confirmation = pending_outbound_confirmation(history)
    if pending_outbound_draft is not None:
        trace_event(
            "state.direct_outbound_draft_loaded",
            target_known=pending_outbound_draft.target is not None,
            content_known=pending_outbound_draft.content is not None,
        )

    async def compose(
        evidence: list[EvidenceFact],
        *,
        pending_resolution_issue: str | None = None,
    ) -> DirectComposerOutput:
        composer_context = runtime._project_context(
            stage="direct_composer",
            request=request,
            domain_context=domain_context,
            policy=policy,
            intent_gate=None,
            execution_plan=knowledge_plan,
            plan_validation=None,
            preflight=AdapterPreflightSnapshot.empty(),
            evidence_facts=evidence,
            history=history,
            action_history=action_history,
        )
        program = select_prompt_program(
            PromptAssemblyContext(
                stage="direct_composer",
                model_family=model_family,
                request=request,
                policy=policy,
                model_visible_context=composer_context,
                domain_context=domain_context,
                intent_gate=None,
                execution_plan=None,
                plan_validation=None,
                evidence_facts=evidence,
                history=history,
                action_history=action_history,
            )
        )
        if pending_resolution_issue is not None:
            program = _pending_resolution_retry_program(
                program,
                pending_resolution_issue,
            )
        prompt_programs.append(program)
        agent = runtime._build_agent("direct_composer")
        try:
            result, executions = await run_composer_kickoff_with_retry(
                agent,
                program,
                timeout_seconds=runtime.settings.llm_timeout_seconds,
                retry_attempts=runtime.settings.planner_transient_retry_attempts,
                base_delay_seconds=runtime.settings.planner_transient_retry_base_seconds,
            )
        except TimeoutError as exc:
            raise AgentRuntimeError("direct composer timed out") from exc
        except ValidationError as exc:
            output = _pending_clarification_output(pending_outbound_draft)
            if output is None:
                raise AgentRuntimeError(
                    "direct composer returned an invalid contract"
                ) from exc
            trace_event(
                "state.direct_outbound_clarification_fallback",
                source="composer_validation_fallback",
            )
        else:
            llm_executions.extend(executions)
            output = coerce_direct_output(result)
            if output is None:
                output = _pending_clarification_output(pending_outbound_draft)
                if output is None:
                    raise AgentRuntimeError(
                        "direct composer returned an invalid contract"
                    )
                trace_event(
                    "state.direct_outbound_clarification_fallback",
                    source="invalid_contract_fallback",
                )
        return output

    async def compose_with_pending_guard(
        evidence: list[EvidenceFact],
    ) -> DirectComposerOutput:
        output = await compose(evidence)
        issue = pending_confirmation_resolution_issue(output, pending_confirmation)
        if issue is None:
            return output
        trace_event(
            "state.direct_pending_resolution_retry",
            issue=issue,
            resolution=output.pending_confirmation_resolution,
            response_mode=output.response_mode,
        )
        output = await compose(evidence, pending_resolution_issue=issue)
        retry_issue = pending_confirmation_resolution_issue(
            output,
            pending_confirmation,
        )
        if retry_issue is None:
            return output
        trace_event(
            "state.direct_pending_resolution_fallback",
            issue=retry_issue,
            resolution=output.pending_confirmation_resolution,
            response_mode=output.response_mode,
        )
        assert pending_confirmation is not None
        return _pending_confirmation_clarification_output(pending_confirmation)

    evidence_facts: list[EvidenceFact] = []
    output = await compose_with_pending_guard(evidence_facts)
    if (
        output.response_mode in {"request_company_info", "answer_company_info"}
        and "document_context" in policy.allowed_capabilities
    ):
        trace_event("state.direct_knowledge_requested")
        evidence_facts = await _collect_company_evidence(
            runtime,
            request,
            knowledge_plan,
            policy,
        )
        trace_event("state.direct_knowledge_loaded", fact_count=len(evidence_facts))
        if evidence_facts:
            output = await compose_with_pending_guard(evidence_facts)
    materialization = await materialize_allowed_direct_output(
        output,
        policy=policy,
        evidence_facts=evidence_facts,
        adapter_client=runtime.preflight_service.adapter_client,
        action_history=action_history,
        pending_outbound_draft=pending_outbound_draft,
        pending_confirmation=pending_confirmation,
    )
    response = ensure_response_ids(materialization.response)
    plan = plan_for_direct_materialization(request, materialization, response)
    directive = ResponseDirective(
        mode=materialization.mode,
        reply_kind=response.reply.kind,
        text=response.reply.text,
        action_intents=plan.action_intents,
        reason_code="direct_message_composer",
    )
    business_facts = derive_business_facts(evidence_facts, request)
    reply_validation = validate_reply(
        response,
        directive,
        plan,
        business_facts,
        evidence_facts,
        policy,
        domain_context=domain_context,
    )
    return AttemptResult(
        plan=plan,
        plan_validation=PlanValidationResult(valid=True),
        preflight=AdapterPreflightSnapshot.empty(),
        evidence_facts=evidence_facts,
        business_facts=business_facts,
        domain_context=domain_context,
        answerability=direct_answerability(materialization, evidence_facts),
        directive=directive,
        response=response,
        reply_validation=reply_validation,
        guardrail_decisions=[
            GuardrailDecision(
                outcome="allow",
                phase="output",
                reason_code="direct_message_composer",
                capability_id=materialization.capability,
            )
        ],
        pending_outbound_draft=materialization.pending_outbound_draft,
    )


def _pending_clarification_output(
    pending: DirectOutboundDraft | None,
) -> DirectComposerOutput | None:
    if pending is None:
        return None
    return DirectComposerOutput(
        response_mode="clarify",
        reply=PrimaryReply(
            kind="clarification",
            text="我没有判断清楚您是要继续填写发送内容，还是切换话题，请确认一下。",
            mentions=[],
        ),
    )


def _pending_confirmation_clarification_output(
    pending: PendingOutboundConfirmation,
) -> DirectComposerOutput:
    draft = DirectOutboundDraft.model_validate(
        {
            "target": {
                "kind": pending.action.target.kind,
                "name": pending.action.target.name,
            },
            "content": pending.action.content.model_dump(mode="json", exclude_none=True),
        }
    )
    return DirectComposerOutput(
        response_mode="clarify",
        pending_confirmation_resolution="ambiguous",
        reply=PrimaryReply(
            kind="clarification",
            text="我没有判断清楚您是要确认、修改还是取消这条发送，请再说明一下。",
            mentions=[],
        ),
        target=draft.target,
        content=draft.content,
    )


def _pending_resolution_retry_program(
    program: PromptProgram,
    issue: str,
) -> PromptProgram:
    feedback = (
        '\n\n<prompt_layer id="ephemeral">\n'
        f"Pending confirmation validation error: {issue}\n"
        "Rewrite the full DirectComposerOutput JSON. Resolve the current message against "
        "the active pending_confirmation, make pending_confirmation_resolution and "
        "response_mode describe the same decision, and preserve exact target/content.\n"
        "</prompt_layer>"
    )
    prompt_text = program.prompt_text + feedback
    layers = program.layers
    if "ephemeral" not in layers:
        layers = (*layers, "ephemeral")
    return replace(
        program,
        prompt_text=prompt_text,
        prompt_hash="sha256:"
        + hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        layers=layers,
    )


async def _collect_company_evidence(
    runtime,
    request: ReplyRequest,
    plan: ExecutionPlan,
    policy: PolicyManifest,
) -> list[EvidenceFact]:
    if "document_context" not in policy.allowed_capabilities:
        return []
    return await runtime.evidence_executor.document_evidence_service.collect(
        request,
        plan,
        policy,
    )
