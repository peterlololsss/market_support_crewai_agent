from __future__ import annotations

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
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.turn import AgentRuntimeError, AttemptResult
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.validation.reply_validator import (
    validate_reply,
)
from market_support_crewai_agent.schemas import ReplyRequest


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
    evidence_facts = await _collect_company_evidence(
        runtime,
        request,
        knowledge_plan,
        policy,
    )
    composer_context = runtime._project_context(
        stage="direct_composer",
        request=request,
        domain_context=domain_context,
        policy=policy,
        intent_gate=intent_gate,
        execution_plan=knowledge_plan,
        plan_validation=PlanValidationResult(valid=True),
        preflight=AdapterPreflightSnapshot.empty(),
        evidence_facts=evidence_facts,
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
            intent_gate=intent_gate,
            execution_plan=knowledge_plan,
            plan_validation=PlanValidationResult(valid=True),
            evidence_facts=evidence_facts,
            history=history,
            action_history=action_history,
        )
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
    llm_executions.extend(executions)
    output = coerce_direct_output(result)
    if output is None:
        raise AgentRuntimeError("direct composer returned an invalid contract")

    materialization = await materialize_allowed_direct_output(
        output,
        policy=policy,
        evidence_facts=evidence_facts,
        adapter_client=runtime.preflight_service.adapter_client,
        action_history=action_history,
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
