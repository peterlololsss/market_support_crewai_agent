from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, replace

from market_support_crewai_agent.runtime.state.action_ledger import (
    ActionLedger,
    ActionLedgerRecord,
    get_action_ledger,
)
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.knowledge.approved_knowledge import (
    ApprovedKnowledgeEvidenceService,
)
from market_support_crewai_agent.runtime.state.audit import (
    AuditStore,
    build_audit_trace,
    get_audit_store,
)
from market_support_crewai_agent.runtime.state.runtime_trace import (
    RuntimeTrace,
    trace_event,
    trace_span,
    use_runtime_trace,
)
from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.canonicalization import (
    CanonicalContext,
    canonicalize_request,
)
from market_support_crewai_agent.runtime.domain.ontology import (
    DomainContext,
    DomainContextBuilder,
)
from market_support_crewai_agent.runtime.domain.capabilities import resolve_type_for_action
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.evidence.document_mcp import DocumentMcpEvidenceService
from market_support_crewai_agent.runtime.evidence.report_scope import ReportScopeEvidenceService
from market_support_crewai_agent.runtime.orchestration.decision import (
    DecisionEngine,
    ResponseDirective,
)
from market_support_crewai_agent.runtime.orchestration.answerability_directives import (
    default_answerability_for_plan,
    directive_from_answerability,
)
from market_support_crewai_agent.runtime.orchestration.crewai_io import (
    coerce_agent_response,
    coerce_alignment_verdict,
    coerce_composer_output,
    coerce_planner_plan,
    plan_spec_error_summary,
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.orchestration.alignment_loop import (
    ensure_aligned_response,
)
from market_support_crewai_agent.runtime.orchestration.reply_history import (
    compact_assistant_result,
)
from market_support_crewai_agent.runtime.context.projection import ContextProjectionManager
from market_support_crewai_agent.runtime.context.pressure import ProjectionLimitError
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.validation.reply_validator import (
    ReplyContractError,
    ValidationResult,
    remove_pre_execution_send_claims,
    validate_reply,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerifier,
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.validation.request_input_guard import (
    validate_reply_request_input,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
    abstention_response_text,
)
from market_support_crewai_agent.runtime.validation.output_guard import (
    output_guard,
)
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
    AnswerabilityGate,
)
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
    plan_spec_for_execution_plan,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import (
    PolicyManifest,
    compile_policy,
    ledger_summary_from_action_history,
)
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.context import IntentGateResult, PromptAssemblyContext
from market_support_crewai_agent.runtime.llm.prompting.router import (
    model_family_from_settings,
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.runtime.orchestration.crewai_agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.orchestration.response_renderer import render_directive
from market_support_crewai_agent.runtime.orchestration.response_ids import ensure_response_ids
from market_support_crewai_agent.schemas import PrimaryReply, ReplyRequest, ReplyResponse
from market_support_crewai_agent.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class AgentRuntimeError(RuntimeError):
    """Raised when the CrewAI runtime cannot produce a valid reply."""


_DEFAULT_SETTINGS = get_settings()
_DEFAULT_CONVERSATION_STORE = ConversationStore.from_settings(_DEFAULT_SETTINGS)
_DEFAULT_ACTION_LEDGER = get_action_ledger()
_DEFAULT_PREFLIGHT_SERVICE = AdapterPreflightService()
_DEFAULT_AUDIT_STORE = get_audit_store()
_DEFAULT_DOCUMENT_EVIDENCE_SERVICE = DocumentMcpEvidenceService(_DEFAULT_SETTINGS)
_DEFAULT_APPROVED_KNOWLEDGE_EVIDENCE_SERVICE = ApprovedKnowledgeEvidenceService(
    settings=_DEFAULT_SETTINGS
)
_DEFAULT_REPORT_SCOPE_EVIDENCE_SERVICE = ReportScopeEvidenceService(_DEFAULT_SETTINGS)


async def build_reply(
        request: ReplyRequest,
        settings: Settings | None = None,
        conversation_store: ConversationStore | None = None,
        action_ledger: ActionLedger | None = None,
        preflight_service: AdapterPreflightService | None = None,
        evidence_executor: EvidenceExecutor | None = None,
        audit_store: AuditStore | None = None,
        alignment_verifier: ReplyAlignmentVerifier | None = None,
) -> ReplyResponse:
    resolved_settings = settings or _DEFAULT_SETTINGS
    if preflight_service is not None:
        resolved_preflight_service = preflight_service
    elif settings is None:
        resolved_preflight_service = _DEFAULT_PREFLIGHT_SERVICE
    else:
        resolved_preflight_service = AdapterPreflightService(
            settings=resolved_settings,
        )

    runtime = CrewAIReplyRuntime(
        resolved_settings,
        conversation_store
        or (
            _DEFAULT_CONVERSATION_STORE
            if settings is None
            else ConversationStore.from_settings(resolved_settings)
        ),
        action_ledger or _DEFAULT_ACTION_LEDGER,
        resolved_preflight_service,
        evidence_executor
        or EvidenceExecutor(
            resolved_preflight_service,
            _DEFAULT_DOCUMENT_EVIDENCE_SERVICE
            if settings is None
            else DocumentMcpEvidenceService(resolved_settings),
            _DEFAULT_APPROVED_KNOWLEDGE_EVIDENCE_SERVICE
            if settings is None
            else ApprovedKnowledgeEvidenceService(settings=resolved_settings),
            _DEFAULT_REPORT_SCOPE_EVIDENCE_SERVICE
            if settings is None
            else ReportScopeEvidenceService(settings=resolved_settings),
        ),
        audit_store or _DEFAULT_AUDIT_STORE,
        alignment_verifier,
    )
    return await runtime.reply(request)


@dataclass(frozen=True)
class RuntimeAttemptResult:
    plan: ExecutionPlan
    plan_validation: PlanValidationResult
    preflight: AdapterPreflightSnapshot
    evidence_facts: list
    business_facts: BusinessFacts
    domain_context: DomainContext
    answerability: AnswerabilityAssessment
    directive: ResponseDirective
    response: ReplyResponse
    reply_validation: ValidationResult
    guardrail_decisions: list[GuardrailDecision]
    composer_output: ComposerReplyOutput | None = None


class CrewAIReplyRuntime:
    """CrewAI runtime boundary used by the FastAPI transport layer."""

    def __init__(
            self,
            settings: Settings,
            conversation_store: ConversationStore | None = None,
            action_ledger: ActionLedger | None = None,
            preflight_service: AdapterPreflightService | None = None,
            evidence_executor: EvidenceExecutor | None = None,
            audit_store: AuditStore | None = None,
            alignment_verifier: ReplyAlignmentVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.conversation_store = conversation_store or ConversationStore.from_settings(
            settings
        )
        self.action_ledger = action_ledger or get_action_ledger()
        self.preflight_service = preflight_service or AdapterPreflightService(
            settings=settings,
        )
        self.evidence_executor = evidence_executor or EvidenceExecutor(
            self.preflight_service,
            DocumentMcpEvidenceService(settings),
            ApprovedKnowledgeEvidenceService(settings=settings),
            ReportScopeEvidenceService(settings=settings),
        )
        self.audit_store = audit_store or get_audit_store()
        self.alignment_verifier = alignment_verifier
        self.agent_factory = CrewAIAgentFactory(settings)
        self.context_projection_manager = ContextProjectionManager.from_settings(settings)

    async def reply(self, request: ReplyRequest) -> ReplyResponse:
        trace = RuntimeTrace(
            context={
                "context_id": request.context_id,
                "conversation_key": request.conversation_key,
            }
        )
        with use_runtime_trace(trace):
            try:
                with trace_span("request.validate"):
                    validate_reply_request_input(request, self.settings)
                    if not self.settings.llm_api_key:
                        raise AgentRuntimeError("YANFU_LLM_API_KEY is not configured")

                with trace_span("state.load_history"):
                    history = self.conversation_store.get_recent(request.conversation_key)
                    action_history = self.action_ledger.recent_executed_for_conversation(
                        request.conversation_key,
                        limit=20,
                    )
                trace_event(
                    "state.history_loaded",
                    history_count=len(history),
                    action_history_count=len(action_history),
                )

                with trace_span("domain.build_context"):
                    domain_context = DomainContextBuilder().build(
                        request,
                        conversation_metadata={
                            "context_id": request.context_id,
                            "conversation_key": request.conversation_key,
                        },
                    )

                with trace_span("domain.canonicalize"):
                    canonical_context = canonicalize_request(
                        request,
                        domain_context=domain_context,
                    )

                model_family = model_family_from_settings(self.settings)
                with trace_span("policy.compile"):
                    policy = compile_policy(
                        request,
                        ledger_summary=ledger_summary_from_action_history(action_history),
                        doc_mcp_enabled=bool(
                            self.settings.doc_mcp_enabled and self.settings.doc_mcp_base_url
                        ),
                        doc_mcp_allowed_channel_types=self.settings.doc_mcp_allowed_channel_types,
                    )
                trace_event(
                    "state.policy_compiled",
                    allowed_capabilities=policy.allowed_capabilities,
                    allowed_actions=policy.allowed_side_effect_actions,
                )

                with trace_span("intent.route"):
                    intent_gate = route_intent(
                        request,
                        canonical_context,
                        policy,
                        history=history,
                    )
                llm_executions: list[dict] = []
                prompt_programs: list[PromptProgram] = []
                alignment_verdicts: list[ReplyAlignmentVerdict] = []
                alignment_remediations: list[dict] = []

                with trace_span("candidate.build"):
                    candidate = await self._build_candidate_response(
                        request=request,
                        canonical_context=canonical_context,
                        domain_context=domain_context,
                        policy=policy,
                        model_family=model_family,
                        intent_gate=intent_gate,
                        history=history,
                        action_history=action_history,
                        prompt_programs=prompt_programs,
                        llm_executions=llm_executions,
                    )
                if (
                    candidate.reply_validation.valid
                    and self.settings.reply_alignment_verifier_enabled
                ):
                    with trace_span("alignment.ensure"):
                        candidate = await self._ensure_aligned_response(
                            request=request,
                            canonical_context=canonical_context,
                            domain_context=domain_context,
                            policy=policy,
                            model_family=model_family,
                            intent_gate=intent_gate,
                            history=history,
                            action_history=action_history,
                            prompt_programs=prompt_programs,
                            llm_executions=llm_executions,
                            alignment_verdicts=alignment_verdicts,
                            alignment_remediations=alignment_remediations,
                            candidate=candidate,
                        )
                trace_event(
                    "state.candidate_ready",
                    reply_kind=candidate.response.reply.kind,
                    action_count=len(candidate.response.actions),
                    reply_valid=candidate.reply_validation.valid,
                    alignment_verdict_count=len(alignment_verdicts),
                )

                with trace_span("audit.record"):
                    self._record_audit_trace(
                        request=request,
                        policy=policy,
                        plan=candidate.plan,
                        directive=candidate.directive,
                        plan_validation=candidate.plan_validation,
                        action_history=action_history,
                        canonical_context=canonical_context,
                        domain_context=candidate.domain_context,
                        preflight=candidate.preflight,
                        evidence_facts=candidate.evidence_facts,
                        business_facts=candidate.business_facts,
                        answerability_assessment=candidate.answerability,
                        response=candidate.response,
                        reply_validation=candidate.reply_validation,
                        guardrail_decisions=candidate.guardrail_decisions,
                        intent_gate=intent_gate,
                        prompt_programs=prompt_programs,
                        llm_executions=llm_executions,
                        alignment_verdicts=alignment_verdicts,
                        alignment_remediations=alignment_remediations,
                        runtime_trace=trace.to_dict(),
                    )
                if not candidate.reply_validation.valid:
                    raise ReplyContractError(
                        "rendered reply failed validation: {}".format(
                            _reply_validation_error_summary(candidate.reply_validation)
                        )
                    )

                with trace_span("state.save_turn"):
                    self.conversation_store.save_turn(
                        request.conversation_key,
                        request.message,
                        compact_assistant_result(candidate.response, candidate.plan),
                    )
                return candidate.response
            finally:
                trace.log_trace(
                    logger,
                    context_id=request.context_id,
                    conversation_key=request.conversation_key,
                )

    async def _build_candidate_response(
            self,
            *,
            request: ReplyRequest,
            canonical_context: CanonicalContext,
            domain_context: DomainContext,
            policy: PolicyManifest,
            model_family,
            intent_gate: IntentGateResult,
            history: list,
            action_history: list[ActionLedgerRecord],
            prompt_programs: list[PromptProgram],
            llm_executions: list[dict],
            alignment_verdict: ReplyAlignmentVerdict | None = None,
            alignment_attempt: int = 0,
    ) -> RuntimeAttemptResult:
        with trace_span("planner.project_context"):
            planner_context = self._project_context(
                stage="planner_intent",
                request=request,
                canonical_context=canonical_context,
                domain_context=domain_context,
                policy=policy,
                intent_gate=intent_gate,
                history=history,
                action_history=action_history,
                alignment_verdict=alignment_verdict,
                alignment_attempt=alignment_attempt,
            )
        with trace_span("planner.assemble_prompt"):
            planner_program = select_prompt_program(
                PromptAssemblyContext(
                    stage="planner_intent",
                    model_family=model_family_from_settings(
                        self.settings,
                        stage="planner_intent",
                    ),
                    request=request,
                    canonical_context=canonical_context,
                    model_visible_context=planner_context,
                    domain_context=domain_context,
                    policy=policy,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    alignment_verdict=alignment_verdict,
                    alignment_attempt=alignment_attempt,
                )
        )
        prompt_programs.append(planner_program)
        try:
            with trace_span("planner.build_agent"):
                planner_agent = self._build_planner_agent()
            frame_result, planner_execution = await run_crewai_kickoff(
                planner_agent,
                planner_program,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            llm_executions.append(planner_execution)
        except asyncio.TimeoutError as exc:
            raise AgentRuntimeError("CrewAI planner timed out") from exc
        except Exception as exc:
            raise AgentRuntimeError("CrewAI planner failed") from exc

        with trace_span("planner.coerce_compile"):
            plan, error_summary = _coerce_planner_plan_with_error(
                frame_result,
                request,
                canonical_context,
                policy,
                domain_context=domain_context,
                history=history,
            )
        if plan is None:
            trace_event("planner.invalid_plan_spec", error=error_summary)
            retry_program = _planner_retry_program(planner_program, error_summary)
            prompt_programs.append(retry_program)
            try:
                frame_result, planner_execution = await run_crewai_kickoff(
                    planner_agent,
                    retry_program,
                    timeout_seconds=self.settings.llm_timeout_seconds,
                )
                llm_executions.append(planner_execution)
            except asyncio.TimeoutError as exc:
                raise AgentRuntimeError("CrewAI planner retry timed out") from exc
            except Exception as exc:
                raise AgentRuntimeError("CrewAI planner retry failed") from exc
            with trace_span("planner.retry_coerce_compile"):
                plan, error_summary = _coerce_planner_plan_with_error(
                    frame_result,
                    request,
                    canonical_context,
                    policy,
                    domain_context=domain_context,
                    history=history,
                )
        if plan is None:
            trace_event("planner.invalid_plan_spec", error=error_summary, retry=True)
            raise AgentRuntimeError(
                "CrewAI planner returned an invalid PlanSpec contract: "
                f"{error_summary}"
            )
        trace_event(
            "state.plan_compiled",
            response_mode=plan.response_mode,
            capabilities=plan.capabilities,
            adapter_resolve_count=len(plan.adapter_resolves),
        )
        return await self._build_candidate_from_plan(
            request=request,
            canonical_context=canonical_context,
            domain_context=domain_context,
            policy=policy,
            model_family=model_family,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            plan=plan,
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
        )

    async def _build_candidate_from_plan(
            self,
            *,
            request: ReplyRequest,
            canonical_context: CanonicalContext,
            domain_context: DomainContext,
            policy: PolicyManifest,
            model_family,
            intent_gate: IntentGateResult,
            history: list,
            action_history: list[ActionLedgerRecord],
            prompt_programs: list[PromptProgram],
            llm_executions: list[dict],
            plan: ExecutionPlan,
            alignment_verdict: ReplyAlignmentVerdict | None = None,
            alignment_attempt: int = 0,
    ) -> RuntimeAttemptResult:
        if plan.plan_spec is None:
            plan = plan.model_copy(
                update={
                    "plan_spec": plan_spec_for_execution_plan(
                        plan,
                        domain_context=domain_context,
                    )
                }
            )
        with trace_span("plan.validate"):
            plan_validation = validate_execution_plan(plan, policy)
        if not plan_validation.valid:
            raise AgentRuntimeError(
                "compiled execution plan failed validation: {}".format(
                    _validation_error_summary(plan_validation)
                )
            )

        with trace_span("evidence.execute"):
            evidence_result = await self.evidence_executor.execute(
                request,
                canonical_context,
                plan,
                policy,
                action_history=action_history,
            )
        trace_event(
            "state.evidence_collected",
            preflight_items=len(evidence_result.preflight.items),
            evidence_fact_count=len(evidence_result.evidence_facts),
            guardrail_decision_count=len(
                getattr(evidence_result, "guardrail_decisions", [])
            ),
        )
        return await self._build_candidate_from_evidence(
            request=request,
            canonical_context=canonical_context,
            domain_context=getattr(evidence_result, "domain_context", domain_context),
            policy=policy,
            model_family=model_family,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            plan=plan,
            plan_validation=plan_validation,
            preflight=evidence_result.preflight,
            evidence_facts=evidence_result.evidence_facts,
            business_facts=evidence_result.business_facts,
            guardrail_decisions=[
                *plan.guardrail_decisions,
                *getattr(evidence_result, "guardrail_decisions", []),
            ],
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
        )

    async def _build_candidate_from_evidence(
            self,
            *,
            request: ReplyRequest,
            canonical_context: CanonicalContext,
            domain_context: DomainContext,
            policy: PolicyManifest,
            model_family,
            intent_gate: IntentGateResult,
            history: list,
            action_history: list[ActionLedgerRecord],
            prompt_programs: list[PromptProgram],
            llm_executions: list[dict],
            plan: ExecutionPlan,
            plan_validation: PlanValidationResult,
            preflight: AdapterPreflightSnapshot,
            evidence_facts: list,
            business_facts: BusinessFacts,
            guardrail_decisions: list[GuardrailDecision],
            alignment_verdict: ReplyAlignmentVerdict | None = None,
            alignment_attempt: int = 0,
    ) -> RuntimeAttemptResult:
        with trace_span("answerability.assess"):
            answerability = AnswerabilityGate().assess(
                request=request,
                canonical_context=canonical_context,
                domain_context=domain_context,
                plan=plan,
                policy=policy,
                evidence_facts=evidence_facts,
            )
        with trace_span("decision.decide"):
            directive = DecisionEngine().decide(
                plan,
                business_facts,
                evidence_facts,
                request,
                policy,
                domain_context,
            )
        forced_directive = directive_from_answerability(answerability, plan)
        if forced_directive is not None:
            directive = forced_directive
        if (
            directive.mode == "clarification"
            and answerability.recommended_response_mode != "clarify"
        ):
            answerability = answerability.model_copy(
                update={
                    "can_answer": False,
                    "ambiguity": "other",
                    "recommended_response_mode": "clarify",
                    "user_facing_reason": directive.text
                    or answerability.user_facing_reason,
                }
            )
        guardrail_decisions = [
            *guardrail_decisions,
            *_guardrail_decisions_from_directive(directive, plan, business_facts),
        ]
        trace_event(
            "state.directive_selected",
            mode=directive.mode,
            reply_kind=directive.reply_kind,
            requires_knowledge_composer=directive.requires_knowledge_composer,
            composer_stage=directive.composer_stage,
        )
        response, composer_output = await self._compose_or_render_response(
            request=request,
            canonical_context=canonical_context,
            domain_context=domain_context,
            policy=policy,
            model_family=model_family,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            plan=plan,
            plan_validation=plan_validation,
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            answerability=answerability,
            directive=directive,
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
            guardrail_decisions=guardrail_decisions,
        )
        with trace_span("reply.validate_attempt"):
            return self._validated_attempt(
                plan=plan,
                plan_validation=plan_validation,
                preflight=preflight,
                evidence_facts=evidence_facts,
                business_facts=business_facts,
                domain_context=domain_context,
                answerability=answerability,
                directive=directive,
                response=response,
                policy=policy,
                guardrail_decisions=guardrail_decisions,
                composer_output=composer_output,
            )

    async def _compose_or_render_response(
            self,
            *,
            request: ReplyRequest,
            canonical_context: CanonicalContext,
            domain_context: DomainContext,
            policy: PolicyManifest,
            model_family,
            intent_gate: IntentGateResult,
            history: list,
            action_history: list[ActionLedgerRecord],
            prompt_programs: list[PromptProgram],
            llm_executions: list[dict],
            plan: ExecutionPlan,
            plan_validation: PlanValidationResult,
            preflight: AdapterPreflightSnapshot,
            evidence_facts: list,
            business_facts: BusinessFacts,
            answerability: AnswerabilityAssessment,
            directive: ResponseDirective,
            alignment_verdict: ReplyAlignmentVerdict | None = None,
            alignment_attempt: int = 0,
            guardrail_decisions: list[GuardrailDecision] | None = None,
    ) -> tuple[ReplyResponse, ComposerReplyOutput | None]:
        if not directive.requires_knowledge_composer:
            with trace_span("reply.render_directive"):
                return (
                    render_directive(
                        directive,
                        plan,
                        business_facts,
                        evidence_facts,
                    ),
                    None,
                )

        composer_stage = directive.composer_stage or "knowledge_composer"
        with trace_span("composer.project_context", stage=composer_stage):
            composer_context = self._project_context(
                stage=composer_stage,
                request=request,
                canonical_context=canonical_context,
                domain_context=domain_context,
                policy=policy,
                intent_gate=intent_gate,
                execution_plan=plan,
                plan_validation=plan_validation,
                preflight=preflight,
                evidence_facts=evidence_facts,
                business_facts=business_facts,
                answerability_assessment=answerability,
                guardrail_decisions=guardrail_decisions or [],
                history=history,
                action_history=action_history,
                alignment_verdict=alignment_verdict,
                alignment_attempt=alignment_attempt,
            )
        with trace_span("composer.assemble_prompt", stage=composer_stage):
            composer_program = select_prompt_program(
                PromptAssemblyContext(
                    stage=composer_stage,
                    model_family=model_family,
                    request=request,
                    canonical_context=canonical_context,
                    model_visible_context=composer_context,
                    domain_context=domain_context,
                    policy=policy,
                    intent_gate=intent_gate,
                    execution_plan=plan,
                    plan_validation=plan_validation,
                    preflight=preflight,
                    evidence_facts=evidence_facts,
                    business_facts=business_facts,
                    answerability_assessment=answerability,
                    guardrail_decisions=guardrail_decisions or [],
                    history=history,
                    action_history=action_history,
                    alignment_verdict=alignment_verdict,
                    alignment_attempt=alignment_attempt,
                )
        )
        prompt_programs.append(composer_program)
        try:
            with trace_span("composer.build_agent", stage=composer_stage):
                composer_agent = self._build_agent(composer_stage)
            result, composer_execution = await run_crewai_kickoff(
                composer_agent,
                composer_program,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            llm_executions.append(composer_execution)
        except asyncio.TimeoutError as exc:
            raise AgentRuntimeError("CrewAI composer timed out") from exc
        except Exception as exc:
            raise AgentRuntimeError("CrewAI composer failed") from exc

        with trace_span("composer.coerce_response"):
            composer_output = coerce_composer_output(result)
            response = (
                composer_output.to_reply_response()
                if composer_output is not None
                else coerce_agent_response(result)
            )
        if response is None:
            raise AgentRuntimeError("CrewAI composer returned an invalid ReplyResponse contract")
        if directive.mode == "action" and directive.action_intents:
            reply_text = remove_pre_execution_send_claims(response.reply.text)
            reply = response.reply.model_copy(update={"text": reply_text})
            if composer_output is not None:
                composer_output = composer_output.model_copy(update={"reply": reply})
            with trace_span("reply.render_action_from_composer"):
                rendered = render_directive(
                    directive.model_copy(
                        update={
                            "text": reply_text,
                            "requires_knowledge_composer": False,
                            "composer_stage": None,
                        }
                    ),
                    plan,
                    business_facts,
                    evidence_facts,
                )
            return ReplyResponse(reply=reply, actions=rendered.actions), composer_output
        return response, composer_output

    def _validated_attempt(
            self,
            *,
            plan: ExecutionPlan,
            plan_validation: PlanValidationResult,
            preflight: AdapterPreflightSnapshot,
            evidence_facts: list,
            business_facts: BusinessFacts,
            domain_context: DomainContext,
            directive: ResponseDirective,
            response: ReplyResponse,
            policy: PolicyManifest,
            guardrail_decisions: list[GuardrailDecision] | None = None,
            answerability: AnswerabilityAssessment | None = None,
            composer_output: ComposerReplyOutput | None = None,
    ) -> RuntimeAttemptResult:
        response = ensure_response_ids(response)
        guardrail_decisions = list(guardrail_decisions or [])
        output_decision = output_guard(
            response=response,
            directive=directive,
            plan=plan,
            policy=policy,
            evidence_facts=evidence_facts,
            domain_context=domain_context,
            composer_output=composer_output,
        )
        guardrail_decisions.append(output_decision)
        if output_decision.outcome == "abstain" and response.reply.kind == "answer":
            directive = ResponseDirective(
                mode="unable",
                reply_kind="unable_to_answer",
                text=abstention_response_text(),
                reason_code=output_decision.reason_code,
            )
            response = ensure_response_ids(
                ReplyResponse(
                    response_id=response.response_id,
                    reply=PrimaryReply(
                        kind="unable_to_answer",
                        text=directive.text,
                        mentions=[],
                    ),
                    actions=[],
                )
            )
        reply_validation = validate_reply(
            response,
            directive,
            plan,
            business_facts,
            evidence_facts,
            policy,
            domain_context=domain_context,
            composer_output=composer_output,
            output_decision=output_decision,
        )
        trace_event(
            "state.reply_validated",
            valid=reply_validation.valid,
            issue_count=len(reply_validation.issues),
            error=_reply_validation_error_summary(reply_validation)
            if not reply_validation.valid
            else "",
            output_guard_outcome=output_decision.outcome,
        )
        return RuntimeAttemptResult(
            plan=plan,
            plan_validation=plan_validation,
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            domain_context=domain_context,
            answerability=answerability or default_answerability_for_plan(plan),
            directive=directive,
            response=response,
            reply_validation=reply_validation,
            guardrail_decisions=guardrail_decisions,
            composer_output=composer_output,
        )

    async def _ensure_aligned_response(
            self,
            *,
            request: ReplyRequest,
            canonical_context: CanonicalContext,
            domain_context: DomainContext,
            policy: PolicyManifest,
            model_family,
            intent_gate: IntentGateResult,
            history: list,
            action_history: list[ActionLedgerRecord],
            prompt_programs: list[PromptProgram],
            llm_executions: list[dict],
            alignment_verdicts: list[ReplyAlignmentVerdict],
            alignment_remediations: list[dict],
            candidate: RuntimeAttemptResult,
    ) -> RuntimeAttemptResult:
        return await ensure_aligned_response(
            self,
            request=request,
            canonical_context=canonical_context,
            domain_context=domain_context,
            policy=policy,
            model_family=model_family,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            alignment_verdicts=alignment_verdicts,
            alignment_remediations=alignment_remediations,
            candidate=candidate,
        )

    async def _verify_reply_alignment(
            self,
            *,
            request: ReplyRequest,
            canonical_context: CanonicalContext,
            policy: PolicyManifest,
            model_family,
            intent_gate: IntentGateResult,
            history: list,
            action_history: list[ActionLedgerRecord],
            prompt_programs: list[PromptProgram],
            llm_executions: list[dict],
            candidate: RuntimeAttemptResult,
            attempt: int,
    ) -> ReplyAlignmentVerdict:
        if self.alignment_verifier is not None:
            with trace_span("alignment.external_verifier", attempt=attempt):
                verdict = await self.alignment_verifier.verify(
                    request=request,
                    canonical_context=canonical_context,
                    domain_context=candidate.domain_context,
                    plan=candidate.plan,
                    directive=candidate.directive,
                    evidence_facts=candidate.evidence_facts,
                    business_facts=candidate.business_facts,
                    response=candidate.response,
                    guardrail_decisions=candidate.guardrail_decisions,
                    attempt=attempt,
                )
            return ReplyAlignmentVerdict.model_validate(verdict)

        with trace_span("alignment.project_context", attempt=attempt):
            verifier_context = self._project_context(
                stage="alignment_verifier",
                request=request,
                canonical_context=canonical_context,
                domain_context=candidate.domain_context,
                policy=policy,
                intent_gate=intent_gate,
                execution_plan=candidate.plan,
                plan_validation=candidate.plan_validation,
                preflight=candidate.preflight,
                evidence_facts=candidate.evidence_facts,
                business_facts=candidate.business_facts,
                answerability_assessment=candidate.answerability,
                guardrail_decisions=candidate.guardrail_decisions,
                history=history,
                action_history=action_history,
                candidate_response=candidate.response,
                alignment_attempt=attempt,
            )
        with trace_span("alignment.assemble_prompt", attempt=attempt):
            verifier_program = select_prompt_program(
                PromptAssemblyContext(
                    stage="alignment_verifier",
                    model_family=model_family,
                    request=request,
                    canonical_context=canonical_context,
                    model_visible_context=verifier_context,
                    domain_context=candidate.domain_context,
                    policy=policy,
                    intent_gate=intent_gate,
                    execution_plan=candidate.plan,
                    plan_validation=candidate.plan_validation,
                    preflight=candidate.preflight,
                    evidence_facts=candidate.evidence_facts,
                    business_facts=candidate.business_facts,
                    answerability_assessment=candidate.answerability,
                    guardrail_decisions=candidate.guardrail_decisions,
                    history=history,
                    action_history=action_history,
                    candidate_response=candidate.response,
                    alignment_attempt=attempt,
                )
        )
        prompt_programs.append(verifier_program)
        try:
            with trace_span("alignment.build_agent", attempt=attempt):
                verifier_agent = self._build_alignment_verifier_agent()
            result, verifier_execution = await run_crewai_kickoff(
                verifier_agent,
                verifier_program,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            llm_executions.append(verifier_execution)
        except asyncio.TimeoutError as exc:
            raise AgentRuntimeError("CrewAI alignment verifier timed out") from exc
        except Exception as exc:
            raise AgentRuntimeError("CrewAI alignment verifier failed") from exc

        with trace_span("alignment.coerce_verdict", attempt=attempt):
            verdict = coerce_alignment_verdict(result)
        if verdict is None:
            raise AgentRuntimeError(
                "CrewAI alignment verifier returned an invalid ReplyAlignmentVerdict contract"
            )
        trace_event(
            "state.alignment_verdict",
            attempt=attempt,
            aligned=verdict.aligned,
            safe_to_return=verdict.safe_to_return,
            remediation=verdict.remediation,
            failure_code=verdict.failure_code,
        )
        return verdict

    def _fallback_attempt(
            self,
            candidate: RuntimeAttemptResult,
            policy: PolicyManifest,
            *,
            kind: str,
            reason_code: str,
    ) -> RuntimeAttemptResult:
        if candidate.plan.compliance.is_compliant is False:
            return candidate
        if kind == "clarification":
            directive = ResponseDirective(
                mode="clarification",
                reply_kind="clarification",
                text="我需要再确认一下你具体想要哪类内容。",
                reason_code=reason_code,
            )
        else:
            directive = ResponseDirective(
                mode="unable",
                reply_kind="unable_to_answer",
                text="当前没有足够证据，我先不展开。",
                reason_code=reason_code,
            )
        response = ReplyResponse(
            reply=PrimaryReply(
                kind=directive.reply_kind,
                text=directive.text,
                mentions=[],
            ),
            actions=[],
        )
        return self._validated_attempt(
            plan=candidate.plan,
            plan_validation=candidate.plan_validation,
            preflight=candidate.preflight,
            evidence_facts=candidate.evidence_facts,
            business_facts=candidate.business_facts,
            domain_context=candidate.domain_context,
            answerability=candidate.answerability,
            directive=directive,
            response=response,
            policy=policy,
            guardrail_decisions=candidate.guardrail_decisions,
        )

    def _record_audit_trace(
            self,
            *,
            request: ReplyRequest,
            policy: PolicyManifest,
            plan: ExecutionPlan,
            directive: ResponseDirective,
            plan_validation: PlanValidationResult,
            action_history: list[ActionLedgerRecord],
            canonical_context: CanonicalContext,
            domain_context: DomainContext,
            preflight: AdapterPreflightSnapshot,
            evidence_facts: list,
            business_facts: BusinessFacts,
            response: ReplyResponse,
            reply_validation,
            answerability_assessment: AnswerabilityAssessment | None = None,
            guardrail_decisions: list[GuardrailDecision] | None = None,
            intent_gate: IntentGateResult | None = None,
            prompt_programs: list[PromptProgram] | None = None,
            llm_executions: list[dict] | None = None,
            alignment_verdicts: list[ReplyAlignmentVerdict] | None = None,
            alignment_remediations: list[dict] | None = None,
            runtime_trace: dict | None = None,
    ) -> None:
        self.audit_store.record(
            build_audit_trace(
                request=request,
                settings=self.settings,
                policy=policy,
                plan=plan,
                directive=directive,
                plan_validation=plan_validation,
                action_history=action_history,
                canonical_context=canonical_context,
                domain_context=domain_context,
                preflight=preflight,
                evidence_facts=evidence_facts,
                business_facts=business_facts,
                answerability_assessment=answerability_assessment,
                response=response,
                reply_validation=reply_validation,
                guardrail_decisions=guardrail_decisions or [],
                intent_gate=intent_gate,
                prompt_programs=prompt_programs,
                llm_executions=llm_executions,
                alignment_verdicts=alignment_verdicts,
                alignment_remediations=alignment_remediations,
                runtime_trace=runtime_trace,
            )
        )

    def _project_context(self, **kwargs):
        try:
            with trace_span("context.project", stage=kwargs.get("stage")):
                return self.context_projection_manager.project_for_stage(**kwargs)
        except ProjectionLimitError as exc:
            raise AgentRuntimeError("model context projection exceeded token budget") from exc

    def _build_planner_agent(self):
        return self.agent_factory.build_planner_agent()

    def _build_agent(self, stage="knowledge_composer"):
        return self.agent_factory.build_composer_agent(stage)

    def _build_alignment_verifier_agent(self):
        return self.agent_factory.build_alignment_verifier_agent()


def _validation_error_summary(validation: PlanValidationResult) -> str:
    return "; ".join(issue.code for issue in validation.issues) or "unknown"


def _coerce_planner_plan_with_error(
    frame_result,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    policy: PolicyManifest,
    *,
    domain_context: DomainContext,
    history: list,
) -> tuple[ExecutionPlan | None, str]:
    try:
        plan = coerce_planner_plan(
            frame_result,
            request,
            canonical_context,
            policy,
            domain_context=domain_context,
            history=history,
        )
    except ValueError as exc:
        return None, f"PlanSpec compile error: {exc}"
    if plan is not None:
        return plan, ""
    return None, plan_spec_error_summary(frame_result)


def _planner_retry_program(program: PromptProgram, error_summary: str) -> PromptProgram:
    feedback = (
        "\n\n<prompt_layer id=\"ephemeral\">\n"
        "Previous PlanSpec validation error:\n"
        f"{error_summary}\n\n"
        "Rewrite the full PlanSpec JSON only. Do not output explanations. "
        "Fix the listed contract errors; every plan_units item must include "
        "evidence_contract_ref or evidence_contract from its selected capability.\n"
        "</prompt_layer>"
    )
    prompt_text = program.prompt_text + feedback
    layers = program.layers
    if "ephemeral" not in layers:
        layers = (*layers, "ephemeral")
    return replace(
        program,
        prompt_text=prompt_text,
        prompt_hash="sha256:" + hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        layers=layers,
    )


def _reply_validation_error_summary(validation) -> str:
    parts = []
    for issue in validation.issues:
        detail_values = [
            issue.metadata.get("contract_issue_code"),
            issue.metadata.get("unit_id"),
            issue.metadata.get("selected_capability_id"),
        ]
        detail = ",".join(str(value) for value in detail_values if value) or issue.message
        parts.append(f"{issue.code}({detail})")
    return "; ".join(dict.fromkeys(parts)) or "unknown"


def _guardrail_decisions_from_directive(
    directive: ResponseDirective,
    plan: ExecutionPlan,
    business_facts: BusinessFacts,
) -> list[GuardrailDecision]:
    if directive.reason_code != "ambiguous_action_resolve":
        return []
    candidates: list[str] = []
    action_types: list[str] = []
    for action in plan.action_intents:
        action_types.append(action.action_type)
        resolve_type = resolve_type_for_action(action.action_type)
        if resolve_type is None:
            continue
        candidates.extend(business_facts.resolve_state(resolve_type).candidates)
    return [
        GuardrailDecision(
            outcome="require_clarification",
            phase="output",
            reason_code="ambiguous_action_resolve",
            metadata={
                "action_types": _unique_strings(action_types),
                "candidates": _unique_strings(candidates),
            },
        )
    ]


def _unique_strings(values: list[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
