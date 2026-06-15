from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from time import perf_counter

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
from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.capabilities import capability_by_name
from market_support_crewai_agent.runtime.domain.canonicalization import (
    CanonicalContext,
    canonicalize_request,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.evidence.document_mcp import DocumentMcpEvidenceService
from market_support_crewai_agent.runtime.orchestration.decision import (
    DecisionEngine,
    ResponseDirective,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.validation.guardrails import (
    ReplyContractError,
    ValidationResult,
    validate_reply,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerifier,
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.validation.input_guardrails import (
    validate_reply_request_input,
)
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    IntentFrame,
    PlanValidationResult,
    compile_intent_frame,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import (
    PolicyManifest,
    compile_policy,
    ledger_summary_from_action_history,
)
from market_support_crewai_agent.runtime.llm.prompt_profiles import (
    PromptProfile,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.runtime.llm.prompt_assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompt_context import IntentGateResult, PromptAssemblyContext
from market_support_crewai_agent.runtime.llm.prompt_router import (
    model_family_from_settings,
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.runtime.orchestration.response_renderer import render_directive
from market_support_crewai_agent.runtime.orchestration.response_ids import ensure_response_ids
from market_support_crewai_agent.schemas import PrimaryReply, ReplyRequest, ReplyResponse
from market_support_crewai_agent.settings import Settings, get_settings


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
    directive: ResponseDirective
    response: ReplyResponse
    reply_validation: ValidationResult


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
        )
        self.audit_store = audit_store or get_audit_store()
        self.alignment_verifier = alignment_verifier

    async def reply(self, request: ReplyRequest) -> ReplyResponse:
        validate_reply_request_input(request, self.settings)
        if not self.settings.llm_api_key:
            raise AgentRuntimeError("YANFU_LLM_API_KEY is not configured")

        history = self.conversation_store.get_recent(request.conversation_key)
        action_history = self.action_ledger.recent_executed_for_conversation(
            request.conversation_key,
            limit=20,
        )
        canonical_context = canonicalize_request(request)
        model_family = model_family_from_settings(self.settings)
        policy = compile_policy(
            request,
            ledger_summary=ledger_summary_from_action_history(action_history),
            doc_mcp_enabled=bool(
                self.settings.doc_mcp_enabled and self.settings.doc_mcp_base_url
            ),
            doc_mcp_allowed_channel_types=self.settings.doc_mcp_allowed_channel_types,
        )
        intent_gate = route_intent(request, canonical_context, policy, history=history)
        llm_executions: list[dict] = []
        prompt_programs: list[PromptProgram] = []
        alignment_verdicts: list[ReplyAlignmentVerdict] = []
        alignment_remediations: list[dict] = []

        candidate = await self._build_candidate_response(
            request=request,
            canonical_context=canonical_context,
            policy=policy,
            model_family=model_family,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
        )
        if candidate.reply_validation.valid and self.settings.reply_alignment_verifier_enabled:
            candidate = await self._ensure_aligned_response(
                request=request,
                canonical_context=canonical_context,
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
        self._record_audit_trace(
            request=request,
            policy=policy,
            plan=candidate.plan,
            directive=candidate.directive,
            plan_validation=candidate.plan_validation,
            action_history=action_history,
            canonical_context=canonical_context,
            preflight=candidate.preflight,
            evidence_facts=candidate.evidence_facts,
            business_facts=candidate.business_facts,
            response=candidate.response,
            reply_validation=candidate.reply_validation,
            intent_gate=intent_gate,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            alignment_verdicts=alignment_verdicts,
            alignment_remediations=alignment_remediations,
        )
        if not candidate.reply_validation.valid:
            raise ReplyContractError(
                "rendered reply failed validation: {}".format(
                    _reply_validation_error_summary(candidate.reply_validation)
                )
            )

        self.conversation_store.save_turn(
            request.conversation_key,
            request.message,
            _compact_assistant_result(candidate.response, candidate.plan),
        )
        return candidate.response

    async def _build_candidate_response(
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
            alignment_verdict: ReplyAlignmentVerdict | None = None,
            alignment_attempt: int = 0,
    ) -> RuntimeAttemptResult:
        planner_program = select_prompt_program(
            PromptAssemblyContext(
                stage="planner_intent",
                model_family=model_family,
                request=request,
                canonical_context=canonical_context,
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
            frame_result, planner_execution = await _run_crewai_kickoff(
                self._build_planner_agent(),
                planner_program,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            llm_executions.append(planner_execution)
        except asyncio.TimeoutError as exc:
            raise AgentRuntimeError("CrewAI planner timed out") from exc
        except Exception as exc:
            raise AgentRuntimeError("CrewAI planner failed") from exc

        intent_frame = _coerce_intent_frame(frame_result)
        if intent_frame is None:
            raise AgentRuntimeError("CrewAI planner returned an invalid IntentFrame contract")
        intent_frame = _resolve_followup_intent_frame(
            intent_frame,
            history,
            canonical_context,
        )

        plan = compile_intent_frame(
            intent_frame,
            request,
            canonical_context,
            policy,
        )
        return await self._build_candidate_from_plan(
            request=request,
            canonical_context=canonical_context,
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
        plan_validation = validate_execution_plan(plan, policy)
        if not plan_validation.valid:
            raise AgentRuntimeError(
                "compiled execution plan failed validation: {}".format(
                    _validation_error_summary(plan_validation)
                )
            )

        evidence_result = await self.evidence_executor.execute(
            request,
            canonical_context,
            plan,
            policy,
            action_history=action_history,
        )
        return await self._build_candidate_from_evidence(
            request=request,
            canonical_context=canonical_context,
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
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
        )

    async def _build_candidate_from_evidence(
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
            plan: ExecutionPlan,
            plan_validation: PlanValidationResult,
            preflight: AdapterPreflightSnapshot,
            evidence_facts: list,
            business_facts: BusinessFacts,
            alignment_verdict: ReplyAlignmentVerdict | None = None,
            alignment_attempt: int = 0,
    ) -> RuntimeAttemptResult:
        directive = DecisionEngine().decide(
            plan,
            business_facts,
            evidence_facts,
            request,
            policy,
        )
        response = await self._compose_or_render_response(
            request=request,
            canonical_context=canonical_context,
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
            directive=directive,
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
        )
        return self._validated_attempt(
            plan=plan,
            plan_validation=plan_validation,
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            directive=directive,
            response=response,
            policy=policy,
        )

    async def _compose_or_render_response(
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
            plan: ExecutionPlan,
            plan_validation: PlanValidationResult,
            preflight: AdapterPreflightSnapshot,
            evidence_facts: list,
            business_facts: BusinessFacts,
            directive: ResponseDirective,
            alignment_verdict: ReplyAlignmentVerdict | None = None,
            alignment_attempt: int = 0,
    ) -> ReplyResponse:
        if not directive.requires_knowledge_composer:
            return render_directive(
                directive,
                plan,
                business_facts,
                evidence_facts,
            )

        composer_stage = directive.composer_stage or "knowledge_composer"
        composer_program = select_prompt_program(
            PromptAssemblyContext(
                stage=composer_stage,
                model_family=model_family,
                request=request,
                canonical_context=canonical_context,
                policy=policy,
                intent_gate=intent_gate,
                execution_plan=plan,
                plan_validation=plan_validation,
                preflight=preflight,
                evidence_facts=evidence_facts,
                business_facts=business_facts,
                history=history,
                action_history=action_history,
                alignment_verdict=alignment_verdict,
                alignment_attempt=alignment_attempt,
            )
        )
        prompt_programs.append(composer_program)
        try:
            result, composer_execution = await _run_crewai_kickoff(
                self._build_agent(composer_stage),
                composer_program,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            llm_executions.append(composer_execution)
        except asyncio.TimeoutError as exc:
            raise AgentRuntimeError("CrewAI composer timed out") from exc
        except Exception as exc:
            raise AgentRuntimeError("CrewAI composer failed") from exc

        response = _coerce_agent_response(result)
        if response is None:
            raise AgentRuntimeError("CrewAI composer returned an invalid ReplyResponse contract")
        return response

    def _validated_attempt(
            self,
            *,
            plan: ExecutionPlan,
            plan_validation: PlanValidationResult,
            preflight: AdapterPreflightSnapshot,
            evidence_facts: list,
            business_facts: BusinessFacts,
            directive: ResponseDirective,
            response: ReplyResponse,
            policy: PolicyManifest,
    ) -> RuntimeAttemptResult:
        response = ensure_response_ids(response)
        reply_validation = validate_reply(
            response,
            directive,
            plan,
            business_facts,
            evidence_facts,
            policy,
        )
        return RuntimeAttemptResult(
            plan=plan,
            plan_validation=plan_validation,
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            directive=directive,
            response=response,
            reply_validation=reply_validation,
        )

    async def _ensure_aligned_response(
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
            alignment_verdicts: list[ReplyAlignmentVerdict],
            alignment_remediations: list[dict],
            candidate: RuntimeAttemptResult,
    ) -> RuntimeAttemptResult:
        replan_count = 0
        refetch_count = 0
        recompose_count = 0
        total_remediations = 0

        while True:
            try:
                verdict = await self._verify_reply_alignment(
                    request=request,
                    canonical_context=canonical_context,
                    policy=policy,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    candidate=candidate,
                    attempt=len(alignment_verdicts),
                )
            except Exception as exc:
                alignment_remediations.append(
                    {
                        "remediation": "return_unable",
                        "reason": "verifier_failed",
                        "error": _safe_short_text(exc),
                    }
                )
                return self._fallback_attempt(
                    candidate,
                    policy,
                    kind="unable",
                    reason_code="alignment_verifier_failed",
                )

            alignment_verdicts.append(verdict)
            if verdict.aligned and verdict.safe_to_return:
                return candidate

            if total_remediations >= self.settings.reply_alignment_max_total_remediations:
                alignment_remediations.append(
                    {
                        "remediation": "return_unable",
                        "reason": "remediation_limit_exceeded",
                        "failure_code": verdict.failure_code,
                    }
                )
                return self._fallback_attempt(
                    candidate,
                    policy,
                    kind="unable",
                    reason_code="alignment_remediation_limit",
                )

            if (
                verdict.remediation == "replan"
                and replan_count < self.settings.reply_alignment_max_replans
            ):
                replan_count += 1
                total_remediations += 1
                alignment_remediations.append(
                    {
                        "remediation": "replan",
                        "failure_code": verdict.failure_code,
                    }
                )
                candidate = await self._build_candidate_response(
                    request=request,
                    canonical_context=canonical_context,
                    policy=policy,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    alignment_verdict=verdict,
                    alignment_attempt=len(alignment_verdicts),
                )
                if not candidate.reply_validation.valid:
                    return candidate
                continue

            if (
                verdict.remediation == "refetch_document_context"
                and refetch_count < self.settings.reply_alignment_max_evidence_refetches
                and candidate.plan.response_mode == "knowledge_answer"
                and "document_context" in candidate.plan.capabilities
                and "document_context" in policy.allowed_capabilities
                and (verdict.refined_evidence_query or "").strip()
            ):
                refetch_count += 1
                total_remediations += 1
                alignment_remediations.append(
                    {
                        "remediation": "refetch_document_context",
                        "failure_code": verdict.failure_code,
                    }
                )
                candidate = await self._build_candidate_from_plan(
                    request=request,
                    canonical_context=canonical_context,
                    policy=policy,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    plan=candidate.plan.model_copy(
                        update={"evidence_query": verdict.refined_evidence_query}
                    ),
                    alignment_verdict=verdict,
                    alignment_attempt=len(alignment_verdicts),
                )
                if not candidate.reply_validation.valid:
                    return candidate
                continue

            if (
                verdict.remediation == "recompose"
                and recompose_count < self.settings.reply_alignment_max_recomposes
                and candidate.directive.requires_knowledge_composer
            ):
                recompose_count += 1
                total_remediations += 1
                alignment_remediations.append(
                    {
                        "remediation": "recompose",
                        "failure_code": verdict.failure_code,
                    }
                )
                response = await self._compose_or_render_response(
                    request=request,
                    canonical_context=canonical_context,
                    policy=policy,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    plan=candidate.plan,
                    plan_validation=candidate.plan_validation,
                    preflight=candidate.preflight,
                    evidence_facts=candidate.evidence_facts,
                    business_facts=candidate.business_facts,
                    directive=candidate.directive,
                    alignment_verdict=verdict,
                    alignment_attempt=len(alignment_verdicts),
                )
                candidate = self._validated_attempt(
                    plan=candidate.plan,
                    plan_validation=candidate.plan_validation,
                    preflight=candidate.preflight,
                    evidence_facts=candidate.evidence_facts,
                    business_facts=candidate.business_facts,
                    directive=candidate.directive,
                    response=response,
                    policy=policy,
                )
                if not candidate.reply_validation.valid:
                    return candidate
                continue

            if verdict.remediation == "return_clarification":
                alignment_remediations.append(
                    {
                        "remediation": "return_clarification",
                        "failure_code": verdict.failure_code,
                    }
                )
                return self._fallback_attempt(
                    candidate,
                    policy,
                    kind="clarification",
                    reason_code="alignment_return_clarification",
                )

            alignment_remediations.append(
                {
                    "remediation": "return_unable",
                    "reason": "unsupported_or_exhausted_remediation",
                    "suggested_remediation": verdict.remediation,
                    "failure_code": verdict.failure_code,
                }
            )
            return self._fallback_attempt(
                candidate,
                policy,
                kind="unable",
                reason_code="alignment_return_unable",
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
            verdict = await self.alignment_verifier.verify(
                request=request,
                canonical_context=canonical_context,
                plan=candidate.plan,
                directive=candidate.directive,
                evidence_facts=candidate.evidence_facts,
                business_facts=candidate.business_facts,
                response=candidate.response,
                attempt=attempt,
            )
            return ReplyAlignmentVerdict.model_validate(verdict)

        verifier_program = select_prompt_program(
            PromptAssemblyContext(
                stage="alignment_verifier",
                model_family=model_family,
                request=request,
                canonical_context=canonical_context,
                policy=policy,
                intent_gate=intent_gate,
                execution_plan=candidate.plan,
                plan_validation=candidate.plan_validation,
                preflight=candidate.preflight,
                evidence_facts=candidate.evidence_facts,
                business_facts=candidate.business_facts,
                history=history,
                action_history=action_history,
                candidate_response=candidate.response,
                alignment_attempt=attempt,
            )
        )
        prompt_programs.append(verifier_program)
        try:
            result, verifier_execution = await _run_crewai_kickoff(
                self._build_alignment_verifier_agent(),
                verifier_program,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            llm_executions.append(verifier_execution)
        except asyncio.TimeoutError as exc:
            raise AgentRuntimeError("CrewAI alignment verifier timed out") from exc
        except Exception as exc:
            raise AgentRuntimeError("CrewAI alignment verifier failed") from exc

        verdict = _coerce_alignment_verdict(result)
        if verdict is None:
            raise AgentRuntimeError(
                "CrewAI alignment verifier returned an invalid ReplyAlignmentVerdict contract"
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
                text="当前没有足够证据安全回复，我先不展开。",
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
            directive=directive,
            response=response,
            policy=policy,
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
            preflight: AdapterPreflightSnapshot,
            evidence_facts: list,
            business_facts: BusinessFacts,
            response: ReplyResponse,
            reply_validation,
            intent_gate: IntentGateResult | None = None,
            prompt_programs: list[PromptProgram] | None = None,
            llm_executions: list[dict] | None = None,
            alignment_verdicts: list[ReplyAlignmentVerdict] | None = None,
            alignment_remediations: list[dict] | None = None,
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
                preflight=preflight,
                evidence_facts=evidence_facts,
                business_facts=business_facts,
                response=response,
                reply_validation=reply_validation,
                intent_gate=intent_gate,
                prompt_programs=prompt_programs,
                llm_executions=llm_executions,
                alignment_verdicts=alignment_verdicts,
                alignment_remediations=alignment_remediations,
            )
        )

    def _build_planner_agent(self):
        return self._build_crewai_agent(
            role="Market Support Reply Planner",
            goal=(
                "Interpret Chinese sales/support requests, evaluate compliance, "
                "and return a bounded IntentFrame for the deterministic harness."
            ),
            backstory=(
                "You plan the support workflow for Shanghai Yanfu Investment. "
                "You do not call tools, send messages, or produce final business facts."
            ),
            inject_date=True,
            prompt_profile=prompt_profile_by_stage(
                "planner_intent",
                model_family_from_settings(self.settings),
            ),
        )

    def _build_agent(self, stage="knowledge_composer"):
        return self._build_crewai_agent(
            role="Market Support Reply Composer",
            goal=(
                "Compose the final ReplyResponse from a validated plan and "
                "deterministic evidence for the external WeWork adapter."
            ),
            backstory=(
                "You are the external agent brain for a market support workflow. "
                "You use the validated plan and evidence facts."
            ),
            inject_date=True,
            prompt_profile=prompt_profile_by_stage(
                stage,
                model_family_from_settings(self.settings),
            ),
        )

    def _build_alignment_verifier_agent(self):
        return self._build_crewai_agent(
            role="Market Support Reply Alignment Verifier",
            goal=(
                "Judge whether the validated ReplyResponse semantically aligns "
                "with the current market support request."
            ),
            backstory=(
                "You are a bounded verifier. You return only a structured verdict "
                "and never call tools, send messages, or mutate actions."
            ),
            inject_date=False,
            prompt_profile=prompt_profile_by_stage(
                "alignment_verifier",
                model_family_from_settings(self.settings),
            ),
        )

    def _build_crewai_agent(
            self,
            *,
            role: str,
            goal: str,
            backstory: str,
            inject_date: bool,
            prompt_profile: PromptProfile | None = None,
    ):
        from crewai import Agent, LLM

        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=LLM(
                model=self.settings.llm_model,
                provider=self.settings.llm_provider,
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                temperature=(
                    prompt_profile.temperature
                    if prompt_profile is not None and prompt_profile.temperature is not None
                    else self.settings.llm_temperature
                ),
                max_tokens=(
                    prompt_profile.max_tokens
                    if prompt_profile is not None and prompt_profile.max_tokens is not None
                    else self.settings.llm_max_tokens
                ),
                timeout=self.settings.llm_timeout_seconds,
            ),
            allow_delegation=False,
            verbose=self.settings.crewai_verbose,
            max_iter=self.settings.crewai_max_iter,
            max_execution_time=self.settings.crewai_max_execution_time,
            max_retry_limit=self.settings.crewai_max_retry_limit,
            planning=False,
            inject_date=inject_date,
            date_format="%Y-%m-%d",
        )


async def _run_crewai_kickoff(
        agent,
        prompt_program: PromptProgram,
        *,
        timeout_seconds: float | None,
):
    started_at = perf_counter()
    result = await asyncio.wait_for(
        agent.kickoff_async(
            prompt_program.prompt_text,
            response_format=prompt_program.profile.response_model,
        ),
        timeout=timeout_seconds,
    )
    latency_ms = (perf_counter() - started_at) * 1000
    return result, _compact_crewai_execution(prompt_program, result, latency_ms)


def _compact_crewai_execution(
        prompt_program: PromptProgram,
        result,
        latency_ms: float,
) -> dict:
    prompt_profile = prompt_program.profile
    return {
        "stage": prompt_profile.stage,
        "prompt_profile_id": prompt_profile.id,
        "prompt_fragment_ids": list(prompt_program.fragment_ids),
        "prompt_hash": prompt_program.prompt_hash,
        "agent_role": str(getattr(result, "agent_role", "") or ""),
        "response_format": getattr(
            prompt_profile.response_model,
            "__name__",
            str(prompt_profile.response_model),
        ),
        "latency_ms": round(latency_ms, 3),
        "usage_metrics": _compact_usage_metrics(
            getattr(result, "usage_metrics", None)
        ),
        "pydantic_type": _pydantic_type_name(getattr(result, "pydantic", None)),
        "raw_length": len(str(getattr(result, "raw", "") or "")),
    }


def _compact_usage_metrics(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(key): _compact_usage_metrics(item)
            for key, item in value.items()
            if not str(key).lower().endswith(("key", "token_value", "secret"))
        }
    if isinstance(value, (list, tuple)):
        return [_compact_usage_metrics(item) for item in value[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _safe_short_text(value)


def _pydantic_type_name(value) -> str:
    if value is None:
        return ""
    return value.__class__.__name__


def _safe_short_text(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= 160:
        return text
    return text[:157] + "..."


def _coerce_intent_frame(result) -> IntentFrame | None:
    if result.pydantic is not None:
        try:
            return IntentFrame.model_validate(result.pydantic)
        except ValueError:
            return None

    try:
        return IntentFrame.model_validate_json(result.raw)
    except ValueError:
        return None


def _coerce_agent_response(result) -> ReplyResponse | None:
    if result.pydantic is not None:
        try:
            return ReplyResponse.model_validate(result.pydantic)
        except ValueError:
            return None

    try:
        return ReplyResponse.model_validate_json(result.raw)
    except ValueError:
        return None


def _coerce_alignment_verdict(result) -> ReplyAlignmentVerdict | None:
    if result.pydantic is not None:
        try:
            return ReplyAlignmentVerdict.model_validate(result.pydantic)
        except ValueError:
            return None

    try:
        return ReplyAlignmentVerdict.model_validate_json(result.raw)
    except ValueError:
        return None


def _compact_assistant_result(response: ReplyResponse, plan: ExecutionPlan) -> str:
    return json.dumps(
        {
            "contract_version": "reply-runtime-history",
            "reply_response": response.model_dump(mode="json", exclude_none=True),
            "pending_plan": _compact_pending_plan(plan),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _compact_pending_plan(plan: ExecutionPlan) -> dict[str, object] | None:
    if plan.response_mode != "clarification" and not plan.ambiguity_slots:
        return None
    return {
        "artifact_kind": plan.artifact_kind,
        "response_mode": plan.response_mode,
        "ambiguity_slots": list(plan.ambiguity_slots),
        "selected_strategy": plan.selected_strategy,
        "report_scope": getattr(plan, "report_scope", "none"),
        "capabilities": list(plan.capabilities),
    }


def _resolve_followup_intent_frame(
    frame: IntentFrame,
    history: list,
    canonical_context: CanonicalContext,
) -> IntentFrame:
    pending = _latest_pending_plan(history)
    if not pending:
        return frame

    pending_strategy = _clean_str(pending.get("selected_strategy"))
    current_strategy = (
        _clean_str(frame.selected_strategy)
        or _clean_str(canonical_context.selected_strategy)
    )
    slots = set(frame.ambiguity_slots)
    pending_slots = set(_string_list(pending.get("ambiguity_slots")))
    pending_artifact = _clean_str(pending.get("artifact_kind"))

    if (
        "strategy" in pending_slots
        and current_strategy
        and pending_artifact in _SEND_ARTIFACTS
    ):
        return _frame_for_followup_send(
            frame,
            artifact_kind=pending_artifact,
            selected_strategy=current_strategy,
            ambiguity_slots=[
                slot
                for slot in frame.ambiguity_slots
                if slot not in {"artifact", "strategy"}
            ],
        )

    if (
        "strategy" in slots
        and not current_strategy
        and pending_strategy
        and frame.artifact_kind in _SEND_ARTIFACTS
        and frame.action_intent == "send"
    ):
        return _frame_for_followup_send(
            frame,
            artifact_kind=frame.artifact_kind,
            selected_strategy=pending_strategy,
            ambiguity_slots=[slot for slot in frame.ambiguity_slots if slot != "strategy"],
        )

    return frame


_SEND_ARTIFACTS = frozenset({"material_pack", "weekly_report", "monthly_report"})


def _frame_for_followup_send(
    frame: IntentFrame,
    *,
    artifact_kind: str,
    selected_strategy: str,
    ambiguity_slots: list[str],
) -> IntentFrame:
    capability = capability_by_name(artifact_kind)
    if capability is None:
        return frame
    updates: dict[str, object] = {
        "artifact_kind": artifact_kind,
        "action_intent": "send",
        "selected_strategy": selected_strategy,
        "requested_capabilities": [capability.name],
        "ambiguity_slots": ambiguity_slots,
    }
    if artifact_kind == "material_pack":
        updates["report_scope"] = "none"
    elif artifact_kind in {"weekly_report", "monthly_report"}:
        updates["report_scope"] = "strategy"
    return frame.model_copy(update=updates)


def _latest_pending_plan(history: list) -> dict[str, object] | None:
    for message in reversed(history or []):
        if getattr(message, "role", None) != "assistant":
            continue
        content = getattr(message, "content", "")
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        pending = payload.get("pending_plan")
        if isinstance(pending, dict):
            return pending
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in (_clean_str(item) for item in value) if item]


def _clean_str(value: object) -> str:
    return str(value or "").strip()


def _validation_error_summary(validation: PlanValidationResult) -> str:
    return "; ".join(issue.code for issue in validation.issues) or "unknown"


def _reply_validation_error_summary(validation) -> str:
    return "; ".join(issue.code for issue in validation.issues) or "unknown"
