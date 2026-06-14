from __future__ import annotations

import asyncio
from time import perf_counter

from market_support_crewai_agent.runtime.action_ledger import (
    ActionLedger,
    ActionLedgerRecord,
    get_action_ledger,
)
from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.audit import (
    AuditStore,
    build_audit_trace,
    get_audit_store,
)
from market_support_crewai_agent.runtime.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.canonicalization import (
    CanonicalContext,
    canonicalize_request,
)
from market_support_crewai_agent.runtime.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.document_mcp import DocumentMcpEvidenceService
from market_support_crewai_agent.runtime.decision import (
    DecisionEngine,
    ResponseDirective,
)
from market_support_crewai_agent.runtime.evidence_executor import EvidenceExecutor
from market_support_crewai_agent.runtime.guardrails import (
    ReplyContractError,
    validate_reply,
)
from market_support_crewai_agent.runtime.input_guardrails import (
    validate_reply_request_input,
)
from market_support_crewai_agent.runtime.planning import (
    ExecutionPlan,
    IntentFrame,
    PlanValidationResult,
    compile_intent_frame,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.policy import (
    PolicyManifest,
    compile_policy,
    ledger_summary_from_action_history,
)
from market_support_crewai_agent.runtime.prompt_profiles import (
    PromptProfile,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.runtime.prompt_assembler import PromptProgram
from market_support_crewai_agent.runtime.prompt_context import IntentGateResult, PromptAssemblyContext
from market_support_crewai_agent.runtime.prompt_router import (
    model_family_from_settings,
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.runtime.response_renderer import render_directive
from market_support_crewai_agent.runtime.response_ids import ensure_response_ids
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse
from market_support_crewai_agent.settings import Settings, get_settings


class AgentRuntimeError(RuntimeError):
    """Raised when the CrewAI runtime cannot produce a valid reply."""


_DEFAULT_SETTINGS = get_settings()
_DEFAULT_CONVERSATION_STORE = ConversationStore.from_settings(_DEFAULT_SETTINGS)
_DEFAULT_ACTION_LEDGER = get_action_ledger()
_DEFAULT_PREFLIGHT_SERVICE = AdapterPreflightService()
_DEFAULT_AUDIT_STORE = get_audit_store()
_DEFAULT_DOCUMENT_EVIDENCE_SERVICE = DocumentMcpEvidenceService(_DEFAULT_SETTINGS)


async def build_reply(
        request: ReplyRequest,
        settings: Settings | None = None,
        conversation_store: ConversationStore | None = None,
        action_ledger: ActionLedger | None = None,
        preflight_service: AdapterPreflightService | None = None,
        evidence_executor: EvidenceExecutor | None = None,
        audit_store: AuditStore | None = None,
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
        ),
        audit_store or _DEFAULT_AUDIT_STORE,
    )
    return await runtime.reply(request)


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
        )
        self.audit_store = audit_store or get_audit_store()

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
        intent_gate = route_intent(request, canonical_context, policy)
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
            )
        )
        planner_agent = self._build_planner_agent()
        llm_executions: list[dict] = []
        prompt_programs: list[PromptProgram] = [planner_program]
        try:
            frame_result, planner_execution = await _run_crewai_kickoff(
                planner_agent,
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

        plan = compile_intent_frame(
            intent_frame,
            request,
            canonical_context,
            policy,
        )
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
        preflight = evidence_result.preflight
        evidence_facts = evidence_result.evidence_facts
        business_facts = evidence_result.business_facts
        directive = DecisionEngine().decide(
            plan,
            business_facts,
            evidence_facts,
            request,
            policy,
        )

        if directive.requires_knowledge_composer:
            composer_agent = self._build_agent()
            composer_program = select_prompt_program(
                PromptAssemblyContext(
                    stage="knowledge_composer",
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
                )
            )
            prompt_programs.append(composer_program)
            try:
                result, composer_execution = await _run_crewai_kickoff(
                    composer_agent,
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
        else:
            response = render_directive(
                directive,
                plan,
                business_facts,
                evidence_facts,
            )

        response = ensure_response_ids(response)
        reply_validation = validate_reply(
            response,
            directive,
            plan,
            business_facts,
            evidence_facts,
            policy,
        )
        self._record_audit_trace(
            request=request,
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
        )
        if not reply_validation.valid:
            raise ReplyContractError(
                "rendered reply failed validation: {}".format(
                    _reply_validation_error_summary(reply_validation)
                )
            )

        self.conversation_store.save_turn(
            request.conversation_key,
            request.message,
            _compact_assistant_result(response),
        )
        return response

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

    def _build_agent(self):
        return self._build_crewai_agent(
            role="Market Support Reply Composer",
            goal=(
                "Compose the final ReplyResponse from a validated plan and "
                "deterministic evidence for the external WeWork adapter."
            ),
            backstory=(
                "You are the external agent brain for a market support workflow. "
                "You use the validated plan and evidence facts, and never send "
                "WeWork messages directly."
            ),
            inject_date=True,
            prompt_profile=prompt_profile_by_stage(
                "knowledge_composer",
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


def _compact_assistant_result(response: ReplyResponse) -> str:
    return response.model_dump_json(exclude_none=True)


def _validation_error_summary(validation: PlanValidationResult) -> str:
    return "; ".join(issue.code for issue in validation.issues) or "unknown"


def _reply_validation_error_summary(validation) -> str:
    return "; ".join(issue.code for issue in validation.issues) or "unknown"
