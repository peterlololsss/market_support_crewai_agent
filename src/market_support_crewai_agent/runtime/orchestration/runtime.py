from __future__ import annotations

import logging
from dataclasses import replace

from pydantic import JsonValue

from market_support_crewai_agent.runtime.state.action_ledger import (
    ActionLedger,
    ActionLedgerRecord,
)
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightService,
)
from market_support_crewai_agent.runtime.state.audit import (
    AuditStore,
)
from market_support_crewai_agent.runtime.state.runtime_trace import (
    trace_span,
)
from market_support_crewai_agent.runtime.domain.ontology import (
    DomainContext,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
    ConversationStore,
)
from market_support_crewai_agent.runtime.orchestration.alignment_loop import (
    ensure_aligned_response,
)
from market_support_crewai_agent.runtime.orchestration.lifecycle import run_reply_turn
from market_support_crewai_agent.runtime.orchestration.alignment import (
    fallback_attempt,
    verify_reply_alignment,
)
from market_support_crewai_agent.runtime.context.projection import (
    ContextProjectionManager,
)
from market_support_crewai_agent.runtime.context.pressure import ProjectionLimitError
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerifier,
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.domain.policy import (
    PolicyManifest,
)
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.context import (
    IntentGateResult,
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.llm.prompting.profiles import ModelFamily
from market_support_crewai_agent.runtime.llm.prompting.router import (
    model_family_from_settings,
    select_prompt_program,
)
from market_support_crewai_agent.runtime.orchestration.crewai_agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.orchestration.audit_trace import (
    record_audit_trace,
)
from market_support_crewai_agent.runtime.orchestration.composer import (
    compose_or_render_response,
)
from market_support_crewai_agent.runtime.orchestration.planner import (
    planner_retry_program,
)
from market_support_crewai_agent.runtime.orchestration.attempt import (
    build_candidate_from_evidence,
    build_candidate_from_plan,
)
from market_support_crewai_agent.runtime.orchestration.attempt_validation import (
    validated_attempt,
)
from market_support_crewai_agent.runtime.orchestration.workflow import (
    build_candidate_response,
)
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse
from market_support_crewai_agent.settings import Settings
from market_support_crewai_agent.runtime.turn import (
    AgentRuntimeError,
    AttemptResult,
    RuntimeDeps,
    build_runtime_deps,
)

logger = logging.getLogger(__name__)


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
    runtime = CrewAIReplyRuntime(
        build_runtime_deps(
            settings=settings,
            conversation_store=conversation_store,
            action_ledger=action_ledger,
            preflight_service=preflight_service,
            evidence_executor=evidence_executor,
            audit_store=audit_store,
            alignment_verifier=alignment_verifier,
        )
    )
    return await runtime.reply(request)


class CrewAIReplyRuntime:
    """CrewAI runtime boundary used by the FastAPI transport layer."""

    def __init__(
        self,
        settings: Settings | RuntimeDeps,
        conversation_store: ConversationStore | None = None,
        action_ledger: ActionLedger | None = None,
        preflight_service: AdapterPreflightService | None = None,
        evidence_executor: EvidenceExecutor | None = None,
        audit_store: AuditStore | None = None,
        alignment_verifier: ReplyAlignmentVerifier | None = None,
    ) -> None:
        deps = (
            settings
            if isinstance(settings, RuntimeDeps)
            else build_runtime_deps(
                settings=settings,
                conversation_store=conversation_store,
                action_ledger=action_ledger,
                preflight_service=preflight_service,
                evidence_executor=evidence_executor,
                audit_store=audit_store,
                alignment_verifier=alignment_verifier,
            )
        )
        self.settings = deps.settings
        self.conversation_store = deps.conversation_store
        self.action_ledger = deps.action_ledger
        self.preflight_service = deps.preflight_service
        self.evidence_executor = deps.evidence_executor
        self.audit_store = deps.audit_store
        self.alignment_verifier = deps.alignment_verifier
        self.agent_factory = CrewAIAgentFactory(deps.settings)
        self.context_projection_manager = ContextProjectionManager.from_settings(
            deps.settings
        )

    async def reply(self, request: ReplyRequest) -> ReplyResponse:
        return await run_reply_turn(self, request)

    async def _build_candidate_response(self, **kwargs) -> AttemptResult:
        return await build_candidate_response(self, **kwargs)

    async def _build_candidate_from_plan(self, **kwargs) -> AttemptResult:
        return await build_candidate_from_plan(self, **kwargs)

    async def _build_candidate_from_evidence(self, **kwargs) -> AttemptResult:
        return await build_candidate_from_evidence(self, **kwargs)

    async def _compose_or_render_response(self, **kwargs):
        return await compose_or_render_response(self, **kwargs)

    def _validated_attempt(self, **kwargs) -> AttemptResult:
        return validated_attempt(self, **kwargs)

    async def _ensure_aligned_response(
        self,
        *,
        request: ReplyRequest,
        domain_context: DomainContext,
        policy: PolicyManifest,
        model_family: ModelFamily,
        intent_gate: IntentGateResult,
        history: list[ConversationMessage],
        action_history: list[ActionLedgerRecord],
        prompt_programs: list[PromptProgram],
        llm_executions: list[dict[str, JsonValue]],
        alignment_verdicts: list[ReplyAlignmentVerdict],
        alignment_remediations: list[dict[str, JsonValue]],
        candidate: AttemptResult,
    ) -> AttemptResult:
        return await ensure_aligned_response(
            self,
            request=request,
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

    async def _verify_reply_alignment(self, **kwargs) -> ReplyAlignmentVerdict:
        return await verify_reply_alignment(self, **kwargs)

    def _fallback_attempt(self, *args, **kwargs) -> AttemptResult:
        return fallback_attempt(self, *args, **kwargs)

    def _record_audit_trace(self, **kwargs) -> None:
        record_audit_trace(self, **kwargs)

    def _project_context(self, **kwargs):
        try:
            with trace_span("context.project", stage=kwargs.get("stage")):
                return self.context_projection_manager.project_for_stage(**kwargs)
        except ProjectionLimitError as exc:
            raise AgentRuntimeError(
                "model context projection exceeded token budget"
            ) from exc

    def _build_planner_agent(self):
        return self.agent_factory.build_planner_agent()

    def _build_planner_fallback_agent(self):
        return self.agent_factory.build_planner_fallback_agent()

    def _planner_fallback_program(
        self,
        ctx: PromptAssemblyContext,
        *,
        error_summary: str | None = None,
    ) -> PromptProgram:
        program = select_prompt_program(
            replace(ctx, model_family=model_family_from_settings(self.settings))
        )
        if error_summary:
            return planner_retry_program(program, error_summary)
        return program

    def _build_agent(self, stage="knowledge_composer"):
        return self.agent_factory.build_composer_agent(stage)

    def _build_alignment_verifier_agent(self):
        return self.agent_factory.build_alignment_verifier_agent()
