from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.context.stage_inputs import (
    PlannerPromptInputV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.observability.runtime_trace import (
    trace_event,
    trace_span,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.planning.planner_input import (
    PlannerRuntimeInputSourceV1,
    build_runtime_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.planning.planner_llm import (
    record_llm_failure,
    run_planner_kickoff_with_retry,
)
from market_support_crewai_agent.runtime.planning.planner_results import (
    PlannerFrameResult,
    coerce_planner_plan_with_error,
)
from market_support_crewai_agent.runtime.planning.planner_retry import (
    planner_alignment_replan_overlay,
    planner_schema_repair_allowed,
    planner_schema_repair_overlay,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology_models import DomainContextV1
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.context import (
    IntentGateResult,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
)
from market_support_crewai_agent.runtime.prompts.router import (
    model_family_from_settings,
    select_stage_input_prompt_program,
)
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.settings_model import Settings


class PlanCandidateBuilderV1(Protocol):
    async def __call__(
        self,
        *,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
        scope_authority: BusinessScopeAuthorityV1,
        state_key_ref: str,
        plan: ExecutionPlanV2,
        recall_state: RecallTurnStateV1 | None = None,
    ) -> V2AttemptResult: ...


class PlannerAgentFactoryV1(Protocol):
    def build_planner_agent(self) -> CrewAIAgentAdapterV1: ...


class PlannerRuntimeV1(Protocol):
    settings: Settings

    @property
    def planner_agent_factory(self) -> PlannerAgentFactoryV1: ...

    @property
    def candidate_from_plan_builder_v2(self) -> PlanCandidateBuilderV1: ...


async def build_candidate_via_planner(
    runtime: PlannerRuntimeV1,
    *,
    request: KernelReplyRequestV1,
    domain_context: DomainContextV1,
    policy: PolicyManifestV2,
    model_family: ModelFamily,
    intent_gate: IntentGateResult,
    history: list[ConversationMessage],
    action_history: Sequence[RecentExecutedActionSummaryViewV1],
    prompt_programs: list[PromptProgram],
    llm_executions: list[dict[str, JsonValue]],
    scope_authority: BusinessScopeAuthorityV1,
    state_key_ref: str,
    alignment_verdict: ReplyAlignmentVerdict | None = None,
    alignment_attempt: int = 0,
    recall_state: RecallTurnStateV1,
    llm_journal: TurnLlmInvocationJournalV1 | None = None,
) -> V2AttemptResult:
    del domain_context, model_family
    planner_model_family = model_family_from_settings(
        runtime.settings,
        stage="planner_intent",
    )
    input_source = PlannerRuntimeInputSourceV1(
        request=request,
        policy=policy,
        scope_authority=scope_authority,
        intent_gate=intent_gate,
        history=history,
        action_history=action_history,
        recall_state=recall_state,
        now=datetime.now(ZoneInfo("Asia/Shanghai")),
        retry_overlay=planner_alignment_replan_overlay(
            alignment_verdict,
            alignment_attempt,
        ),
    )
    planner_input = build_runtime_planner_prompt_input_v1(input_source)

    async def run_round(
        round_input: PlannerPromptInputV1,
    ) -> tuple[PlannerFrameResult, CrewAIAgentAdapterV1 | None, PromptProgram]:
        with trace_span("planner.assemble_prompt"):
            target_program = select_stage_input_prompt_program(
                round_input,
                planner_model_family,
            )
        prompt_programs.append(target_program)
        try:
            active_agent = None
            if target_program.scene_key != "wecom_direct.v1":
                with trace_span("planner.build_agent"):
                    active_agent = runtime.planner_agent_factory.build_planner_agent()
            frame_result, planner_executions = await run_planner_kickoff_with_retry(
                active_agent,
                target_program,
                planner_input=round_input,
                timeout_seconds=runtime.settings.llm_timeout_seconds,
                retry_attempts=runtime.settings.planner_transient_retry_attempts,
                base_delay_seconds=0.0,
                journal=llm_journal,
                settings=runtime.settings,
            )
            llm_executions.extend(planner_executions)
        except ProviderInvocationError as exc:
            if exc.code == "provider_timeout":
                raise AgentRuntimeError("CrewAI planner timed out") from None
            raise AgentRuntimeError("planner_provider_failure") from None
        except RuntimeError:
            raise AgentRuntimeError("planner_internal_failure") from None
        return frame_result, active_agent, target_program

    frame_result, active_planner_agent, active_planner_program = await run_round(
        planner_input
    )

    with trace_span("planner.coerce_compile"):
        plan, error_summary = coerce_planner_plan_with_error(
            frame_result,
            policy,
            scope_authority=scope_authority,
        )
    if plan is None and planner_schema_repair_allowed(planner_input.retry_overlay):
        trace_event("planner.invalid_plan_spec", error=error_summary)
        repair_input = build_runtime_planner_prompt_input_v1(
            replace(
                input_source,
                retry_overlay=planner_schema_repair_overlay(error_summary),
            )
        )
        frame_result, active_planner_agent, active_planner_program = await run_round(
            repair_input
        )
        with trace_span("planner.retry_coerce_compile"):
            plan, error_summary = coerce_planner_plan_with_error(
                frame_result,
                policy,
                scope_authority=scope_authority,
            )
    if plan is None:
        trace_event("planner.invalid_plan_spec", error=error_summary, retry=True)
        record_llm_failure(
            active_planner_agent,
            active_planner_program.profile.stage,
            "provider_output_contract",
        )
        raise AgentRuntimeError(
            f"CrewAI planner returned an invalid PlanSpec contract: {error_summary}"
        )
    trace_event(
        "state.plan_compiled",
        response_mode=plan.response_mode,
        selected_manifest_refs=tuple(
            ref.manifest_id for ref in plan.selected_manifest_refs
        ),
        adapter_resolve_count=len(plan.adapter_resolves),
    )
    candidate_builder = runtime.candidate_from_plan_builder_v2
    return await candidate_builder(
        request=request,
        policy=policy,
        scope_authority=scope_authority,
        state_key_ref=state_key_ref,
        plan=plan,
        recall_state=recall_state,
    )
