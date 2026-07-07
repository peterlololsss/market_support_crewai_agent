from __future__ import annotations

from pydantic import JsonValue

from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
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
from market_support_crewai_agent.runtime.orchestration.planner import (
    can_fallback_to_default_planner,
    coerce_planner_plan_with_error,
    is_empty_planner_result,
    planner_retry_program,
    record_llm_failure,
    run_planner_kickoff_with_retry,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.state.runtime_trace import (
    trace_event,
    trace_span,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError, AttemptResult
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import ReplyRequest


async def build_candidate_via_planner(
    runtime,
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
    alignment_verdict: ReplyAlignmentVerdict | None = None,
    alignment_attempt: int = 0,
) -> AttemptResult:
    with trace_span("planner.project_context"):
        planner_context = runtime._project_context(
            stage="planner_intent",
            request=request,
            domain_context=domain_context,
            policy=policy,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
        )
    planner_prompt_context = PromptAssemblyContext(
        stage="planner_intent",
        model_family=model_family_from_settings(
            runtime.settings,
            stage="planner_intent",
        ),
        request=request,
        model_visible_context=planner_context,
        domain_context=domain_context,
        policy=policy,
        intent_gate=intent_gate,
        history=history,
        action_history=action_history,
        alignment_verdict=alignment_verdict,
        alignment_attempt=alignment_attempt,
    )
    with trace_span("planner.assemble_prompt"):
        planner_program = select_prompt_program(planner_prompt_context)
    prompt_programs.append(planner_program)
    active_planner_program = planner_program
    try:
        with trace_span("planner.build_agent"):
            planner_agent = runtime._build_planner_agent()
        active_planner_agent = planner_agent
        frame_result, planner_executions = await run_planner_kickoff_with_retry(
            planner_agent,
            planner_program,
            timeout_seconds=runtime.settings.llm_timeout_seconds,
            retry_attempts=runtime.settings.planner_transient_retry_attempts,
            base_delay_seconds=runtime.settings.planner_transient_retry_base_seconds,
        )
        llm_executions.extend(planner_executions)
    except TimeoutError as exc:
        raise AgentRuntimeError("CrewAI planner timed out") from exc
    except Exception as exc:
        raise AgentRuntimeError("CrewAI planner failed") from exc
    if is_empty_planner_result(frame_result) and can_fallback_to_default_planner(
        runtime.settings
    ):
        active_planner_program = runtime._planner_fallback_program(
            planner_prompt_context
        )
        prompt_programs.append(active_planner_program)
        trace_event("planner.fallback_to_default_llm", reason="empty_output")
        try:
            with trace_span("planner.build_fallback_agent"):
                active_planner_agent = runtime._build_planner_fallback_agent()
            frame_result, planner_executions = await run_planner_kickoff_with_retry(
                active_planner_agent,
                active_planner_program,
                timeout_seconds=runtime.settings.llm_timeout_seconds,
                retry_attempts=runtime.settings.planner_transient_retry_attempts,
                base_delay_seconds=runtime.settings.planner_transient_retry_base_seconds,
            )
            llm_executions.extend(planner_executions)
        except TimeoutError as exc:
            raise AgentRuntimeError("CrewAI planner fallback timed out") from exc
        except Exception as exc:
            raise AgentRuntimeError("CrewAI planner fallback failed") from exc

    with trace_span("planner.coerce_compile"):
        plan, error_summary = coerce_planner_plan_with_error(
            frame_result,
            request,
            policy,
            domain_context=domain_context,
            history=history,
        )
    if plan is None:
        trace_event("planner.invalid_plan_spec", error=error_summary)
        retry_program = planner_retry_program(active_planner_program, error_summary)
        prompt_programs.append(retry_program)
        try:
            frame_result, planner_executions = await run_planner_kickoff_with_retry(
                active_planner_agent,
                retry_program,
                timeout_seconds=runtime.settings.llm_timeout_seconds,
                retry_attempts=runtime.settings.planner_transient_retry_attempts,
                base_delay_seconds=runtime.settings.planner_transient_retry_base_seconds,
            )
            llm_executions.extend(planner_executions)
        except TimeoutError as exc:
            raise AgentRuntimeError("CrewAI planner retry timed out") from exc
        except Exception as exc:
            raise AgentRuntimeError("CrewAI planner retry failed") from exc
        if (
            active_planner_agent is planner_agent
            and is_empty_planner_result(frame_result)
            and can_fallback_to_default_planner(runtime.settings)
        ):
            fallback_program = runtime._planner_fallback_program(
                planner_prompt_context,
                error_summary=error_summary,
            )
            prompt_programs.append(fallback_program)
            trace_event("planner.fallback_to_default_llm", reason="empty_output")
            try:
                with trace_span("planner.build_fallback_agent"):
                    active_planner_agent = runtime._build_planner_fallback_agent()
                frame_result, planner_executions = await run_planner_kickoff_with_retry(
                    active_planner_agent,
                    fallback_program,
                    timeout_seconds=runtime.settings.llm_timeout_seconds,
                    retry_attempts=runtime.settings.planner_transient_retry_attempts,
                    base_delay_seconds=runtime.settings.planner_transient_retry_base_seconds,
                )
                llm_executions.extend(planner_executions)
            except TimeoutError as exc:
                raise AgentRuntimeError("CrewAI planner fallback timed out") from exc
            except Exception as exc:
                raise AgentRuntimeError("CrewAI planner fallback failed") from exc
        with trace_span("planner.retry_coerce_compile"):
            plan, error_summary = coerce_planner_plan_with_error(
                frame_result,
                request,
                policy,
                domain_context=domain_context,
                history=history,
            )
    if plan is None:
        trace_event("planner.invalid_plan_spec", error=error_summary, retry=True)
        record_llm_failure(
            active_planner_agent,
            active_planner_program.profile.stage,
            f"invalid PlanSpec contract after retry: {error_summary}",
        )
        raise AgentRuntimeError(
            f"CrewAI planner returned an invalid PlanSpec contract: {error_summary}"
        )
    trace_event(
        "state.plan_compiled",
        response_mode=plan.response_mode,
        capabilities=plan.capabilities,
        adapter_resolve_count=len(plan.adapter_resolves),
    )
    return await runtime._build_candidate_from_plan(
        request=request,
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
