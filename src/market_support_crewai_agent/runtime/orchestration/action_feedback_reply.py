from __future__ import annotations

from typing import Protocol

from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.models import stable_json
from market_support_crewai_agent.runtime.context.projection import (
    ContextProjectionManager,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.llm.prompting.context import (
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.llm.prompting.router import (
    model_family_from_settings,
    select_prompt_program,
)
from market_support_crewai_agent.runtime.orchestration.composer import (
    run_composer_kickoff_with_retry,
)
from market_support_crewai_agent.runtime.orchestration.crewai_io import (
    coerce_composer_output,
)
from market_support_crewai_agent.runtime.orchestration.response_ids import (
    ensure_response_ids,
)
from market_support_crewai_agent.runtime.state.runtime_trace import trace_span
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    PrimaryReply,
    ReplyRequest,
)
from market_support_crewai_agent.settings import Settings

_TERMINAL_OUTCOMES = frozenset({"complete", "partial", "failed"})
_COUNT_FIELDS = (
    "target_count",
    "attempted_count",
    "accepted_count",
    "failed_count",
    "unattempted_count",
)


class ActionFeedbackRuntime(Protocol):
    settings: Settings
    conversation_store: ConversationStore

    def _build_agent(self, stage: str) -> object: ...


async def compose_action_feedback_reply(
    runtime: ActionFeedbackRuntime,
    feedback: ActionFeedbackRequest,
) -> PrimaryReply | None:
    summary = _terminal_execution_summary(feedback)
    if summary is None:
        return None

    request = _feedback_reply_request(feedback, summary)
    policy = compile_policy(request, outbound_messaging_enabled=False)
    history = runtime.conversation_store.get_recent(feedback.conversation_key)
    with trace_span("action_feedback.project_context"):
        projected = ContextProjectionManager.from_settings(
            runtime.settings
        ).project_for_stage(
            stage="action_feedback_composer",
            request=request,
            policy=policy,
            history=history,
        )
    with trace_span("action_feedback.assemble_prompt"):
        program = select_prompt_program(
            PromptAssemblyContext(
                stage="action_feedback_composer",
                model_family=model_family_from_settings(runtime.settings),
                request=request,
                policy=policy,
                model_visible_context=projected,
                history=history,
            )
        )
    try:
        with trace_span("action_feedback.build_agent"):
            agent = runtime._build_agent("action_feedback_composer")
        result, _executions = await run_composer_kickoff_with_retry(
            agent,
            program,
            timeout_seconds=runtime.settings.llm_timeout_seconds,
            retry_attempts=runtime.settings.planner_transient_retry_attempts,
            base_delay_seconds=runtime.settings.planner_transient_retry_base_seconds,
        )
    except TimeoutError as exc:
        raise AgentRuntimeError("CrewAI action feedback composer timed out") from exc
    except Exception as exc:
        raise AgentRuntimeError("CrewAI action feedback composer failed") from exc

    output = coerce_composer_output(result)
    if (
        output is None
        or output.response_mode != "answer"
        or output.reply.kind != "answer"
        or not output.reply.text.strip()
        or output.reply.mentions
        or output.claims
        or output.evidence_ids
        or output.missing_inputs
    ):
        raise AgentRuntimeError(
            "CrewAI action feedback composer returned an invalid reply contract"
        )
    return ensure_response_ids(output.to_reply_response()).reply


def _terminal_execution_summary(
    feedback: ActionFeedbackRequest,
) -> dict[str, JsonValue] | None:
    executions = [
        execution
        for execution in feedback.executions
        if execution.action_type == "execute_prepared_outbound_message"
    ]
    if len(executions) != 1:
        return None
    execution = executions[0]
    result = execution.adapter_result
    outcome = result.get("outcome")
    if outcome not in _TERMINAL_OUTCOMES:
        return None

    summary: dict[str, JsonValue] = {
        "event": "outbound_execution_feedback",
        "execution_status": execution.status,
        "outcome": outcome,
        "accepted_not_delivered": True,
    }
    for field in _COUNT_FIELDS:
        value = result.get(field)
        if type(value) is int and value >= 0:
            summary[field] = value
    replayed = result.get("replayed")
    if type(replayed) is bool:
        summary["replayed"] = replayed
    reason = result.get("reason")
    if isinstance(reason, str) and reason.strip():
        summary["reason"] = reason.strip()[:128]
    return summary


def _feedback_reply_request(
    feedback: ActionFeedbackRequest,
    summary: dict[str, JsonValue],
) -> ReplyRequest:
    return ReplyRequest(
        conversation_key=feedback.conversation_key,
        group_id=feedback.group_id,
        sender_id=feedback.sender_id,
        context_id=feedback.context_id,
        message=stable_json(summary),
        is_group=False,
        group_name="direct_message",
        dist_channel_name="direct_message",
        sender_nickname=feedback.sender_id,
        available_artifacts=[],
        channel_type="non_bank",
        allowed_read_capabilities=[],
    )
