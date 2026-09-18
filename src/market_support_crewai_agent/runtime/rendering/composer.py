from __future__ import annotations

from typing import Literal, Protocol, assert_never, final

from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
    SmalltalkComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIProviderResultV1,
    CrewAITransportInvariantError,
)
from market_support_crewai_agent.runtime.integrations.crewai.io import (
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.integrations.crewai.result_coercion import (
    coerce_composer_output,
)
from market_support_crewai_agent.runtime.integrations.crewai.retry import (
    RetryPolicy,
    run_with_retry,
)
from market_support_crewai_agent.runtime.observability.runtime_trace import (
    trace_event,
)
from market_support_crewai_agent.runtime.planning import planner_llm
from market_support_crewai_agent.runtime.prompts import (
    direct_provider_client,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    InvocationJournalError,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    PromptGovernanceError,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
    provider_failure_code_from_exception,
)
from market_support_crewai_agent.runtime.prompts.provider_response_text import (
    DirectProviderTransportError,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    ProviderTargetError,
)
from market_support_crewai_agent.runtime.prompts.router import (
    model_family_from_settings,
    select_stage_input_prompt_program,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerPromptInputV1,
    V2ComposerOutputRejected,
    V2ComposerUnavailable,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.settings_model import Settings

_EXECUTION_ROW_ADAPTER = TypeAdapter(dict[str, JsonValue])
_DIRECT_PROVIDER = direct_provider_client
_FAILURE_RECORDER = planner_llm


class _ComposerAgentFactoryV1(Protocol):
    def build_composer_agent(
        self,
        stage: Literal["knowledge_composer", "smalltalk_composer"],
    ) -> CrewAIAgentAdapterV1: ...


class _ComposerRuntimeV1(Protocol):
    @property
    def settings(self) -> Settings: ...

    @property
    def composer_agent_factory(self) -> _ComposerAgentFactoryV1: ...


@final
class CrewAIV2Composer:
    def __init__(self, runtime: _ComposerRuntimeV1) -> None:
        self._runtime = runtime

    async def compose(
        self,
        input_value: ComposerPromptInputV1,
    ) -> ComposerReplyOutput:
        match input_value:
            case KnowledgeComposerPromptInputV1(unit_groundings=groundings):
                stage = "knowledge_composer"
                if not any(grounding.allowed_evidence_ids for grounding in groundings):
                    raise V2ComposerUnavailable("v2_composer_evidence_required")
            case SmalltalkComposerPromptInputV1():
                stage = "smalltalk_composer"
            case _:
                assert_never(input_value)
        program = select_stage_input_prompt_program(
            input_value,
            model_family_from_settings(self._runtime.settings),
        )
        agent = (
            None
            if program.scene_key == "wecom_direct.v1"
            else self._runtime.composer_agent_factory.build_composer_agent(stage)
        )
        try:
            output, _executions = await run_composer_kickoff_with_retry(
                agent,
                program,
                timeout_seconds=self._runtime.settings.llm_timeout_seconds,
                retry_attempts=self._runtime.settings.planner_transient_retry_attempts,
                base_delay_seconds=0.0,
                composer_input=input_value,
                settings=self._runtime.settings,
            )
        except ProviderInvocationError:
            raise AgentRuntimeError("composer_provider_failure") from None
        except (
            CrewAITransportInvariantError,
            DirectProviderTransportError,
            InvocationJournalError,
            PromptGovernanceError,
            ProviderTargetError,
        ):
            raise AgentRuntimeError("composer_internal_failure") from None
        if output is None:
            _FAILURE_RECORDER.record_llm_failure(
                agent,
                stage,
                "provider_output_contract",
            )
            raise V2ComposerOutputRejected("v2_composer_output_contract_invalid")
        return output


async def run_composer_kickoff_with_retry(
    composer_agent: CrewAIAgentAdapterV1 | None,
    composer_program: PromptProgram,
    *,
    timeout_seconds: float | None,
    retry_attempts: int,
    base_delay_seconds: float,
    composer_input: KnowledgeComposerPromptInputV1
    | SmalltalkComposerPromptInputV1
    | None = None,
    settings: Settings | None = None,
) -> tuple[ComposerReplyOutput | None, list[dict[str, JsonValue]]]:
    executions: list[dict[str, JsonValue]] = []

    async def call() -> CrewAIProviderResultV1:
        if composer_program.scene_key == "wecom_direct.v1":
            if composer_input is None or settings is None:
                raise ContextViewInvariantError("direct_composer_input_required")
            direct_result, execution = await _DIRECT_PROVIDER.run_direct_prompt_program(
                prompt_program=composer_program,
                stage_input=composer_input,
                settings=settings,
            )
            result = CrewAIProviderResultV1(
                raw=direct_result.raw,
                pydantic=direct_result.pydantic,
                agent_role="direct_provider",
                usage_metrics=None,
            )
        else:
            result, execution = await run_crewai_kickoff(
                composer_agent,
                composer_program,
                timeout_seconds=timeout_seconds,
            )
        executions.append(_EXECUTION_ROW_ADAPTER.validate_python(execution))
        return result

    try:
        result = await run_with_retry(
            call,
            policy=RetryPolicy(
                retry_attempts=retry_attempts,
                base_delay_seconds=base_delay_seconds,
            ),
            should_retry_exception=_composer_exception_retry_reason,
            on_retry=_trace_composer_retry,
        )
    except Exception as exc:
        _FAILURE_RECORDER.record_llm_failure(
            composer_agent,
            composer_program.profile.stage,
            provider_failure_code_from_exception(exc),
        )
        raise
    return coerce_composer_output(result), executions


def _composer_exception_retry_reason(exc: Exception) -> str | None:
    return provider_failure_code_from_exception(exc)


def _trace_composer_retry(attempt: int, delay_seconds: float, reason: str) -> None:
    trace_event(
        "composer.transient_retry",
        attempt=attempt,
        delay_ms=round(delay_seconds * 1000, 3),
        reason=reason,
    )
