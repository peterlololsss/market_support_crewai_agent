from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, final

from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.context.stage_inputs import (
    SanitizedAlignmentVerifierInputV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIProviderResultV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.io import (
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.integrations.crewai.result_coercion import (
    coerce_alignment_verdict,
)
from market_support_crewai_agent.runtime.planning import planner_llm
from market_support_crewai_agent.runtime.prompts import direct_provider_client
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
)
from market_support_crewai_agent.runtime.prompts.router import (
    select_stage_input_prompt_program,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
    ReplyAlignmentVerifier,
)
from market_support_crewai_agent.settings_model import Settings

_EXECUTION_ROW_ADAPTER = TypeAdapter(dict[str, JsonValue])
_DIRECT_PROVIDER = direct_provider_client
_FAILURE_RECORDER = planner_llm


class _AlignmentAgentFactoryV1(Protocol):
    def build_alignment_verifier_agent(self) -> CrewAIAgentAdapterV1: ...


class _AlignmentVerifierRuntimeV1(Protocol):
    @property
    def settings(self) -> Settings: ...

    @property
    def alignment_verifier(self) -> ReplyAlignmentVerifier | None: ...

    @property
    def alignment_agent_factory(self) -> _AlignmentAgentFactoryV1: ...


@dataclass(frozen=True, slots=True)
class InternalAlignmentVerifierSourceV1:
    model_family: ModelFamily
    prompt_programs: list[PromptProgram]
    llm_executions: list[dict[str, JsonValue]]


async def verify_reply_alignment(
    runtime: _AlignmentVerifierRuntimeV1,
    input_value: SanitizedAlignmentVerifierInputV1,
    internal_source: InternalAlignmentVerifierSourceV1,
) -> ReplyAlignmentVerdict:
    verifier: ReplyAlignmentVerifier
    if runtime.alignment_verifier is not None:
        verifier = runtime.alignment_verifier
    else:
        verifier = _CrewAIReplyAlignmentVerifier(runtime, internal_source)
    verdict = await verifier.verify(input_value)
    return ReplyAlignmentVerdict.model_validate(verdict)


@final
class _CrewAIReplyAlignmentVerifier:
    def __init__(
        self,
        runtime: _AlignmentVerifierRuntimeV1,
        source: InternalAlignmentVerifierSourceV1,
    ) -> None:
        self._runtime = runtime
        self._source = source

    async def verify(
        self,
        input_value: SanitizedAlignmentVerifierInputV1,
    ) -> ReplyAlignmentVerdict:
        program = select_stage_input_prompt_program(
            input_value,
            self._source.model_family,
        )
        self._source.prompt_programs.append(program)
        agent: CrewAIAgentAdapterV1 | None = None
        try:
            if program.scene_key == "wecom_direct.v1":
                (
                    direct_result,
                    execution,
                ) = await _DIRECT_PROVIDER.run_direct_prompt_program(
                    prompt_program=program,
                    stage_input=input_value,
                    settings=self._runtime.settings,
                )
                result = CrewAIProviderResultV1(
                    raw=direct_result.raw,
                    pydantic=direct_result.pydantic,
                    agent_role="direct_provider",
                    usage_metrics=None,
                )
            else:
                agent = self._runtime.alignment_agent_factory.build_alignment_verifier_agent()
                result, execution = await run_crewai_kickoff(
                    agent,
                    program,
                    timeout_seconds=self._runtime.settings.llm_timeout_seconds,
                )
            self._source.llm_executions.append(
                _EXECUTION_ROW_ADAPTER.validate_python(execution)
            )
        except ProviderInvocationError as exc:
            if agent is not None:
                _FAILURE_RECORDER.record_llm_failure(
                    agent,
                    "alignment_verifier",
                    exc.code,
                )
            raise AgentRuntimeError("alignment_verifier_provider_failure") from None
        except Exception:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
            if agent is not None:
                _FAILURE_RECORDER.record_llm_failure(
                    agent,
                    "alignment_verifier",
                    "provider_internal_error",
                )
            raise AgentRuntimeError("alignment_verifier_internal_failure") from None
        verdict = coerce_alignment_verdict(result)
        if verdict is None:
            raise AgentRuntimeError(
                "CrewAI alignment verifier returned an invalid ReplyAlignmentVerdict contract"
            )
        return verdict


__all__ = [
    "InternalAlignmentVerifierSourceV1",
    "verify_reply_alignment",
]
