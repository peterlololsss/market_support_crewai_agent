from __future__ import annotations

from typing import TYPE_CHECKING

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAICompletionCaptureTargetV1,
    CrewAICompletionInvokeV1,
)

if TYPE_CHECKING:
    from crewai.llms.base_llm import BaseLLM as CrewAISdkLLM


def completion_capture_target(llm: CrewAISdkLLM) -> CrewAICompletionCaptureTargetV1:
    state: dict[str, CrewAICompletionInvokeV1 | None] = {
        "current": completion_from_sdk(llm)
    }

    def current() -> CrewAICompletionInvokeV1 | None:
        return state["current"]

    def replace(value: CrewAICompletionInvokeV1 | None) -> None:
        state["current"] = value
        if value is None:
            if "_handle_completion" in llm.__dict__:
                delattr(llm, "_handle_completion")
            return
        llm._handle_completion = value

    return CrewAICompletionCaptureTargetV1(current=current, replace=replace)


def completion_from_sdk(llm: CrewAISdkLLM) -> CrewAICompletionInvokeV1 | None:
    from crewai.llms.providers.openai.completion import OpenAICompletion

    class _CaptureReadyOpenAICompletion(OpenAICompletion):
        def crewai_completion_capture(self) -> CrewAICompletionInvokeV1:
            return self._handle_completion

    if isinstance(llm, OpenAICompletion):
        llm.__class__ = _CaptureReadyOpenAICompletion
    if isinstance(llm, _CaptureReadyOpenAICompletion):
        return llm.crewai_completion_capture()
    return None
