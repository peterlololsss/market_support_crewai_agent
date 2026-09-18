from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAICompletionCaptureTargetV1,
    CrewAICompletionInvokeV1,
    CrewAICompletionValueV1,
    CrewAIKickoffOutputV1,
    CrewAILlmAdapterV1,
    GeminiGenerateContentAdapterV1,
    GeminiModelsClientV1,
    GeminiSyncClientV1,
)

GeminiGenerateFn = Callable[[GeminiGenerateContentAdapterV1], JsonValue]
type CompletionByFormatFn = Callable[
    [type[BaseModel]],
    CrewAICompletionValueV1,
]
type PromptObserverFn = Callable[[str], None]


def make_llm_adapter(
    *,
    provider: str = "openai",
    model: str = "model",
    base_url: str | None = None,
    api_key: str | None = None,
    client_params: dict[str, JsonValue] | None = None,
    max_tokens: int | None = None,
    max_output_tokens: int | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stop_sequences: tuple[str, ...] | None = None,
    thinking_config: JsonValue | None = None,
    max_retries: int | None = 0,
    gemini_retry_attempts: int | None = None,
    completion: CrewAICompletionInvokeV1 | None = None,
    gemini_generate: GeminiGenerateFn | None = None,
) -> CrewAILlmAdapterV1:
    holder: dict[str, CrewAICompletionInvokeV1 | None] = {"value": completion}

    def current() -> CrewAICompletionInvokeV1 | None:
        return holder["value"]

    def replace(value: CrewAICompletionInvokeV1 | None) -> None:
        holder["value"] = value

    def sync_client() -> GeminiSyncClientV1:
        if gemini_generate is None:
            raise AssertionError("gemini_generate_required")
        return GeminiSyncClientV1(
            models=GeminiModelsClientV1(generate_content_fn=gemini_generate)
        )

    return CrewAILlmAdapterV1(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        client_params=client_params or {},
        max_tokens=max_tokens,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_sequences=stop_sequences,
        thinking_config=thinking_config,
        max_retries=max_retries,
        gemini_retry_attempts=gemini_retry_attempts,
        completion_capture=CrewAICompletionCaptureTargetV1(
            current=current,
            replace=replace,
        ),
        gemini_sync_client_factory=sync_client if gemini_generate is not None else None,
    )


def make_agent_adapter(
    *,
    role: str = "planner",
    llm: CrewAILlmAdapterV1 | None = None,
    result: CrewAIKickoffOutputV1 | None = None,
    on_prompt: Callable[[str, type[BaseModel]], CrewAIKickoffOutputV1] | None = None,
) -> CrewAIAgentAdapterV1:
    async def kickoff(
        prompt: str, response_format: type[BaseModel]
    ) -> CrewAIKickoffOutputV1:
        if on_prompt is not None:
            return on_prompt(prompt, response_format)
        if result is not None:
            return result
        return CrewAIKickoffOutputV1(raw="{}", pydantic=None)

    return CrewAIAgentAdapterV1(
        role=role,
        llm=llm or make_llm_adapter(),
        kickoff_async=kickoff,
    )


def invoke_captured_completion(
    llm: CrewAILlmAdapterV1,
    prompt: str,
    response_format: type[BaseModel],
) -> CrewAIKickoffOutputV1:
    completion_value = invoke_completion(
        llm,
        params={
            "model": llm.model,
            "messages": [{"role": "user", "content": prompt}],
        },
        response_model=response_format,
    )
    if isinstance(completion_value, BaseModel):
        return CrewAIKickoffOutputV1(
            raw=completion_value.model_dump_json(),
            pydantic=completion_value,
        )
    if (
        isinstance(completion_value, (str, int, float, bool))
        or completion_value is None
    ):
        return CrewAIKickoffOutputV1(raw=completion_value, pydantic=None)
    return CrewAIKickoffOutputV1(
        raw=json.dumps(completion_value, ensure_ascii=False, sort_keys=True),
        pydantic=None,
    )


def make_completion_agent_adapter(
    completion: CompletionByFormatFn,
    *,
    role: str = "planner",
    model: str = "model",
    provider: str = "openai",
    base_url: str | None = None,
    on_prompt: PromptObserverFn | None = None,
) -> CrewAIAgentAdapterV1:
    def handle_completion(
        *,
        params: dict[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> CrewAICompletionValueV1:
        del params, available_functions, from_task, from_agent
        if response_model is None:
            raise AssertionError("response_model_missing")
        return completion(response_model)

    llm = make_llm_adapter(
        provider=provider,
        model=model,
        base_url=base_url,
        completion=handle_completion,
    )

    def kickoff(prompt: str, response_format: type[BaseModel]) -> CrewAIKickoffOutputV1:
        if on_prompt is not None:
            on_prompt(prompt)
        return invoke_captured_completion(llm, prompt, response_format)

    return make_agent_adapter(role=role, llm=llm, on_prompt=kickoff)


def invoke_completion(
    llm: CrewAILlmAdapterV1,
    **kwargs: JsonValue | type[BaseModel] | None,
) -> JsonValue | BaseModel:
    if llm.completion_capture is None:
        raise AssertionError("completion_capture_missing")
    current = llm.completion_capture.current()
    if current is None:
        raise AssertionError("completion_missing")
    return current(**kwargs)
