from __future__ import annotations

import asyncio
from time import perf_counter

from pydantic import BaseModel, JsonValue, ValidationError

from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.direct_provider_io_capture import (
    DirectProviderIoCaptureError,
    DirectProviderIoCaptureScopeV1,
)
from market_support_crewai_agent.runtime.prompts.direct_provider_journal import (
    close_direct_journal_error,
    close_direct_journal_success,
    direct_target_from_settings,
    elapsed_ms,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
    resolve_turn_llm_invocation_journal,
)
from market_support_crewai_agent.runtime.prompts.profiles import SceneKeyV1
from market_support_crewai_agent.runtime.prompts.program_models import (
    DirectProviderSynthesisV1,
    ProviderJsonSchemaFormatV1,
    ProviderTargetConfigV1,
    direct_synthesis_stage_kind,
    resolve_active_prompt_program_v2,
    synthesize_direct_provider_messages,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
    ProviderInvocationError,
    normalize_provider_failure_code,
    provider_http_failure_code,
)
from market_support_crewai_agent.runtime.prompts.provider_response_text import (
    DirectPromptProgramResultV1,
    DirectProviderTransportError,
    build_direct_prompt_program_result,
    gemini_response_text,
    openai_response_text,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    INTERNAL_CREWAI_SDK_ENDPOINT,
)
from market_support_crewai_agent.runtime.prompts.provider_transport import (
    build_provider_transport_envelope,
    capture_provider_text_output,
)
from market_support_crewai_agent.settings_model import Settings


async def run_direct_provider_text(
    *,
    synthesis: DirectProviderSynthesisV1,
    target: ProviderTargetConfigV1,
    api_key: str,
    response_model: type[BaseModel],
    scene_key: SceneKeyV1 = "scene_neutral.v1",
    journal: TurnLlmInvocationJournalV1 | None = None,
    purpose: str = "direct_provider_text",
) -> str:
    import httpx

    stage_kind = direct_synthesis_stage_kind(synthesis)
    if target.normalized_endpoint == INTERNAL_CREWAI_SDK_ENDPOINT:
        raise DirectProviderTransportError("provider_endpoint_scheme_invalid")
    program, _execution_spec = resolve_active_prompt_program_v2(
        stage=stage_kind,
        scene_key=scene_key,
    )
    if program.program_id != synthesis.user_payload.prompt_program_id:
        raise DirectProviderTransportError("prompt_program_v2_dispatch_mismatch")
    envelope = build_provider_transport_envelope(synthesis, target)
    journal = resolve_turn_llm_invocation_journal(journal)
    reserved = (
        journal.reserve(
            stage_kind=stage_kind,
            program_id=synthesis.user_payload.prompt_program_id,
            target_slot=target.target_slot,
            purpose=purpose,
        )
        if journal is not None and stage_kind != "llm_health_probe"
        else None
    )
    started_at = perf_counter()
    headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
    timeout = httpx.Timeout(target.timeout_seconds)
    try:
        try:
            async with (
                DirectProviderIoCaptureScopeV1(),
                httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client,
            ):
                match envelope.variant:
                    case "openai_chat_completions":
                        response = await client.post(
                            target.normalized_endpoint + "/chat/completions",
                            headers=headers,
                            json=envelope.body,
                        )
                        _ = response.raise_for_status()
                        text = openai_response_text(response.content)
                    case "gemini_generate_content":
                        response = await client.post(
                            target.normalized_endpoint
                            + f"/models/{target.model_name}:generateContent",
                            headers=headers,
                            json=envelope.body,
                        )
                        _ = response.raise_for_status()
                        text = gemini_response_text(response.content)
        except DirectProviderIoCaptureError:
            raise ProviderInvocationError("direct_provider_stdio_violation") from None
        except httpx.HTTPStatusError as exc:
            raise ProviderInvocationError(
                provider_http_failure_code(exc.response.status_code)
            ) from None
        except httpx.RequestError as exc:
            code: ProviderFailureCodeV1 = (
                "provider_timeout"
                if isinstance(exc, httpx.TimeoutException)
                else "provider_transport_unavailable"
            )
            raise ProviderInvocationError(code) from None
        capture = capture_provider_text_output(text, response_model=response_model)
        if capture.status != "available_text" or capture.text is None:
            raise ProviderInvocationError(
                normalize_provider_failure_code(
                    capture.error_code or "provider_output_contract"
                )
            )
    except ProviderInvocationError as exc:
        if reserved is not None and journal is not None:
            close_direct_journal_error(
                journal,
                reserved,
                program=program,
                synthesis=synthesis,
                target=target,
                envelope=envelope,
                response_model=response_model,
                surfaced_code=exc.code,
                latency_ms=elapsed_ms(started_at),
            )
        raise
    except asyncio.CancelledError:
        if reserved is not None and journal is not None:
            close_direct_journal_error(
                journal,
                reserved,
                program=program,
                synthesis=synthesis,
                target=target,
                envelope=envelope,
                response_model=response_model,
                surfaced_code="provider_transport_unavailable",
                latency_ms=elapsed_ms(started_at),
            )
        raise
    except ValidationError:
        if reserved is not None and journal is not None:
            close_direct_journal_error(
                journal,
                reserved,
                program=program,
                synthesis=synthesis,
                target=target,
                envelope=envelope,
                response_model=response_model,
                surfaced_code="provider_internal_error",
                latency_ms=elapsed_ms(started_at),
            )
        raise ProviderInvocationError("provider_internal_error") from None
    if reserved is not None and journal is not None:
        close_direct_journal_success(
            journal,
            reserved,
            program=program,
            synthesis=synthesis,
            target=target,
            envelope=envelope,
            response_model=response_model,
            out1=capture.out1(),
            latency_ms=elapsed_ms(started_at),
            output_bytes=capture.byte_count,
        )
    return capture.text


async def run_direct_prompt_program(
    *,
    prompt_program: PromptProgram,
    stage_input: BaseModel,
    settings: Settings,
    journal: TurnLlmInvocationJournalV1 | None = None,
) -> tuple[DirectPromptProgramResultV1, dict[str, JsonValue]]:
    if prompt_program.scene_key != "wecom_direct.v1":
        raise DirectProviderTransportError("direct_prompt_scene_required")
    if not prompt_program.static_prompt_text:
        raise DirectProviderTransportError("direct_prompt_static_instructions_required")
    target, api_key = direct_target_from_settings(
        settings,
        stage=prompt_program.profile.stage,
        temperature=(
            prompt_program.profile.temperature
            if prompt_program.profile.temperature is not None
            else settings.llm_temperature
        ),
        max_tokens=(
            prompt_program.profile.max_tokens
            if prompt_program.profile.max_tokens is not None
            else settings.llm_max_tokens
        ),
    )
    if api_key is None:
        raise ProviderInvocationError("provider_auth_failed")
    response_model = prompt_program.profile.response_model
    active_program, execution_spec = resolve_active_prompt_program_v2(
        stage=prompt_program.profile.stage,
        scene_key=prompt_program.scene_key,
    )
    synthesis = synthesize_direct_provider_messages(
        program=active_program,
        execution_spec=execution_spec,
        prompt_text=prompt_program.static_prompt_text,
        stage_input=stage_input,
        output_schema=ProviderJsonSchemaFormatV1(
            name=response_model.__name__,
            json_schema=response_model.model_json_schema(),
        ),
    )
    started_at = perf_counter()
    text = await run_direct_provider_text(
        synthesis=synthesis,
        target=target,
        api_key=api_key,
        response_model=response_model,
        scene_key=prompt_program.scene_key,
        journal=journal,
        purpose="direct_prompt_program",
    )
    return build_direct_prompt_program_result(
        prompt_program,
        text,
        latency_ms=round((perf_counter() - started_at) * 1000, 3),
    )
