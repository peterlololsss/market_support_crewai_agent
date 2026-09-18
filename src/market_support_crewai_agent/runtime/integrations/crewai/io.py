from __future__ import annotations

from time import perf_counter
from typing import ClassVar, Literal

import anyio
from pydantic import ConfigDict, Field, JsonValue

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIProviderResultV1,
    HealthInvocationLogContextV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.invocation_audit import (
    close_journal_row,
)
from market_support_crewai_agent.runtime.integrations.crewai.observability import (
    TraceAttributesV1,
    compact_crewai_execution,
    log_health_execution,
    log_llm_execution,
    record_llm_success,
    trace_attributes,
)
from market_support_crewai_agent.runtime.integrations.crewai.provider_call import (
    close_provider_transport_error,
    run_provider_call,
)
from market_support_crewai_agent.runtime.integrations.crewai.request_capture import (
    install_crewai_request_capture,
    require_request_capture,
)
from market_support_crewai_agent.runtime.integrations.crewai.targeting import (
    crewai_transport_envelope,
    is_gemini_agent,
    reject_invalid_dispatch,
    target_slot_for_stage,
)
from market_support_crewai_agent.runtime.observability.runtime_trace import trace_span
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.direct_provider_io_capture import (
    DirectProviderIoCaptureError,
    DirectProviderIoCaptureScopeV1,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
    resolve_turn_llm_invocation_journal,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
    provider_failure_code_from_exception,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    ProviderMessageEnvelopeV1,
)
from market_support_crewai_agent.runtime.prompts.provider_transport import (
    capture_provider_text_output,
)
from market_support_crewai_agent.schemas.base import StrictModel


class DirectSafeProviderTransportV1(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["direct-safe-provider-transport.v1"] = (
        "direct-safe-provider-transport.v1"
    )
    osh1: str = Field(pattern=r"^osh1:[0-9a-f]{64}$")
    poh1: str = Field(pattern=r"^poh1:[0-9a-f]{64}$")
    prh1: str = Field(pattern=r"^prh1:[0-9a-f]{64}$")
    out1: str | None = Field(default=None, pattern=r"^out1:[0-9a-f]{64}$")
    status: Literal["success", "output_contract_error", "transport_error"]
    error_code: str | None = Field(default=None, max_length=80)


async def run_crewai_kickoff(
    agent: CrewAIAgentAdapterV1 | None,
    prompt_program: PromptProgram,
    *,
    timeout_seconds: float | None,
    journal: TurnLlmInvocationJournalV1 | None = None,
    health_log_context: HealthInvocationLogContextV1 | None = None,
) -> tuple[CrewAIProviderResultV1, dict[str, JsonValue]]:
    reject_invalid_dispatch(prompt_program, health_log_context is not None)
    journal = resolve_turn_llm_invocation_journal(journal)
    started_at = perf_counter()
    gemini_structured = is_gemini_agent(agent)
    envelope = crewai_transport_envelope(agent, prompt_program)
    request_captures: list[ProviderMessageEnvelopeV1] = []
    reserved = (
        journal.reserve(
            stage_kind=prompt_program.profile.stage,
            program_id=prompt_program.program_id,
            target_slot=target_slot_for_stage(prompt_program.profile.stage),
            purpose="crewai_kickoff",
        )
        if journal is not None
        else None
    )
    restore_crewai_capture = None
    try:
        restore_crewai_capture = (
            None
            if gemini_structured
            else install_crewai_request_capture(
                agent,
                prompt_program,
                envelope,
                request_captures,
            )
        )
        try:
            trace_attrs = _typed_trace_attributes(
                trace_attributes(
                    prompt_program,
                    osh1=envelope.osh1(),
                    poh1=envelope.poh1(),
                    health_log_context=health_log_context,
                )
            )
            with trace_span(
                "llm.crewai_kickoff",
                **trace_attrs,
            ):
                async with DirectProviderIoCaptureScopeV1():
                    result = await run_provider_call(
                        agent,
                        prompt_program,
                        envelope,
                        request_captures,
                        timeout_seconds=timeout_seconds,
                        gemini_structured=gemini_structured,
                    )
            request_capture = require_request_capture(request_captures)
        finally:
            if restore_crewai_capture is not None:
                restore_crewai_capture()
    except DirectProviderIoCaptureError:
        close_provider_transport_error(
            journal,
            reserved,
            prompt_program,
            envelope,
            request_captures,
            "direct_provider_stdio_violation",
            round((perf_counter() - started_at) * 1000),
            health_log_context,
        )
        raise ProviderInvocationError("direct_provider_stdio_violation") from None
    except TimeoutError:
        close_provider_transport_error(
            journal,
            reserved,
            prompt_program,
            envelope,
            request_captures,
            "provider_timeout",
            round((perf_counter() - started_at) * 1000),
            health_log_context,
        )
        raise ProviderInvocationError("provider_timeout") from None
    except ProviderInvocationError as exc:
        close_provider_transport_error(
            journal,
            reserved,
            prompt_program,
            envelope,
            request_captures,
            exc.code,
            round((perf_counter() - started_at) * 1000),
            health_log_context,
        )
        raise ProviderInvocationError(exc.code) from None
    except anyio.get_cancelled_exc_class():
        close_provider_transport_error(
            journal,
            reserved,
            prompt_program,
            envelope,
            request_captures,
            "provider_transport_unavailable",
            round((perf_counter() - started_at) * 1000),
            health_log_context,
        )
        raise
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
        code = provider_failure_code_from_exception(exc)
        close_provider_transport_error(
            journal,
            reserved,
            prompt_program,
            envelope,
            request_captures,
            code,
            round((perf_counter() - started_at) * 1000),
            health_log_context,
        )
        raise ProviderInvocationError(code) from None
    latency_ms = (perf_counter() - started_at) * 1000
    output_capture = capture_provider_text_output(
        result.raw,
        response_model=prompt_program.profile.response_model,
    )
    if reserved is not None and journal is not None:
        close_journal_row(
            journal,
            reserved,
            envelope=envelope,
            request_capture=request_capture,
            output_capture=output_capture,
            prompt_program=prompt_program,
            latency_ms=round(latency_ms),
        )
    transport = DirectSafeProviderTransportV1(
        osh1=envelope.osh1(),
        poh1=envelope.poh1(),
        prh1=request_capture.prh1(),
        out1=output_capture.out1()
        if output_capture.status == "available_text"
        else None,
        status="success"
        if output_capture.status == "available_text"
        else "output_contract_error",
        error_code=output_capture.error_code,
    )
    if health_log_context is None:
        log_llm_execution(
            prompt_program,
            result,
            latency_ms,
            mode="gemini_structured" if gemini_structured else "crewai",
            prh1=transport.prh1,
            out1=transport.out1,
        )
    else:
        log_health_execution(
            prompt_program,
            health_log_context,
            status=transport.status,
            error_code=transport.error_code,
            input_bytes=request_capture.byte_count,
            output_bytes=output_capture.byte_count
            if output_capture.status == "available_text"
            else None,
            latency_ms=latency_ms,
        )
    if result.pydantic is not None:
        record_llm_success(agent, prompt_program.profile.stage)
    return result, compact_crewai_execution(prompt_program, result, latency_ms)


def _typed_trace_attributes(
    attributes: TraceAttributesV1,
) -> dict[str, str | int]:
    return {
        name: value
        for name, value in attributes.items()
        if isinstance(value, (str, int))
    }
