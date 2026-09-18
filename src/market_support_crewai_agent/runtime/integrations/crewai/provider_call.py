from __future__ import annotations

import anyio
from anyio.to_thread import run_sync

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIProviderResultV1,
    CrewAITransportInvariantError,
    HealthInvocationLogContextV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.gemini_transport import (
    run_gemini_structured,
)
from market_support_crewai_agent.runtime.integrations.crewai.invocation_audit import (
    close_transport_error,
)
from market_support_crewai_agent.runtime.integrations.crewai.observability import (
    log_health_transport_error,
)
from market_support_crewai_agent.runtime.integrations.crewai.targeting import agent_role
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
    TurnLlmInvocationRowV1,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderTransportEnvelopeV1,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    ProviderMessageEnvelopeV1,
)


async def run_provider_call(
    agent: CrewAIAgentAdapterV1 | None,
    prompt_program: PromptProgram,
    envelope: ProviderTransportEnvelopeV1,
    request_captures: list[ProviderMessageEnvelopeV1],
    *,
    timeout_seconds: float | None,
    gemini_structured: bool,
) -> CrewAIProviderResultV1:
    with anyio.fail_after(timeout_seconds):
        if gemini_structured:
            return await run_sync(
                run_gemini_structured,
                agent,
                prompt_program,
                envelope,
                request_captures,
            )
        if agent is None:
            raise CrewAITransportInvariantError("crewai_agent_required")
        kickoff_result = await agent.kickoff(
            prompt_program.prompt_text,
            response_format=prompt_program.profile.response_model,
        )
    return CrewAIProviderResultV1(
        raw=kickoff_result.raw,
        pydantic=kickoff_result.pydantic,
        agent_role=agent_role(agent),
        usage_metrics=None,
    )


def close_provider_transport_error(
    journal: TurnLlmInvocationJournalV1 | None,
    reserved: TurnLlmInvocationRowV1 | None,
    prompt_program: PromptProgram,
    envelope: ProviderTransportEnvelopeV1,
    captures: list[ProviderMessageEnvelopeV1],
    error_code: ProviderFailureCodeV1,
    latency_ms: int,
    health_log_context: HealthInvocationLogContextV1 | None,
) -> None:
    close_transport_error(
        journal,
        reserved,
        prompt_program=prompt_program,
        envelope=envelope,
        captures=captures,
        error_code=error_code,
        latency_ms=latency_ms,
    )
    log_health_transport_error(
        prompt_program,
        health_log_context,
        captures,
        error_code=error_code,
        latency_ms=latency_ms,
    )
