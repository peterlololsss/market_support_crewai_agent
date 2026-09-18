from __future__ import annotations

from collections.abc import Mapping

from market_support_crewai_agent.health.models import (
    HealthProbeResponse,
    HealthTargetSlot,
    LlmHealthTarget,
)
from market_support_crewai_agent.runtime.hashing import health_target_hmac
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAILlmAdapterV1,
    CrewAIProviderResultV1,
    HealthInvocationLogContextV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.io import (
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    LlmHealthProbeOutputV1,
)
from market_support_crewai_agent.runtime.prompts.neutral_program import (
    assemble_neutral_prompt_program,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderIdV1,
    TransportVariantV1,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
    ProviderInvocationError,
    provider_failure_code_from_exception,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    ProviderTargetError,
    build_provider_target_identity,
)
from market_support_crewai_agent.settings_model import Settings


def discover_llm_health_targets(
    settings: Settings,
    *,
    process_health_key: bytes,
) -> tuple[LlmHealthTarget, ...]:
    rows: tuple[tuple[HealthTargetSlot, str, str, str], ...] = (
        (
            "composer",
            settings.llm_provider,
            settings.llm_model,
            settings.llm_base_url,
        ),
        (
            "planner",
            settings.planner_llm_provider,
            settings.planner_llm_model,
            settings.planner_llm_base_url,
        ),
    )
    targets: list[LlmHealthTarget] = []
    for target_slot, provider, model_name, base_url in rows:
        identity = build_provider_target_identity(
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            target_slot=target_slot,
        )
        targets.append(
            LlmHealthTarget(
                target_slot=target_slot,
                htk1=health_target_key(
                    process_health_key=process_health_key,
                    target_slot=target_slot,
                    provider_id=identity.provider_id,
                    model_family=identity.model_family,
                    model_name=identity.model_name,
                    normalized_endpoint=identity.normalized_endpoint,
                    transport_variant=identity.transport_variant,
                ),
            )
        )
    return tuple(targets)


def probe_retry_reason(result: CrewAIProviderResultV1) -> ProviderFailureCodeV1 | None:
    parsed = getattr(result, "pydantic", None)
    if isinstance(parsed, (HealthProbeResponse, LlmHealthProbeOutputV1)) and parsed.ok:
        return None
    return "provider_output_contract"


def health_probe_program() -> PromptProgram:
    return assemble_neutral_prompt_program(
        stage="llm_health_probe",
        response_model=LlmHealthProbeOutputV1,
        temperature=0.0,
        max_tokens=64,
    )


def health_target_key(
    *,
    process_health_key: bytes,
    target_slot: HealthTargetSlot,
    provider_id: ProviderIdV1,
    model_family: ModelFamily,
    model_name: str,
    normalized_endpoint: str,
    transport_variant: TransportVariantV1,
) -> str:
    return health_target_hmac(
        process_health_key,
        {
            "target_slot": target_slot,
            "provider_id": provider_id,
            "model_family": model_family,
            "model_name": model_name,
            "normalized_endpoint": normalized_endpoint,
            "transport_variant": transport_variant,
        },
    )


async def dispatch_health_probe(
    agent: CrewAIAgentAdapterV1,
    target: LlmHealthTarget,
    *,
    timeout_seconds: float,
) -> ProviderFailureCodeV1 | None:
    try:
        result, _execution = await run_crewai_kickoff(
            agent,
            health_probe_program(),
            timeout_seconds=timeout_seconds,
            health_log_context=HealthInvocationLogContextV1(
                target_slot=target.target_slot,
                htk1=target.htk1,
            ),
        )
        return probe_retry_reason(result)
    except ProviderInvocationError as exc:
        return exc.code
    except TimeoutError:
        return "provider_timeout"
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return provider_failure_code_from_exception(exc)


def target_for_agent(
    agent: CrewAIAgentAdapterV1 | None,
    stage: str,
    targets_by_key: Mapping[str, LlmHealthTarget],
    process_health_key: bytes,
) -> LlmHealthTarget | None:
    if agent is None:
        return None
    llm = agent.llm
    if not llm.model:
        return None
    target_slot: HealthTargetSlot = (
        "planner" if stage == "planner_intent" else "composer"
    )
    try:
        identity = build_provider_target_identity(
            provider=llm.provider,
            model_name=llm.model,
            base_url=agent_base_url(llm),
            target_slot=target_slot,
        )
    except ProviderTargetError:
        return None
    key = health_target_key(
        process_health_key=process_health_key,
        target_slot=target_slot,
        provider_id=identity.provider_id,
        model_family=identity.model_family,
        model_name=identity.model_name,
        normalized_endpoint=identity.normalized_endpoint,
        transport_variant=identity.transport_variant,
    )
    return targets_by_key.get(key)


def agent_base_url(llm: CrewAILlmAdapterV1) -> str:
    if llm.base_url:
        return llm.base_url
    http_options = llm.client_params.get("http_options")
    if not isinstance(http_options, dict):
        return ""
    value = http_options.get("base_url")
    return value if isinstance(value, str) else ""
