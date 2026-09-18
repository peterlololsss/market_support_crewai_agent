from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import final
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.context.stage_inputs import (
    SanitizedAlignmentVerifierInputV1,
    build_alignment_verifier_prompt_input_v1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.prompts.context import (
    render_prompt_context_layers,
)
from market_support_crewai_agent.runtime.validation.alignment import (
    AlignmentVerifierInvocationV1,
    InternalAlignmentVerifierSourceV1,
    build_runtime_alignment_verifier_input_v1,
    verify_reply_alignment,
)
from market_support_crewai_agent.runtime.validation.alignment_runtime_contracts import (
    AlignmentAgentFactoryV1,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    NoopReplyAlignmentVerifier,
    ReplyAlignmentVerdict,
    ReplyAlignmentVerifier,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_completion_agent_adapter
from tests.unit.llm._composer_stage_contract_fixtures import composer_scenario
from tests.unit.llm._stage_input_fixtures import stage_sources


@pytest.mark.anyio
async def test_noop_verifier_consumes_the_strict_stage_dto() -> None:
    # Given: one fully validated verifier input with selected contracts and groundings.
    input_value = build_alignment_verifier_prompt_input_v1(stage_sources()[3])

    # When: the no-op implementation verifies the same injected protocol object.
    verdict = await NoopReplyAlignmentVerifier().verify(input_value)

    # Then: it returns the unchanged closed verdict contract.
    assert isinstance(input_value, SanitizedAlignmentVerifierInputV1)
    assert verdict == ReplyAlignmentVerdict(
        aligned=True,
        safe_to_return=True,
        confidence=1.0,
    )


def test_verdict_rejects_verifier_authored_repairs_or_actions() -> None:
    # Given: a model output that attempts to repair the response and authorize an action.
    payload = {
        "aligned": False,
        "safe_to_return": False,
        "failure_code": "wrong_action",
        "remediation": "recompose",
        "repaired_response": {"reply": {"text": "forged"}},
        "actions": [{"type": "send_weekly_report"}],
    }

    # When/Then: the unchanged verdict boundary rejects both undeclared authority fields.
    with pytest.raises(ValidationError):
        _ = ReplyAlignmentVerdict.model_validate(payload)


@pytest.mark.anyio
async def test_internal_and_injected_verifiers_receive_identical_strict_dto() -> None:
    # Given: one repeated-unit candidate and both verifier transport paths.
    scenario = composer_scenario("knowledge_answer", unit_count=2)
    input_value = build_runtime_alignment_verifier_input_v1(
        AlignmentVerifierInvocationV1(
            request=scenario.request,
            policy=scenario.invocation.policy,
            scope_authority=scenario.authority,
            plan=scenario.plan,
            evidence=CanonicalEvidenceExecutionResultV1(
                preflight=AdapterPreflightSnapshot.empty(),
                canonical_facts=(),
                resolve_bindings=(),
                groundings=scenario.groundings,
                domain_context=DomainContextV1Builder().build(
                    scenario.request,
                    scope_authority=scenario.authority,
                ),
            ),
            response=ReplyResponse(
                reply=PrimaryReply(kind="answer", text="grounded answer")
            ),
            reason_code="knowledge_answer_composer",
            history=(),
            attempt=0,
            now=datetime(2026, 7, 19, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
            locator_safety=LocatorSafetyClassifierV1(
                internal_origins=frozenset(),
                secrets=(),
            ),
        )
    )
    injected = _CaptureVerifier()
    internal_agent = _CaptureAgent()
    settings = Settings(llm_api_key="test-key")
    alignment_agent_factory = _CaptureAlignmentAgentFactory(
        _capturing_agent_adapter(internal_agent, settings)
    )
    injected_runtime = _Runtime(
        settings=settings,
        alignment_agent_factory=alignment_agent_factory,
        alignment_verifier=injected,
    )
    internal_runtime = _Runtime(
        alignment_verifier=None,
        settings=settings,
        alignment_agent_factory=alignment_agent_factory,
    )
    internal_source = InternalAlignmentVerifierSourceV1(
        model_family="generic",
        prompt_programs=[],
        llm_executions=[],
    )

    # When: the same DTO is dispatched through injected and internal verification.
    injected_verdict = await verify_reply_alignment(
        injected_runtime,
        input_value,
        internal_source,
    )
    internal_verdict = await verify_reply_alignment(
        internal_runtime,
        input_value,
        internal_source,
    )

    # Then: external identity and internal serialized bytes are exactly the same input.
    strict_runtime = render_prompt_context_layers(input_value)["runtime"].strip()
    assert injected.input_value is input_value
    assert internal_agent.prompt.count(strict_runtime) == 1
    assert injected_verdict == internal_verdict
    assert (
        input_value.selected_capabilities.manifest_refs
        == scenario.plan.selected_manifest_refs
    )
    assert tuple(item.unit_id for item in input_value.unit_groundings) == (
        "unit-1",
        "unit-2",
    )


@final
class _CaptureVerifier:
    def __init__(self) -> None:
        self.input_value: SanitizedAlignmentVerifierInputV1 | None = None

    async def verify(
        self,
        input_value: SanitizedAlignmentVerifierInputV1,
    ) -> ReplyAlignmentVerdict:
        self.input_value = input_value
        return ReplyAlignmentVerdict(aligned=True, safe_to_return=True)


@final
class _CaptureAgent:
    def __init__(self) -> None:
        self.prompt = ""

    def observe_prompt(self, prompt: str) -> None:
        self.prompt = prompt


def _capturing_agent_adapter(
    agent: _CaptureAgent,
    settings: Settings,
) -> CrewAIAgentAdapterV1:
    return make_completion_agent_adapter(
        lambda _response_format: ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
        ),
        role="alignment_verifier",
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        on_prompt=agent.observe_prompt,
    )


@final
class _CaptureAlignmentAgentFactory:
    def __init__(self, agent: CrewAIAgentAdapterV1) -> None:
        self._agent = agent

    def build_alignment_verifier_agent(self) -> CrewAIAgentAdapterV1:
        return self._agent


@dataclass(frozen=True, slots=True)
class _Runtime:
    settings: Settings
    alignment_agent_factory: AlignmentAgentFactoryV1
    alignment_verifier: ReplyAlignmentVerifier | None
