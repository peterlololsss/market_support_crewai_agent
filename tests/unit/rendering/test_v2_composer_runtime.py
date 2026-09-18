from __future__ import annotations

from dataclasses import dataclass
from typing import final

import pytest
from typing_extensions import override

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
    SmalltalkComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    UnitBusinessFactsV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
    ExecutionUnitGroundingV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerPromptInputV1,
    V2Composer,
)
from market_support_crewai_agent.runtime.v2_attempt import (
    CandidatePlanRuntimeV1,
    build_candidate_from_plan_v2,
)
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from tests.unit.evidence._report_scope_evidence_fixtures import report_inputs
from tests.unit.llm._composer_stage_contract_fixtures import (
    ComposerScenario,
    composer_scenario,
)


@final
class _IgnoredPreflight:
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot.empty()


@final
class _FixedEvidenceExecutor(EvidenceExecutor):
    def __init__(self, result: CanonicalEvidenceExecutionResultV1) -> None:
        super().__init__(_IgnoredPreflight())
        self._result: CanonicalEvidenceExecutionResultV1 = result

    @override
    async def execute_v2(
        self,
        request: KernelReplyRequestV1,
        plan: ExecutionPlanV2,
        policy: PolicyManifestV2,
        *,
        scope_authority: BusinessScopeAuthorityV1,
        state_key_ref: str | None = None,
        document_cache_config: DocumentMcpCacheConfigV1 | None = None,
        alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
    ) -> CanonicalEvidenceExecutionResultV1:
        del (
            request,
            plan,
            policy,
            scope_authority,
            state_key_ref,
            document_cache_config,
            alignment_refetch_request,
        )
        return self._result


@dataclass(frozen=True, slots=True)
class _Runtime(CandidatePlanRuntimeV1):
    evidence_executor: EvidenceExecutor
    document_cache_config: DocumentMcpCacheConfigV1 | None
    locator_safety: LocatorSafetyClassifierV1
    v2_composer: V2Composer | None


class _CapturingComposer:
    def __init__(self, response_text: str) -> None:
        self.response_text: str = response_text
        self.inputs: list[ComposerPromptInputV1] = []

    async def compose(self, input_value: ComposerPromptInputV1) -> ComposerReplyOutput:
        self.inputs.append(input_value)
        return ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(kind="answer", text=self.response_text, mentions=[]),
        )


@dataclass(frozen=True, slots=True)
class _ComposerTrap:
    async def compose(self, input_value: ComposerPromptInputV1) -> ComposerReplyOutput:
        del input_value
        raise AssertionError("composer_must_not_run_without_product_list_evidence")


def _runtime(
    evidence: CanonicalEvidenceExecutionResultV1,
    composer: V2Composer | None,
    locator_safety: LocatorSafetyClassifierV1,
) -> _Runtime:
    return _Runtime(
        evidence_executor=_FixedEvidenceExecutor(evidence),
        document_cache_config=None,
        locator_safety=locator_safety,
        v2_composer=composer,
    )


def _scenario_evidence(
    scenario: ComposerScenario,
) -> CanonicalEvidenceExecutionResultV1:
    return CanonicalEvidenceExecutionResultV1(
        preflight=scenario.invocation.preflight,
        canonical_facts=tuple(
            fact
            for grounding in scenario.groundings
            for fact in grounding.allowed_evidence
        ),
        resolve_bindings=(),
        groundings=scenario.groundings,
        domain_context=DomainContextV1Builder().build(
            scenario.request,
            scope_authority=scenario.authority,
        ),
        media_bindings=scenario.invocation.media_bindings,
    )


def _empty_product_evidence() -> tuple[
    KernelReplyRequestV1,
    PolicyManifestV2,
    BusinessScopeAuthorityV1,
    ExecutionPlanV2,
    CanonicalEvidenceExecutionResultV1,
]:
    source = report_inputs()
    unit = source.plan.units[0]
    grounding = ExecutionUnitGroundingV1(
        unit_id=unit.unit_id,
        manifest_ref=unit.manifest_ref,
        answerability=unit.answerability_policy,
        scope=unit.scope,
        evidence_query=unit.evidence_query,
        action_intents=unit.action_intents,
        business_facts=UnitBusinessFactsV1(evidence_fact_count=0),
    )
    evidence = CanonicalEvidenceExecutionResultV1(
        preflight=source.preflight,
        canonical_facts=(),
        resolve_bindings=(),
        groundings=(grounding,),
        domain_context=DomainContextV1Builder().build(
            source.request,
            scope_authority=source.scope,
        ),
    )
    return source.request, source.policy, source.scope, source.plan, evidence


@pytest.mark.anyio
async def test_product_list_without_admitted_products_abstains_before_composer() -> (
    None
):
    # Given: an eligible product-list plan with no admitted product evidence.
    request, policy, scope, plan, evidence = _empty_product_evidence()
    runtime = _runtime(
        evidence,
        _ComposerTrap(),
        LocatorSafetyClassifierV1(internal_origins=frozenset(), secrets=()),
    )

    # When: the public V2 candidate boundary renders the plan.
    result = await build_candidate_from_plan_v2(
        runtime,
        request=request,
        policy=policy,
        scope_authority=scope,
        plan=plan,
    )

    # Then: it abstains before the composer can fabricate a product list.
    assert result.response.reply.kind == "unable_to_answer"
    assert result.reason_code == "product_list_evidence_missing"


@pytest.mark.anyio
async def test_active_v2_attempt_uses_injected_knowledge_composer() -> None:
    # Given: an eligible knowledge-answer plan and a concrete composer.
    scenario = composer_scenario("knowledge_answer")
    composer = _CapturingComposer("策略说明")

    # When: the active V2 boundary builds the candidate.
    result = await build_candidate_from_plan_v2(
        _runtime(
            _scenario_evidence(scenario),
            composer,
            scenario.invocation.locator_safety,
        ),
        request=scenario.request,
        policy=scenario.invocation.policy,
        scope_authority=scenario.authority,
        plan=scenario.plan,
    )

    # Then: the public response and injected role input agree with the plan.
    assert result.response.reply.text == "策略说明"
    assert result.reason_code == "knowledge_answer_composer"
    assert isinstance(composer.inputs[0], KnowledgeComposerPromptInputV1)


@pytest.mark.anyio
async def test_active_v2_attempt_uses_injected_smalltalk_composer() -> None:
    # Given: a smalltalk plan and a concrete composer.
    scenario = composer_scenario("smalltalk")
    composer = _CapturingComposer("你好")

    # When: the active V2 boundary builds the candidate.
    result = await build_candidate_from_plan_v2(
        _runtime(
            _scenario_evidence(scenario),
            composer,
            scenario.invocation.locator_safety,
        ),
        request=scenario.request,
        policy=scenario.invocation.policy,
        scope_authority=scenario.authority,
        plan=scenario.plan,
    )

    # Then: the smalltalk projection omits knowledge groundings.
    assert result.response.reply.text == "你好"
    assert result.reason_code == "smalltalk_composer"
    assert isinstance(composer.inputs[0], SmalltalkComposerPromptInputV1)
    assert "unit_groundings" not in composer.inputs[0].model_dump(mode="json")


@pytest.mark.anyio
async def test_active_v2_composer_unavailable_returns_safe_abstention() -> None:
    # Given: an eligible knowledge plan with no configured composer.
    scenario = composer_scenario("knowledge_answer")

    # When: the active V2 boundary attempts to build a candidate.
    result = await build_candidate_from_plan_v2(
        _runtime(
            _scenario_evidence(scenario),
            None,
            scenario.invocation.locator_safety,
        ),
        request=scenario.request,
        policy=scenario.invocation.policy,
        scope_authority=scenario.authority,
        plan=scenario.plan,
    )

    # Then: no composer becomes a safe, typed abstention.
    assert result.response.reply.kind == "unable_to_answer"
    assert result.reason_code == "composer_not_available"
