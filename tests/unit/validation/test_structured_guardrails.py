from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import pytest
from typing_extensions import override

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
    SmalltalkComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
    PostPlanDocumentKnowledgeProvider,
    PostPlanStaticKnowledgeProvider,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.recall.approved_static_catalog import (
    APPROVED_STATIC_MANIFEST_REF,
)
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
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from tests.unit.llm._composer_stage_contract_fixtures import composer_scenario


@dataclass(frozen=True, slots=True)
class _DocumentProvider(PostPlanDocumentKnowledgeProvider):
    @override
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[GatewayDocumentContextV1, ...]:
        del request, evidence_query, cache_authority
        return ()


@dataclass(frozen=True, slots=True)
class _StaticProvider(PostPlanStaticKnowledgeProvider):
    @override
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
    ) -> tuple[GatewayStaticContextV1, ...]:
        del request, evidence_query
        return (
            GatewayStaticContextV1(
                entry_id="company_shareholders",
                manifest_ref=APPROVED_STATIC_MANIFEST_REF,
                text="Approved company context.",
            ),
        )


class _EmptyPreflight:
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot.empty()


@dataclass(frozen=True, slots=True)
class _MarkerComposer:
    marker: str

    async def compose(
        self,
        input_value: ComposerPromptInputV1,
    ) -> ComposerReplyOutput:
        match input_value:
            case KnowledgeComposerPromptInputV1(unit_groundings=groundings):
                evidence_ids = [groundings[0].allowed_evidence_ids[0]]
            case SmalltalkComposerPromptInputV1():
                raise AssertionError("knowledge_composer_input_required")
            case _:
                assert_never(input_value)
        return ComposerReplyOutput(
            response_mode="answer",
            evidence_ids=evidence_ids,
            reply=PrimaryReply(kind="answer", text=self.marker, mentions=[]),
        )


@dataclass(frozen=True, slots=True)
class _CandidateRuntimeStub(CandidatePlanRuntimeV1):
    evidence_executor: EvidenceExecutor
    v2_composer: V2Composer | None
    locator_safety: LocatorSafetyClassifierV1
    document_cache_config: DocumentMcpCacheConfigV1 | None


def _evidence_executor() -> EvidenceExecutor:
    return EvidenceExecutor(
        _EmptyPreflight(),
        internal_company_knowledge_gateway=InternalCompanyKnowledgeGatewayV1(
            document_provider=_DocumentProvider(),
            static_provider=_StaticProvider(),
        ),
    )


@pytest.mark.anyio
async def test_active_v2_public_boundary_rejects_direct_image_marker() -> None:
    # Given: direct knowledge evidence and a composer returning an uncited media marker.
    scenario = composer_scenario("knowledge_answer", direct=True)
    marker = "%%company_shareholders.png%%"
    runtime = _CandidateRuntimeStub(
        evidence_executor=_evidence_executor(),
        v2_composer=_MarkerComposer(marker=marker),
        locator_safety=LocatorSafetyClassifierV1(
            internal_origins=frozenset(),
            secrets=(),
        ),
        document_cache_config=None,
    )

    # When: the public candidate builder executes the complete active V2 boundary.
    result = await build_candidate_from_plan_v2(
        runtime,
        request=scenario.request,
        policy=scenario.invocation.policy,
        scope_authority=scenario.authority,
        plan=scenario.plan,
    )

    # Then: the marker is replaced by a validated, effect-free abstention.
    assert result.reason_code == "composer_output_rejected"
    assert result.response.reply.kind == "unable_to_answer"
    assert marker not in result.response.reply.text
    assert result.response.reply.mentions == []
    assert result.response.actions == []
    assert result.reply_validation.valid is True
