from __future__ import annotations

import anyio
import pytest
from typing_extensions import override

from market_support_crewai_agent.runtime import turn
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
    kernel_channel_type,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.planning import PlanSpec
from market_support_crewai_agent.runtime.recall.approved_static_catalog import (
    APPROVED_STATIC_MANIFEST_REF,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_agents import install_fake_planner
from tests.helpers.reply_contract_requests import make_v2_envelope


class _DocumentProvider:
    def __init__(self, contexts: tuple[GatewayDocumentContextV1, ...]) -> None:
        self._contexts: tuple[GatewayDocumentContextV1, ...] = contexts
        self.calls: int = 0
        self.cache_authorities: list[DocumentMcpCacheAuthorityV1 | None] = []

    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[GatewayDocumentContextV1, ...]:
        del request, evidence_query
        self.calls += 1
        self.cache_authorities.append(cache_authority)
        return self._contexts


class _StaticProvider:
    def __init__(self, contexts: tuple[GatewayStaticContextV1, ...]) -> None:
        self._contexts: tuple[GatewayStaticContextV1, ...] = contexts
        self.calls: int = 0

    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
    ) -> tuple[GatewayStaticContextV1, ...]:
        del request, evidence_query
        self.calls += 1
        return self._contexts


class _Composer:
    def __init__(self) -> None:
        self.inputs: list[ComposerPromptInputV1] = []

    async def compose(
        self,
        input_value: ComposerPromptInputV1,
    ) -> ComposerReplyOutput:
        self.inputs.append(input_value)
        return ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(kind="answer", text="已根据内部资料说明。", mentions=[]),
        )


class _EmptyPreflightService(AdapterPreflightService):
    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot.empty()


def _knowledge_plan(envelope: VerifiedRequestEnvelopeV1) -> PlanSpec:
    scope = business_scope_authority_v1(envelope.request.business_scope)
    return PlanSpec.model_validate(
        {
            "plan_id": "company-knowledge",
            "user_intent_summary": "answer internal company knowledge",
            "plan_units": [
                {
                    "unit_id": "company-knowledge",
                    "selected_capability_id": "answer_internal_company_knowledge",
                    "domain_scope": {
                        "kind": "distribution",
                        "business_scope_ref": scope.business_scope_ref,
                        "channel_kind": kernel_channel_type(envelope.request),
                    },
                    "answerability_policy": "answer",
                    "output_schema_ref": (
                        "answer_internal_company_knowledge:output_schema"
                    ),
                    "steps": [
                        {
                            "step_id": "company-query",
                            "description": "query internal company knowledge",
                            "evidence_query": "company profile",
                        }
                    ],
                }
            ],
        }
    )


def _install_planner(
    runtime: CrewAIReplyRuntime,
    envelope: VerifiedRequestEnvelopeV1,
) -> None:
    install_fake_planner(runtime, _knowledge_plan(envelope))


def _runtime(
    *,
    settings: Settings,
    gateway: InternalCompanyKnowledgeGatewayV1 | None = None,
    composer: _Composer | None = None,
) -> CrewAIReplyRuntime:
    return CrewAIReplyRuntime(
        settings,
        preflight_service=_EmptyPreflightService(settings=settings),
        coordinator=ReplyStateTransactionCoordinatorV1(),
        internal_company_knowledge_gateway=gateway,
        v2_composer=composer or _Composer(),
    )


def test_lifecycle_injected_gateway_receives_sealed_cache_authority_and_replays() -> (
    None
):
    # Given
    document = _DocumentProvider(
        (GatewayDocumentContextV1(document_id="company", text="Company profile"),)
    )
    static = _StaticProvider(())
    runtime = _runtime(
        settings=Settings(
            llm_api_key="test-key",
            doc_mcp_cache_ttl_seconds=30,
            reply_alignment_verifier_enabled=False,
        ),
        gateway=InternalCompanyKnowledgeGatewayV1(
            document_provider=document,
            static_provider=static,
        ),
    )
    envelope = make_v2_envelope("介绍一下公司")
    _install_planner(runtime, envelope)

    # When
    async def run() -> tuple[ReplyResponse, ReplyResponse]:
        response = await runtime.reply(envelope)
        replay = await runtime.reply(envelope)
        return response, replay

    response, replay = anyio.run(run)

    # Then
    assert replay == response
    assert response.reply.text == "已根据内部资料说明。"
    assert document.calls == static.calls == 1
    assert len(document.cache_authorities) == 1
    assert document.cache_authorities[0] is not None
    assert document.cache_authorities[0].state_key_ref == envelope.state_key_ref


def test_lifecycle_static_provider_remains_available_without_document_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    static = _StaticProvider(
        (
            GatewayStaticContextV1(
                entry_id="company_shareholders",
                manifest_ref=APPROVED_STATIC_MANIFEST_REF,
                text="Approved company profile",
            ),
        )
    )

    def document_client_must_not_be_built(settings: Settings) -> None:
        del settings
        raise AssertionError("Document MCP client must stay disabled")

    monkeypatch.setattr(turn, "DocumentMcpClient", document_client_must_not_be_built)
    monkeypatch.setattr(
        turn,
        "ApprovedStaticKnowledgeGatewayAdapter",
        lambda: static,
    )
    runtime = _runtime(
        settings=Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=False,
            doc_mcp_base_url="",
            reply_alignment_verifier_enabled=False,
        )
    )
    envelope = make_v2_envelope("介绍一下公司")
    _install_planner(runtime, envelope)

    # When
    response = anyio.run(runtime.reply, envelope)

    # Then
    assert response.reply.text == "已根据内部资料说明。"
    assert static.calls == 1


def test_lifecycle_disabled_knowledge_authority_does_zero_provider_or_cache_work() -> (
    None
):
    # Given
    document = _DocumentProvider(
        (GatewayDocumentContextV1(document_id="company", text="Company profile"),)
    )
    static = _StaticProvider(
        (
            GatewayStaticContextV1(
                entry_id="company",
                manifest_ref=APPROVED_STATIC_MANIFEST_REF,
                text="Approved profile",
            ),
        )
    )
    runtime = _runtime(
        settings=Settings(
            llm_api_key="test-key",
            doc_mcp_cache_ttl_seconds=30,
            reply_alignment_verifier_enabled=False,
        ),
        gateway=InternalCompanyKnowledgeGatewayV1(
            document_provider=document,
            static_provider=static,
        ),
    )
    envelope = make_v2_envelope(
        "介绍一下公司",
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": [],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
    _install_planner(runtime, envelope)

    # When/Then
    with pytest.raises(AgentRuntimeError, match="invalid PlanSpec contract"):
        _ = anyio.run(runtime.reply, envelope)
    assert document.calls == static.calls == 0
    assert document.cache_authorities == []
