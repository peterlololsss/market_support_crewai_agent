from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import anyio
import pytest
from typing_extensions import override

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
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.planning import (
    ExecutionPlanV2,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerPromptInputV1,
    V2Composer,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.v2_attempt import (
    CandidatePlanRuntimeV1,
    V2AttemptResult,
    build_candidate_from_plan_v2,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_json import JsonInput
from tests.helpers.reply_contract_requests import make_v2_envelope


class _EmptyPreflight:
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot.empty()


class _DocumentProvider(PostPlanDocumentKnowledgeProvider):
    def __init__(self) -> None:
        self.calls: int = 0
        self.cache_authorities: list[DocumentMcpCacheAuthorityV1 | None] = []

    @override
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
        return (
            GatewayDocumentContextV1(document_id="company", text="Company context"),
        )


class _StaticProvider(PostPlanStaticKnowledgeProvider):
    def __init__(self) -> None:
        self.calls: int = 0

    @override
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
    ) -> tuple[GatewayStaticContextV1, ...]:
        del request, evidence_query
        self.calls += 1
        return ()


class _Composer:
    async def compose(self, input_value: ComposerPromptInputV1) -> ComposerReplyOutput:
        del input_value
        return ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(kind="answer", text="已根据资料说明。", mentions=[]),
        )


@dataclass(frozen=True, slots=True)
class _Runtime(CandidatePlanRuntimeV1):
    evidence_executor: EvidenceExecutor
    document_cache_config: DocumentMcpCacheConfigV1 | None
    locator_safety: LocatorSafetyClassifierV1
    v2_composer: V2Composer | None


def _cache_config() -> DocumentMcpCacheConfigV1:
    return DocumentMcpCacheConfigV1(
        client_contract_version="document-mcp-client.v1",
        corpus_version="company-public.v1",
        request_schema_hash="osh1:" + "1" * 64,
        response_schema_hash="osh1:" + "2" * 64,
        timeout_milliseconds=1_000,
        max_candidates=8,
        ttl_seconds=30,
        capacity=8,
    )


def _runtime(
    document: PostPlanDocumentKnowledgeProvider,
    static: PostPlanStaticKnowledgeProvider,
) -> _Runtime:
    settings = Settings(llm_api_key="test-key")
    return _Runtime(
        evidence_executor=EvidenceExecutor(
            _EmptyPreflight(),
            internal_company_knowledge_gateway=InternalCompanyKnowledgeGatewayV1(
                document_provider=document,
                static_provider=static,
            ),
        ),
        document_cache_config=_cache_config(),
        locator_safety=LocatorSafetyClassifierV1.from_settings(settings),
        v2_composer=_Composer(),
    )


def _inputs(
    *,
    scene: Literal["direct", "group"],
    enabled: bool,
) -> tuple[
    VerifiedRequestEnvelopeV1,
    BusinessScopeAuthorityV1,
    PolicyManifestV2,
    ExecutionPlanV2,
]:
    identity: dict[str, JsonInput]
    presentation: dict[str, JsonInput]
    scope: dict[str, JsonInput]
    if scene == "direct":
        identity = {
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:test",
            "direct_thread_ref": "direct:thread-1",
            "principal_ref": "principal:sender-1",
        }
        presentation = {
            "contract_version": "direct-presentation.v1",
            "principal_name": "p",
        }
        scope = {"kind": "unscoped"}
        manifest_id = "answer_internal_company_knowledge"
    else:
        identity = {
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": "tenant:test",
            "group_ref": "group:group-1",
            "principal_ref": "principal:sender-1",
        }
        presentation = {
            "contract_version": "group-presentation.v1",
            "conversation_name": "g",
            "principal_name": "p",
        }
        scope = {
            "kind": "distribution",
            "dist_channel_name": "d",
            "channel_type": "bank",
            "available_artifacts": [],
        }
        manifest_id = "answer_internal_company_knowledge"
    envelope = make_v2_envelope(
        identity=identity,
        presentation=presentation,
        business_scope=scope,
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"] if enabled else [],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
    authority = business_scope_authority_v1(envelope.request.business_scope)
    policy = PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(
            envelope.request,
            authority,
            group_recall_mode="shortcut",
        ),
        policy_ledger_summary_v1((), 0),
    )
    plan_policy = policy
    if not enabled:
        _, _, enabled_policy, _ = _inputs(scene=scene, enabled=True)
        plan_policy = enabled_policy
    plan = finalize_execution_plan_v2(
        DeterministicPlanOriginInputV1(
            user_need="company knowledge",
            units=(
                DeterministicPlanUnitV1(
                    unit_id="company-knowledge",
                    manifest_id=manifest_id,
                    answerability_policy="answer",
                    evidence_query="company context",
                ),
            ),
        ),
        plan_policy,
        authority,
        origin="deterministic",
    )
    return envelope, authority, policy, plan


def _run_candidate(
    runtime: _Runtime,
    envelope: VerifiedRequestEnvelopeV1,
    authority: BusinessScopeAuthorityV1,
    policy: PolicyManifestV2,
    plan: ExecutionPlanV2,
) -> V2AttemptResult:
    async def run() -> V2AttemptResult:
        return await build_candidate_from_plan_v2(
            runtime,
            request=envelope.request,
            policy=policy,
            scope_authority=authority,
            plan=plan,
            state_key_ref=envelope.state_key_ref,
        )

    return anyio.run(run)


def test_active_v2_direct_disabled_gateway_does_zero_provider_work() -> None:
    envelope, authority, policy, plan = _inputs(scene="direct", enabled=False)
    document = _DocumentProvider()
    static = _StaticProvider()

    with pytest.raises(AgentRuntimeError, match="execution_plan_v2_invalid"):
        _ = _run_candidate(
            _runtime(document, static), envelope, authority, policy, plan
        )

    assert document.calls == static.calls == 0


def test_active_v2_group_selected_gateway_gets_verified_cache_authority() -> None:
    envelope, authority, policy, plan = _inputs(scene="group", enabled=True)
    document = _DocumentProvider()
    static = _StaticProvider()

    result = _run_candidate(
        _runtime(document, static), envelope, authority, policy, plan
    )

    assert result.reply_validation.valid
    assert result.response.reply.text == "已根据资料说明。"
    assert document.calls == static.calls == 1
    assert document.cache_authorities[0] is not None
    assert document.cache_authorities[0].state_key_ref == envelope.state_key_ref


def test_direct_internal_company_knowledge_gateway_runs_after_plan_with_recall_off() -> (
    None
):
    envelope, authority, policy, plan = _inputs(scene="direct", enabled=True)
    document = _DocumentProvider()
    static = _StaticProvider()

    result = _run_candidate(
        _runtime(document, static), envelope, authority, policy, plan
    )

    assert policy.recall_mode == "off"
    assert result.reply_validation.valid
    assert result.response.actions == []
    assert result.response.reply.mentions == []
    assert document.calls == static.calls == 1
