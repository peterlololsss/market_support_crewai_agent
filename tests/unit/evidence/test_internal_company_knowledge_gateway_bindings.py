from __future__ import annotations

import anyio
import pytest

from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
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
from market_support_crewai_agent.runtime.recall.approved_static_catalog import (
    APPROVED_STATIC_MANIFEST_REF,
)
from tests.unit.evidence.internal_company_knowledge_gateway_fixtures import (
    DocumentProvider,
    EmptyPreflight,
    StaticProvider,
    cache_config,
    collect_gateway,
    direct_inputs,
    group_inputs,
)


def test_gateway_collects_both_sources_for_a_selected_group_plan_with_group_scope() -> (
    None
):
    request, policy, plan, state_key_ref = group_inputs()
    document = DocumentProvider(
        (GatewayDocumentContextV1(document_id="strategy", text="Strategy context"),)
    )
    static = StaticProvider(
        (
            GatewayStaticContextV1(
                entry_id="company_shareholders",
                manifest_ref=APPROVED_STATIC_MANIFEST_REF,
                text="Company context",
            ),
        )
    )
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=document,
        static_provider=static,
    )

    result = collect_gateway(
        gateway,
        request=request,
        plan=plan,
        policy=policy,
        state_key_ref=state_key_ref,
        document_cache_config=cache_config(),
    )

    assert len(result.facts) == 2
    assert all(fact.scope.kind == "distribution" for fact in result.facts)
    assert document.calls == static.calls == 1


@pytest.mark.parametrize("state_key_ref", (None, "not-a-canonical-state-ref"))
def test_gateway_rejects_unselected_media_and_drops_cache_authority_for_invalid_state_ref(
    state_key_ref: str | None,
) -> None:
    request, policy, plan, _ = direct_inputs(enabled=True)
    document = DocumentProvider(())
    static = StaticProvider(
        (
            GatewayStaticContextV1(
                entry_id="company_shareholders",
                manifest_ref=APPROVED_STATIC_MANIFEST_REF,
                text="%%company_shareholders.png%%",
                selected_asset_ids=("not-registered",),
            ),
        )
    )
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=document,
        static_provider=static,
    )

    result = collect_gateway(
        gateway,
        request=request,
        plan=plan,
        policy=policy,
        state_key_ref=state_key_ref,
        document_cache_config=cache_config(),
    )

    assert result.facts == ()
    assert result.media_bindings == ()
    assert document.cache_authorities == [None]


def test_gateway_cache_authority_is_sensitive_to_the_final_state_and_policy_inputs() -> (
    None
):
    request, policy, plan, _ = direct_inputs(enabled=True)
    document = DocumentProvider(())
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=document,
        static_provider=StaticProvider(()),
    )
    first_state_ref = "csk1:" + "1" * 64
    second_state_ref = "csk1:" + "2" * 64

    _ = collect_gateway(
        gateway,
        request=request,
        plan=plan,
        policy=policy,
        state_key_ref=first_state_ref,
        document_cache_config=cache_config(),
    )
    _ = collect_gateway(
        gateway,
        request=request,
        plan=plan,
        policy=policy,
        state_key_ref=second_state_ref,
        document_cache_config=cache_config(),
    )

    first, second = document.cache_authorities
    assert first is not None and second is not None
    assert first.key_for_document_query(
        query="company public fact",
        document_ids=("company",),
        max_results=1,
    ) != second.key_for_document_query(
        query="company public fact",
        document_ids=("company",),
        max_results=1,
    )


def test_execute_v2_grounds_native_gateway_facts_without_a_legacy_evidence_conversion() -> (
    None
):
    request, policy, plan, state_key_ref = direct_inputs(enabled=True)
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=DocumentProvider(
            (GatewayDocumentContextV1(document_id="company", text="Company office"),)
        ),
        static_provider=StaticProvider(
            (
                GatewayStaticContextV1(
                    entry_id="company_shareholders",
                    manifest_ref=APPROVED_STATIC_MANIFEST_REF,
                    text="Company facts",
                ),
            )
        ),
    )
    executor = EvidenceExecutor(
        EmptyPreflight(),
        internal_company_knowledge_gateway=gateway,
    )

    async def run() -> CanonicalEvidenceExecutionResultV1:
        return await executor.execute_v2(
            request,
            plan,
            policy,
            scope_authority=business_scope_authority_v1(request.business_scope),
            state_key_ref=state_key_ref,
            document_cache_config=cache_config(),
        )

    result = anyio.run(run)

    assert {fact.source_type for fact in result.canonical_facts} == {
        "document_mcp",
        "approved_static_knowledge",
    }
    assert len(result.groundings[0].allowed_evidence) == 2
