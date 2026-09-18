from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    ApprovedStaticKnowledgeGatewayAdapter,
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
)
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.recall.approved_static_catalog import (
    APPROVED_STATIC_MANIFEST_REF,
)
from tests.unit.recall._direct_media_policy_fixtures import (
    DirectImageBudgetSelector,
    DocumentProvider,
    StaticProvider,
    direct_media_inputs,
    document_mcp_cache_config,
)


@pytest.mark.anyio
async def test_disabled_direct_policy_skips_prepopulated_provider_caches_and_media() -> (
    None
):
    # Given: disabled unified authority and providers armed with prepopulated caches.
    inputs = direct_media_inputs(enabled=False)
    document = DocumentProvider(
        (GatewayDocumentContextV1(document_id="company", text="must not escape"),)
    )
    static = StaticProvider()
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=document,
        static_provider=static,
    )

    # When: the post-plan gateway is called with an otherwise selected plan.
    result = await gateway.collect(
        request=inputs.request,
        plan=inputs.plan,
        policy=inputs.policy,
        state_key_ref=inputs.state_key_ref,
        document_cache_config=document_mcp_cache_config(),
    )

    # Then: the unified bit returns before selector, cache, network, or media work.
    assert (document.calls, static.calls) == (0, 0)
    assert document.cache == {"warm": "document"}
    assert static.cache == {"warm": "static"}
    assert result.facts == ()
    assert result.media_bindings == ()


@pytest.mark.anyio
async def test_direct_forbidden_image_marker_is_rejected_before_fact_admission() -> (
    None
):
    # Given: a validated direct knowledge plan and exact entry/ref/asset selection.
    inputs = direct_media_inputs(enabled=True)
    static = StaticProvider(
        (
            GatewayStaticContextV1(
                entry_id="company_shareholders",
                manifest_ref=APPROVED_STATIC_MANIFEST_REF,
                text="%%company_shareholders.png%%",
                selected_asset_ids=("company_shareholders_chart",),
            ),
        )
    )
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=DocumentProvider(),
        static_provider=static,
    )

    # When: both eligible providers run after plan validation.
    result = await gateway.collect(
        request=inputs.request,
        plan=inputs.plan,
        policy=inputs.policy,
        state_key_ref=inputs.state_key_ref,
        document_cache_config=document_mcp_cache_config(),
    )

    # Then: direct static media is rejected even when catalog-valid and evidence-bound.
    assert result.facts == ()
    assert result.media_bindings == ()


@pytest.mark.anyio
async def test_direct_static_selector_receives_zero_image_budget() -> None:
    inputs = direct_media_inputs(enabled=True)
    selector = DirectImageBudgetSelector()
    provider = ApprovedStaticKnowledgeGatewayAdapter(selector=selector)

    result = await provider.collect(
        request=inputs.request,
        evidence_query="company shareholding",
    )

    assert result == ()
    assert selector.max_images == [0]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "manifest_ref",
    (
        None,
        ManifestRefV1(manifest_id="general.smalltalk", manifest_version="2026-07-18.1"),
        ManifestRefV1.model_construct(
            manifest_id="answer_internal_company_knowledge",
            manifest_version="2026-07-15.1",
        ),
    ),
    ids=("missing", "unselected", "invalid-version"),
)
async def test_static_result_rejects_missing_wrong_or_unselected_manifest_ref(
    manifest_ref: ManifestRefV1 | None,
) -> None:
    # Given: an enabled selected plan but a static row without its exact authority ref.
    inputs = direct_media_inputs(enabled=True)
    context = (
        GatewayStaticContextV1.model_construct(
            entry_id="company_shareholders",
            text="Approved company facts",
            selected_asset_ids=(),
        )
        if manifest_ref is None
        else GatewayStaticContextV1.model_construct(
            entry_id="company_shareholders",
            manifest_ref=manifest_ref,
            text="Approved company facts",
            selected_asset_ids=(),
        )
    )
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=DocumentProvider(),
        static_provider=StaticProvider((context,)),
    )

    # When: the malformed provider row crosses deterministic admission.
    result = await gateway.collect(
        request=inputs.request,
        plan=inputs.plan,
        policy=inputs.policy,
        state_key_ref=inputs.state_key_ref,
        document_cache_config=document_mcp_cache_config(),
    )

    # Then: no fact or media is inferred from entry topic or plan proximity.
    assert result.facts == ()
    assert result.media_bindings == ()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "selected_asset_ids"),
    (
        ("%%arbitrary.png%%", ()),
        ("%%company_shareholders.png%%", ()),
        ("%%company_shareholders.png%", ("company_shareholders_chart",)),
    ),
    ids=("unknown", "unselected", "malformed"),
)
async def test_static_result_rejects_unknown_unselected_or_malformed_marker(
    text: str,
    selected_asset_ids: tuple[str, ...],
) -> None:
    # Given: an exact manifest-bound entry with a marker outside exact selection.
    inputs = direct_media_inputs(enabled=True)
    context = GatewayStaticContextV1(
        entry_id="company_shareholders",
        manifest_ref=APPROVED_STATIC_MANIFEST_REF,
        text=text,
        selected_asset_ids=selected_asset_ids,
    )
    gateway = InternalCompanyKnowledgeGatewayV1(
        document_provider=DocumentProvider(),
        static_provider=StaticProvider((context,)),
    )

    # When: the row crosses marker admission.
    result = await gateway.collect(
        request=inputs.request,
        plan=inputs.plan,
        policy=inputs.policy,
        state_key_ref=inputs.state_key_ref,
        document_cache_config=document_mcp_cache_config(),
    )

    # Then: the entire static fact and all media bindings are rejected.
    assert result.facts == ()
    assert result.media_bindings == ()
