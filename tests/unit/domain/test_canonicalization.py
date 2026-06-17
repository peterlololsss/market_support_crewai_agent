from __future__ import annotations

from market_support_crewai_agent.runtime.domain.entity_resolution import (
    CanonicalEntityResolver,
    DomainEntity,
    EntityCatalog,
)
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.ontology import (
    DistributionChannel,
    DomainContext,
    Product,
    Strategy,
)
from market_support_crewai_agent.schemas import ReplyRequest


def make_request(message: str, **overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": message,
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def test_canonicalize_does_not_resolve_numeric_strategy_substring_by_default():
    context = canonicalize_request(make_request("1000所有号的周报我想看看"))

    assert context.strategy_status == "unknown"
    assert context.selected_strategy is None
    assert context.entities == ()


def test_canonicalize_resolves_numeric_strategy_only_from_explicit_alias_catalog():
    channel = DistributionChannel(id="channel:current", name="test channel", kind="bank")
    strategy = Strategy(
        id="strategy:1000",
        name="中证1000",
        channel_id=channel.id,
        provenance="test",
    )
    resolver = CanonicalEntityResolver(
        base_catalog=EntityCatalog(
            (
                DomainEntity(
                    entity_id=strategy.id,
                    type="strategy",
                    canonical_name=strategy.name,
                    aliases=("1000",),
                    channel_id=channel.id,
                    provenance="test_alias_catalog",
                ),
            )
        )
    )
    context = canonicalize_request(
        make_request("1000所有号的周报我想看看"),
        domain_context=DomainContext(channel=channel, strategies=(strategy,)),
        resolver=resolver,
    )

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "中证1000"
    assert context.entities[0].source == "exact_alias"
    strategy_resolution = next(
        resolution
        for resolution in context.resolutions
        if resolution.type == "strategy"
    )
    assert strategy_resolution.confidence >= 0.72
    assert "exact_alias" in strategy_resolution.candidates[0].candidate_sources


def test_canonicalize_resolves_direct_catalog_strategy():
    context = canonicalize_request(
        make_request(
            "灵活对冲材料包发一下",
            available_strategies=["灵活对冲", "完全对冲"],
        )
    )

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "灵活对冲"
    assert context.entities[0].source == "exact_name"


def test_canonicalize_does_not_mark_numeric_strategy_mentions_ambiguous_by_default():
    context = canonicalize_request(make_request("500和1000周报都发一下"))

    assert context.strategy_status == "unknown"
    assert context.selected_strategy is None
    assert context.strategy_candidates == ("中证500", "中证1000")
    assert context.ambiguities == ()


def test_canonicalize_marks_generic_index_enhancement_request_ambiguous():
    context = canonicalize_request(
        make_request(
            "指增材料包发一下",
            available_strategies=["中证500指增", "中证1000指增"],
        )
    )

    assert context.strategy_status == "unknown"
    assert context.selected_strategy is None
    assert context.strategy_candidates == ("中证500指增", "中证1000指增")


def test_canonicalize_uses_single_available_strategy_as_default():
    context = canonicalize_request(
        make_request("材料包发一下", available_strategies=["指增"])
    )

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "指增"
    assert context.entities[0].raw_text == ""
    assert context.entities[0].source == "context_default"


def test_canonicalize_does_not_confuse_a500_with_500():
    context = canonicalize_request(
        make_request(
            "中证A500指增介绍一下",
            available_strategies=["中证500", "中证A500"],
        )
    )

    assert context.strategy_status == "resolved"
    assert context.selected_strategy == "中证A500"


def test_canonicalize_does_not_resolve_strategy_typo_without_explicit_alias():
    context = canonicalize_request(
        make_request(
            "介绍一下宗曾全子",
            available_strategies=["中证全指", "中证500"],
        )
    )

    assert context.strategy_status == "unknown"
    assert context.selected_strategy is None
    assert context.entities == ()


def test_ambiguous_product_names_across_strategies_require_scope():
    domain_context = make_product_domain_context(
        products=[
            ("product:a:alpha", "产品A", ("strategy:alpha",), "channel:current"),
            ("product:a:beta", "产品A", ("strategy:beta",), "channel:current"),
        ]
    )
    resolver = CanonicalEntityResolver()

    ambiguous = resolver.resolve_request(
        make_request(
            "产品A表现怎么样",
            available_strategies=["策略A", "策略B"],
        ),
        domain_context=domain_context,
    )
    product_resolution = ambiguous.by_type("product")[0]

    assert product_resolution.status == "ambiguous"
    assert [candidate.entity_id for candidate in product_resolution.candidates] == [
        "product:a:alpha",
        "product:a:beta",
    ]

    scoped = resolver.resolve_request(
        make_request(
            "策略A产品A表现怎么样",
            available_strategies=["策略A", "策略B"],
        ),
        domain_context=domain_context,
    )
    scoped_product_resolution = scoped.by_type("product")[0]

    assert scoped_product_resolution.status == "resolved"
    assert scoped_product_resolution.entity_id == "product:a:alpha"
    assert scoped_product_resolution.confidence > product_resolution.confidence


def test_channel_context_disambiguates_same_product_mention():
    domain_context = make_product_domain_context(
        products=[
            ("product:current", "产品A", (), "channel:current"),
            ("product:other", "产品A", (), "channel:other"),
        ]
    )

    result = CanonicalEntityResolver().resolve_request(
        make_request("产品A表现怎么样", available_strategies=[]),
        domain_context=domain_context,
    )
    product_resolution = result.by_type("product")[0]

    assert product_resolution.status == "resolved"
    assert product_resolution.entity_id == "product:current"


def test_unknown_strategy_mention_returns_unresolved_not_nearest_keyword():
    context = canonicalize_request(
        make_request(
            "中证900周报发一下",
            available_strategies=["中证500", "中证1000"],
        )
    )
    strategy_resolution = next(
        resolution
        for resolution in context.resolutions
        if resolution.type == "strategy"
    )

    assert context.strategy_status == "unknown"
    assert context.selected_strategy is None
    assert strategy_resolution.status == "unresolved"
    assert strategy_resolution.mention.raw_text == "中证900"
    assert context.resolution_metrics.low_confidence >= 1


def test_chinese_artifact_mentions_are_candidates_not_authoritative_resolutions():
    context = canonicalize_request(
        make_request(
            "材料包、周报、月报都看一下",
            available_strategies=[],
        )
    )
    artifact_resolutions = [
        resolution
        for resolution in context.resolutions
        if resolution.type == "artifact"
    ]

    assert [resolution.status for resolution in artifact_resolutions] == [
        "unresolved",
        "unresolved",
        "unresolved",
    ]
    assert all(
        "semantic_example" in resolution.candidates[0].candidate_sources
        for resolution in artifact_resolutions
    )
    assert context.resolution_metrics.semantic_candidate_used >= 3


def test_report_type_mentions_are_typed_separately_from_artifacts():
    context = canonicalize_request(
        make_request(
            "这份周报是什么时间段",
            available_strategies=[],
        )
    )
    report_type_resolutions = [
        resolution
        for resolution in context.resolutions
        if resolution.type == "report_type"
    ]

    assert [resolution.status for resolution in report_type_resolutions] == [
        "unresolved"
    ]
    assert "semantic_example" in report_type_resolutions[0].candidates[0].candidate_sources


def test_custom_ontology_exact_alias_resolves_without_keyword_fallbacks():
    channel = DistributionChannel(
        id="channel:current",
        name="test channel",
        kind="bank",
    )
    domain_context = DomainContext(channel=channel)
    resolver = CanonicalEntityResolver(
        base_catalog=EntityCatalog(
            (
                DomainEntity(
                    entity_id="artifact:combined_material",
                    type="artifact",
                    canonical_name="combined_material",
                    aliases=("组合材料",),
                    examples=(),
                    description="A bundled client-facing material artifact.",
                    artifact_type="material_pack",
                    provenance="test_ontology",
                ),
            )
        )
    )

    result = resolver.resolve_request(
        make_request(
            "请发组合材料",
            available_materials=[],
            available_strategies=[],
        ),
        domain_context=domain_context,
    )
    artifact_resolution = next(
        resolution
        for resolution in result.resolutions
        if resolution.entity_id == "artifact:combined_material"
    )

    assert artifact_resolution.status == "resolved"
    assert artifact_resolution.candidates[0].candidate_sources == ("exact_alias",)


def test_semantic_description_does_not_resolve_without_structured_alias():
    channel = DistributionChannel(
        id="channel:current",
        name="test channel",
        kind="bank",
    )
    domain_context = DomainContext(channel=channel)
    resolver = CanonicalEntityResolver(
        base_catalog=EntityCatalog(
            (
                DomainEntity(
                    entity_id="artifact:client_deck",
                    type="artifact",
                    canonical_name="client_deck",
                    aliases=(),
                    examples=(),
                    description="Client facing product material pack deck.",
                    artifact_type="material_pack",
                    provenance="test_ontology",
                ),
            )
        )
    )

    result = resolver.resolve_request(
        make_request(
            "please send the product material pack",
            available_materials=[],
            available_strategies=[],
        ),
        domain_context=domain_context,
    )
    assert not any(
        resolution.entity_id == "artifact:client_deck"
        for resolution in result.resolutions
    )


def make_product_domain_context(
    *,
    products: list[tuple[str, str, tuple[str, ...], str]],
) -> DomainContext:
    channel = DistributionChannel(
        id="channel:current",
        name="test channel",
        kind="bank",
        provenance="test",
    )
    strategies = (
        Strategy(
            id="strategy:alpha",
            name="策略A",
            channel_id="channel:current",
            provenance="test",
        ),
        Strategy(
            id="strategy:beta",
            name="策略B",
            channel_id="channel:current",
            provenance="test",
        ),
    )
    return DomainContext(
        channel=channel,
        strategies=strategies,
        products=tuple(
            Product(
                id=product_id,
                name=name,
                channel_id=channel_id,
                strategy_ids=strategy_ids,
                provenance="test",
            )
            for product_id, name, strategy_ids, channel_id in products
        ),
    )
