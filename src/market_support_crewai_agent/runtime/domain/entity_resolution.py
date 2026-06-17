from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from market_support_crewai_agent.runtime.domain.ontology import (
    Artifact,
    DistributionChannel,
    DomainContext,
    Product,
    Strategy,
)
from market_support_crewai_agent.schemas import ReplyRequest

EntityType = Literal[
    "channel",
    "strategy",
    "product",
    "artifact",
    "time_period",
    "metric",
    "report_type",
]
MentionSource = Literal["request", "evidence"]
MentionExtractionSource = Literal[
    "canonical_name",
    "exact_alias",
    "semantic_example",
    "semantic_description",
    "structured_parser",
    "context_default",
]
CandidateSource = Literal[
    "entity_id",
    "exact_name",
    "exact_alias",
    "semantic_example",
    "semantic_description",
    "semantic_provider",
    "context_default",
    "llm_disambiguator",
]
ResolutionStatus = Literal["resolved", "unresolved", "ambiguous"]
ResolverStage = Literal[
    "mention_extraction",
    "candidate_generation",
    "scoring",
    "disambiguation",
    "abstention",
]

_LOGGER = logging.getLogger(__name__)
_RESOLUTION_THRESHOLD = 0.72
_AMBIGUITY_MARGIN = 0.08
_TYPE_ORDER: tuple[EntityType, ...] = (
    "channel",
    "strategy",
    "artifact",
    "report_type",
    "time_period",
    "metric",
    "product",
)


@dataclass(frozen=True)
class EvidenceSnippet:
    text: str
    source_id: str = ""
    artifact_id: str | None = None
    artifact_type: str | None = None
    channel_id: str | None = None
    strategy_id: str | None = None
    product_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DomainEntity:
    entity_id: str
    type: EntityType
    canonical_name: str
    aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    description: str = ""
    channel_id: str | None = None
    strategy_ids: tuple[str, ...] = ()
    artifact_type: str | None = None
    source_id: str = ""
    provenance: str = "ontology"
    metadata: dict[str, object] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "type": self.type,
            "canonical_name": self.canonical_name,
            "channel_id": self.channel_id,
            "strategy_ids": list(self.strategy_ids),
            "artifact_type": self.artifact_type,
            "source_id": self.source_id,
            "provenance": self.provenance,
            "description": self.description,
            "examples": list(self.examples[:5]),
        }


@dataclass(frozen=True)
class EntityMention:
    mention_id: str
    type: EntityType
    raw_text: str
    source: MentionSource = "request"
    source_id: str = ""
    span: tuple[int, int] | None = None
    extraction_source: MentionExtractionSource = "structured_parser"
    evidence: dict[str, object] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict:
        return {
            "mention_id": self.mention_id,
            "type": self.type,
            "raw_text": self.raw_text,
            "source": self.source,
            "source_id": self.source_id,
            "span": list(self.span) if self.span is not None else None,
            "extraction_source": self.extraction_source,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class ScoreFeature:
    name: str
    value: float
    reason: str

    def to_prompt_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EntityCandidateScore:
    entity_id: str
    type: EntityType
    canonical_name: str
    confidence: float
    candidate_sources: tuple[CandidateSource, ...]
    features: tuple[ScoreFeature, ...] = ()
    rationale: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "type": self.type,
            "canonical_name": self.canonical_name,
            "confidence": round(self.confidence, 4),
            "candidate_sources": list(self.candidate_sources),
            "features": [feature.to_prompt_dict() for feature in self.features],
            "rationale": self.rationale,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class EntityResolution:
    status: ResolutionStatus
    type: EntityType
    mention: EntityMention
    entity_id: str | None = None
    canonical_name: str | None = None
    confidence: float = 0.0
    candidates: tuple[EntityCandidateScore, ...] = ()
    resolver_stage: ResolverStage = "abstention"
    rationale: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def ambiguous(self) -> bool:
        return self.status == "ambiguous"

    def to_prompt_dict(self) -> dict:
        return {
            "status": self.status,
            "type": self.type,
            "mention": self.mention.to_prompt_dict(),
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "confidence": round(self.confidence, 4),
            "resolver_stage": self.resolver_stage,
            "rationale": self.rationale,
            "evidence": dict(self.evidence),
            "candidates": [
                candidate.to_prompt_dict()
                for candidate in self.candidates[:5]
            ],
        }


@dataclass(frozen=True)
class EntityResolutionMetrics:
    resolved: int = 0
    unresolved: int = 0
    ambiguous: int = 0
    low_confidence: int = 0
    exact_alias_used: int = 0
    semantic_candidate_used: int = 0

    def to_prompt_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "unresolved": self.unresolved,
            "ambiguous": self.ambiguous,
            "low_confidence": self.low_confidence,
            "exact_alias_used": self.exact_alias_used,
            "semantic_candidate_used": self.semantic_candidate_used,
        }


@dataclass(frozen=True)
class EntityResolutionResult:
    resolutions: tuple[EntityResolution, ...]
    metrics: EntityResolutionMetrics
    mentions: tuple[EntityMention, ...] = ()

    def by_type(self, entity_type: EntityType) -> tuple[EntityResolution, ...]:
        return tuple(
            resolution
            for resolution in self.resolutions
            if resolution.type == entity_type
        )

    def to_prompt_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_prompt_dict(),
            "mentions": [
                mention.to_prompt_dict()
                for mention in self.mentions
            ],
            "resolutions": [
                resolution.to_prompt_dict()
                for resolution in self.resolutions
            ],
        }


@dataclass(frozen=True)
class EntityCatalog:
    entities: tuple[DomainEntity, ...]

    def by_type(self, entity_type: EntityType) -> tuple[DomainEntity, ...]:
        return tuple(entity for entity in self.entities if entity.type == entity_type)

    def with_entities(self, entities: Sequence[DomainEntity]) -> EntityCatalog:
        return EntityCatalog(tuple(_unique_entities((*self.entities, *entities))))

    @classmethod
    def from_domain_context(cls, domain_context: DomainContext) -> EntityCatalog:
        entities: list[DomainEntity] = [_channel_entity(domain_context.channel)]
        entities.extend(
            _strategy_entity(strategy)
            for strategy in domain_context.strategies
        )
        entities.extend(
            _product_entity(product)
            for product in domain_context.products
        )
        entities.extend(
            _artifact_entity(artifact)
            for artifact in domain_context.artifacts
        )
        return EntityCatalog(tuple(_unique_entities(entities)))


@dataclass(frozen=True)
class EntityDisambiguationDecision:
    status: ResolutionStatus
    entity_id: str | None = None
    confidence: float = 0.0
    rationale: str = ""


class SemanticCandidateProvider(Protocol):
    def candidates_for(
        self,
        mention: EntityMention,
        catalog: EntityCatalog,
    ) -> tuple[EntityCandidateScore, ...]:
        ...


class EntityDisambiguator(Protocol):
    def disambiguate(
        self,
        mention: EntityMention,
        candidates: tuple[EntityCandidateScore, ...],
    ) -> EntityDisambiguationDecision:
        ...


@dataclass(frozen=True)
class _RawCandidate:
    entity: DomainEntity
    candidate_sources: tuple[CandidateSource, ...]
    base_score: float
    evidence: dict[str, object]


@dataclass(frozen=True)
class _ResolutionContext:
    request: ReplyRequest
    domain_context: DomainContext
    prior_resolutions: tuple[EntityResolution, ...] = ()


class DefaultMentionExtractor:
    def extract(
        self,
        request: ReplyRequest,
        catalog: EntityCatalog,
        evidence_snippets: Sequence[EvidenceSnippet] = (),
    ) -> tuple[EntityMention, ...]:
        mentions: list[EntityMention] = []
        request_text = _user_visible_text(request.message)
        mentions.extend(
            self._extract_from_text(
                request_text,
                source="request",
                source_id=request.context_id or "",
                catalog=catalog,
                evidence={},
            )
        )
        for snippet in evidence_snippets:
            evidence = {
                "artifact_id": snippet.artifact_id,
                "artifact_type": snippet.artifact_type,
                "channel_id": snippet.channel_id,
                "strategy_id": snippet.strategy_id,
                "product_ids": list(snippet.product_ids),
            }
            mentions.extend(
                self._extract_from_text(
                    snippet.text,
                    source="evidence",
                    source_id=snippet.source_id,
                    catalog=catalog,
                    evidence={
                        key: value
                        for key, value in evidence.items()
                        if value not in (None, "", [])
                    },
                )
            )
        return tuple(_dedupe_mentions(mentions))

    def _extract_from_text(
        self,
        text: str,
        *,
        source: MentionSource,
        source_id: str,
        catalog: EntityCatalog,
        evidence: dict[str, object],
    ) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        mention_keys: set[tuple[EntityType, tuple[int, int], str]] = set()
        for entity in catalog.entities:
            for phrase, extraction_source in _entity_phrases(entity):
                for span, raw_text in _find_phrase_occurrences(text, phrase):
                    key = (entity.type, span, _normalize(raw_text))
                    if key in mention_keys:
                        continue
                    mention_keys.add(key)
                    mentions.append(
                        EntityMention(
                            mention_id=_mention_id(
                                source,
                                source_id,
                                entity.type,
                                raw_text,
                                span,
                            ),
                            type=entity.type,
                            raw_text=raw_text,
                            source=source,
                            source_id=source_id,
                            span=span,
                            extraction_source=extraction_source,
                            evidence=evidence,
                        )
                    )
        if len(catalog.by_type("strategy")) > 1:
            mentions.extend(
                _structured_unknown_strategy_mentions(
                    text,
                    source=source,
                    source_id=source_id,
                    evidence=evidence,
                    existing=mentions,
                )
            )
        return mentions


class DefaultCandidateGenerator:
    def __init__(
        self,
        semantic_candidate_provider: SemanticCandidateProvider | None = None,
    ) -> None:
        self.semantic_candidate_provider = semantic_candidate_provider

    def generate(
        self,
        mention: EntityMention,
        catalog: EntityCatalog,
    ) -> tuple[_RawCandidate, ...]:
        raw_candidates: list[_RawCandidate] = []
        if mention.extraction_source == "context_default":
            strategy_id = str(mention.evidence.get("strategy_id") or "")
            if strategy_id:
                return tuple(
                    _RawCandidate(
                        entity=entity,
                        candidate_sources=("context_default",),
                        base_score=0.78,
                        evidence={
                            "candidate_source": ["context_default"],
                            "entity_source_id": entity.source_id,
                            "entity_provenance": entity.provenance,
                        },
                    )
                    for entity in catalog.by_type(mention.type)
                    if entity.entity_id == strategy_id
                )
        mention_norm = _normalize(mention.raw_text)
        for entity in catalog.by_type(mention.type):
            sources, score = _candidate_sources_and_score(mention.raw_text, entity)
            if sources:
                raw_candidates.append(
                    _RawCandidate(
                        entity=entity,
                        candidate_sources=sources,
                        base_score=score,
                        evidence={
                            "candidate_source": list(sources),
                            "entity_source_id": entity.source_id,
                            "entity_provenance": entity.provenance,
                        },
                    )
                )
                continue
        return tuple(_dedupe_raw_candidates(raw_candidates))

    def semantic_candidates(
        self,
        mention: EntityMention,
        catalog: EntityCatalog,
    ) -> tuple[EntityCandidateScore, ...]:
        if self.semantic_candidate_provider is None:
            return ()
        return self.semantic_candidate_provider.candidates_for(mention, catalog)


class CandidateScorer:
    def score(
        self,
        candidate: _RawCandidate,
        mention: EntityMention,
        context: _ResolutionContext,
    ) -> EntityCandidateScore:
        features = [
            ScoreFeature(
                name="candidate_source",
                value=candidate.base_score,
                reason="best candidate-generation source score",
            )
        ]
        confidence = candidate.base_score

        channel_boost = _channel_context_boost(candidate.entity, context.domain_context)
        if channel_boost:
            confidence += channel_boost
            features.append(
                ScoreFeature(
                    name="channel_context",
                    value=channel_boost,
                    reason="candidate belongs to current distribution channel",
                )
            )

        strategy_boost = _strategy_context_boost(candidate.entity, context)
        if strategy_boost:
            confidence += strategy_boost
            features.append(
                ScoreFeature(
                    name="strategy_context",
                    value=strategy_boost,
                    reason="candidate relationship matches resolved strategy context",
                )
            )

        artifact_boost = _artifact_scope_boost(candidate.entity, mention)
        if artifact_boost:
            confidence += artifact_boost
            features.append(
                ScoreFeature(
                    name="artifact_scope",
                    value=artifact_boost,
                    reason="candidate matches evidence artifact scope",
                )
            )

        confidence = max(0.0, min(confidence, 1.0))
        return EntityCandidateScore(
            entity_id=candidate.entity.entity_id,
            type=candidate.entity.type,
            canonical_name=candidate.entity.canonical_name,
            confidence=confidence,
            candidate_sources=candidate.candidate_sources,
            features=tuple(features),
            rationale=_candidate_rationale(candidate.candidate_sources),
            evidence={
                **candidate.evidence,
                "mention_id": mention.mention_id,
                "mention_source": mention.source,
                "mention_source_id": mention.source_id,
            },
        )


class CanonicalEntityResolver:
    def __init__(
        self,
        *,
        mention_extractor: DefaultMentionExtractor | None = None,
        candidate_generator: DefaultCandidateGenerator | None = None,
        scorer: CandidateScorer | None = None,
        disambiguator: EntityDisambiguator | None = None,
        base_catalog: EntityCatalog | None = None,
        resolution_threshold: float = _RESOLUTION_THRESHOLD,
        ambiguity_margin: float = _AMBIGUITY_MARGIN,
    ) -> None:
        self.mention_extractor = mention_extractor or DefaultMentionExtractor()
        self.candidate_generator = candidate_generator or DefaultCandidateGenerator()
        self.scorer = scorer or CandidateScorer()
        self.disambiguator = disambiguator
        self.base_catalog = base_catalog or EntityCatalog(_BUILTIN_ONTOLOGY_ENTITIES)
        self.resolution_threshold = resolution_threshold
        self.ambiguity_margin = ambiguity_margin

    def resolve_request(
        self,
        request: ReplyRequest,
        *,
        domain_context: DomainContext | None = None,
        evidence_snippets: Sequence[EvidenceSnippet] = (),
    ) -> EntityResolutionResult:
        if domain_context is None:
            from market_support_crewai_agent.runtime.domain.ontology import (
                DomainContextBuilder,
            )

            domain_context = DomainContextBuilder().build(request)
        catalog = self.base_catalog.with_entities(
            EntityCatalog.from_domain_context(domain_context).entities
        )
        mentions = list(
            self.mention_extractor.extract(
                request,
                catalog,
                evidence_snippets=evidence_snippets,
            )
        )
        if not any(mention.type == "strategy" for mention in mentions):
            mentions.extend(_context_default_strategy_mentions(request, domain_context))

        resolutions: list[EntityResolution] = []
        for entity_type in _TYPE_ORDER:
            for mention in [item for item in mentions if item.type == entity_type]:
                resolution = self._resolve_mention(
                    mention,
                    catalog,
                    _ResolutionContext(
                        request=request,
                        domain_context=domain_context,
                        prior_resolutions=tuple(resolutions),
                    ),
                )
                resolutions.append(resolution)

        result = EntityResolutionResult(
            resolutions=tuple(resolutions),
            metrics=_metrics_for(resolutions),
            mentions=tuple(mentions),
        )
        _LOGGER.debug(
            "canonical_entity_resolution metrics=%s",
            result.metrics.to_prompt_dict(),
        )
        return result

    def _resolve_mention(
        self,
        mention: EntityMention,
        catalog: EntityCatalog,
        context: _ResolutionContext,
    ) -> EntityResolution:
        scored = [
            self.scorer.score(candidate, mention, context)
            for candidate in self.candidate_generator.generate(mention, catalog)
        ]
        scored.extend(self.candidate_generator.semantic_candidates(mention, catalog))
        scored = list(_dedupe_scored_candidates(scored))
        scored.sort(key=lambda candidate: candidate.confidence, reverse=True)

        if self.disambiguator is not None and len(scored) > 1:
            decision = self.disambiguator.disambiguate(mention, tuple(scored))
            resolved = _resolution_from_disambiguator(mention, scored, decision)
            if resolved is not None:
                return resolved

        if not scored:
            return EntityResolution(
                status="unresolved",
                type=mention.type,
                mention=mention,
                resolver_stage="abstention",
                rationale="no candidates generated for typed mention",
                evidence={
                    "mention_id": mention.mention_id,
                    "stage": "candidate_generation",
                },
            )

        top = scored[0]
        if top.confidence < self.resolution_threshold:
            return EntityResolution(
                status="unresolved",
                type=mention.type,
                mention=mention,
                confidence=top.confidence,
                candidates=tuple(scored),
                resolver_stage="abstention",
                rationale="top candidate did not pass confidence threshold",
                evidence={
                    "mention_id": mention.mention_id,
                    "threshold": self.resolution_threshold,
                    "top_candidate_id": top.entity_id,
                },
            )

        if len(scored) > 1 and top.confidence - scored[1].confidence <= self.ambiguity_margin:
            return EntityResolution(
                status="ambiguous",
                type=mention.type,
                mention=mention,
                confidence=top.confidence,
                candidates=tuple(scored),
                resolver_stage="abstention",
                rationale="top candidates are too close to choose safely",
                evidence={
                    "mention_id": mention.mention_id,
                    "ambiguity_margin": self.ambiguity_margin,
                },
            )

        return EntityResolution(
            status="resolved",
            type=mention.type,
            mention=mention,
            entity_id=top.entity_id,
            canonical_name=top.canonical_name,
            confidence=top.confidence,
            candidates=tuple(scored),
            resolver_stage="disambiguation",
            rationale=top.rationale,
            evidence={
                "mention_id": mention.mention_id,
                "candidate_id": top.entity_id,
                "candidate_sources": list(top.candidate_sources),
            },
        )


DefaultCanonicalEntityResolver = CanonicalEntityResolver


_BUILTIN_ONTOLOGY_ENTITIES: tuple[DomainEntity, ...] = (
    DomainEntity(
        entity_id="artifact:material_pack",
        type="artifact",
        canonical_name="material_pack",
        examples=(
            "材料包",
            "产品资料",
            "宣传材料",
            "推介材料",
            "对客材料",
            "PPT",
            "一页通",
            "一夜通",
            "一夜痛",
            "要素表",
            "开放日历",
            "排期表",
            "销售日期表",
            "推荐资料",
            "培训视频",
        ),
        description="Official product material pack, decks, one-pagers, factsheets, calendars, and sales material.",
        artifact_type="material_pack",
        provenance="built_in_ontology",
    ),
    DomainEntity(
        entity_id="artifact:weekly_report",
        type="artifact",
        canonical_name="weekly_report",
        examples=("周报", "本周报告", "weekly report", "weekly"),
        description="Weekly performance report or recently updated weekly report artifact.",
        artifact_type="weekly_report",
        provenance="built_in_ontology",
    ),
    DomainEntity(
        entity_id="artifact:monthly_report",
        type="artifact",
        canonical_name="monthly_report",
        examples=("月报", "月度报告", "monthly report", "monthly"),
        description="Monthly performance report artifact.",
        artifact_type="monthly_report",
        provenance="built_in_ontology",
    ),
    DomainEntity(
        entity_id="report_type:weekly_report",
        type="report_type",
        canonical_name="weekly_report",
        examples=("周报", "本周报告", "weekly report", "weekly"),
        description="Weekly report request, weekly performance report, or latest weekly report type.",
        artifact_type="weekly_report",
        provenance="built_in_ontology",
    ),
    DomainEntity(
        entity_id="report_type:monthly_report",
        type="report_type",
        canonical_name="monthly_report",
        examples=("月报", "月度报告", "monthly report", "monthly"),
        description="Monthly report request, monthly performance report, or latest monthly report type.",
        artifact_type="monthly_report",
        provenance="built_in_ontology",
    ),
    DomainEntity(
        entity_id="time_period:current_week",
        type="time_period",
        canonical_name="current_week",
        examples=("本周", "这周", "最近一周"),
        description="Current or most recent week.",
        provenance="built_in_ontology",
    ),
    DomainEntity(
        entity_id="time_period:current_month",
        type="time_period",
        canonical_name="current_month",
        examples=("本月", "这个月", "最近一个月"),
        description="Current or most recent month.",
        provenance="built_in_ontology",
    ),
    DomainEntity(
        entity_id="metric:excess_return",
        type="metric",
        canonical_name="excess_return",
        examples=("超额", "超额收益", "相对收益"),
        description="Excess or relative return metric.",
        provenance="built_in_ontology",
    ),
    DomainEntity(
        entity_id="metric:max_drawdown",
        type="metric",
        canonical_name="max_drawdown",
        examples=("最大回撤", "回撤"),
        description="Maximum drawdown metric.",
        provenance="built_in_ontology",
    ),
)


def _channel_entity(channel: DistributionChannel) -> DomainEntity:
    return DomainEntity(
        entity_id=channel.id,
        type="channel",
        canonical_name=channel.name,
        aliases=(channel.name,),
        channel_id=channel.id,
        source_id=channel.source_id,
        provenance=channel.provenance,
    )


def _strategy_entity(strategy: Strategy) -> DomainEntity:
    return DomainEntity(
        entity_id=strategy.id,
        type="strategy",
        canonical_name=strategy.name,
        aliases=(strategy.name,),
        channel_id=strategy.channel_id,
        source_id=strategy.source_id,
        provenance=strategy.provenance,
    )


def _product_entity(product: Product) -> DomainEntity:
    return DomainEntity(
        entity_id=product.id,
        type="product",
        canonical_name=product.name,
        aliases=(product.name,),
        channel_id=product.channel_id,
        strategy_ids=product.strategy_ids,
        source_id=product.source_id,
        provenance=product.provenance,
    )


def _artifact_entity(artifact: Artifact) -> DomainEntity:
    examples = (artifact.title,) if artifact.title else ()
    return DomainEntity(
        entity_id=artifact.id,
        type="artifact",
        canonical_name=artifact.artifact_type,
        aliases=(),
        examples=examples,
        channel_id=artifact.scope.channel_id,
        strategy_ids=(artifact.scope.strategy_id,) if artifact.scope.strategy_id else (),
        artifact_type=artifact.artifact_type,
        source_id=artifact.scope.source_id,
        provenance=artifact.scope.provenance,
        metadata={
            "source_type": artifact.source_type,
            "fact_types": list(artifact.fact_types),
            "product_ids": list(artifact.scope.product_ids),
        },
    )


def _entity_phrases(
    entity: DomainEntity,
) -> tuple[tuple[str, MentionExtractionSource], ...]:
    phrases: list[tuple[str, MentionExtractionSource]] = []
    if entity.canonical_name:
        phrases.append((entity.canonical_name, "canonical_name"))
    phrases.extend((alias, "exact_alias") for alias in entity.aliases)
    phrases.extend((example, "semantic_example") for example in entity.examples)
    return tuple(
        (phrase, source)
        for phrase, source in _unique_phrase_sources(phrases)
        if phrase
    )


def _candidate_sources_and_score(
    mention_text: str,
    entity: DomainEntity,
) -> tuple[tuple[CandidateSource, ...], float]:
    mention_norm = _normalize(mention_text)
    if not mention_norm:
        return (), 0.0
    sources: list[CandidateSource] = []
    scores: list[float] = []
    if mention_norm == _normalize(entity.entity_id):
        sources.append("entity_id")
        scores.append(0.99)
    if mention_norm == _normalize(entity.canonical_name):
        sources.append("exact_name")
        scores.append(0.94)
    if any(mention_norm == _normalize(alias) for alias in entity.aliases):
        sources.append("exact_alias")
        scores.append(0.91)
    semantic_example_match = any(
        mention_norm == _normalize(example) for example in entity.examples
    )
    if semantic_example_match:
        sources.append("semantic_example")
        scores.append(0.55)
    return tuple(_unique_sources(sources)), max(scores or [0.0])


def _channel_context_boost(entity: DomainEntity, domain_context: DomainContext) -> float:
    if not entity.channel_id:
        return 0.0
    if entity.channel_id == domain_context.channel.id:
        return 0.04
    return -0.12


def _strategy_context_boost(
    entity: DomainEntity,
    context: _ResolutionContext,
) -> float:
    strategy_ids = {
        resolution.entity_id
        for resolution in context.prior_resolutions
        if resolution.type == "strategy" and resolution.status == "resolved" and resolution.entity_id
    }
    if not strategy_ids:
        return 0.0
    if entity.type == "strategy":
        return 0.10 if entity.entity_id in strategy_ids else -0.05
    if entity.type != "product" or not entity.strategy_ids:
        return 0.0
    if strategy_ids.intersection(entity.strategy_ids):
        return 0.12
    return -0.08


def _artifact_scope_boost(entity: DomainEntity, mention: EntityMention) -> float:
    artifact_type = str(mention.evidence.get("artifact_type") or "")
    if not artifact_type or entity.artifact_type != artifact_type:
        return 0.0
    return 0.10


def _candidate_rationale(sources: tuple[CandidateSource, ...]) -> str:
    if "context_default" in sources:
        return "single available strategy selected from request context"
    if "exact_alias" in sources:
        return "exact alias generated a candidate, then context scoring selected it"
    if any(source.startswith("semantic") for source in sources):
        return "semantic provider or ontology examples generated a non-authoritative candidate"
    if "exact_name" in sources:
        return "canonical entity name generated a candidate, then context scoring selected it"
    return "candidate selected by resolver scoring"


def _resolution_from_disambiguator(
    mention: EntityMention,
    candidates: list[EntityCandidateScore],
    decision: EntityDisambiguationDecision,
) -> EntityResolution | None:
    if decision.status != "resolved" or not decision.entity_id:
        return None
    selected = next(
        (candidate for candidate in candidates if candidate.entity_id == decision.entity_id),
        None,
    )
    if selected is None:
        return None
    confidence = min(1.0, max(selected.confidence, decision.confidence))
    return EntityResolution(
        status="resolved",
        type=mention.type,
        mention=mention,
        entity_id=selected.entity_id,
        canonical_name=selected.canonical_name,
        confidence=confidence,
        candidates=tuple(candidates),
        resolver_stage="disambiguation",
        rationale=decision.rationale or "closed-set LLM disambiguator selected candidate",
        evidence={
            "mention_id": mention.mention_id,
            "candidate_id": selected.entity_id,
            "candidate_sources": [
                *selected.candidate_sources,
                "llm_disambiguator",
            ],
        },
    )


def _metrics_for(resolutions: Sequence[EntityResolution]) -> EntityResolutionMetrics:
    counts = Counter(resolution.status for resolution in resolutions)
    low_confidence = sum(
        1
        for resolution in resolutions
        if resolution.status == "unresolved" and bool(resolution.candidates)
    )
    exact_alias_used = sum(
        1
        for resolution in resolutions
        if any(
            "exact_alias" in candidate.candidate_sources
            for candidate in resolution.candidates
        )
    )
    semantic_candidate_used = sum(
        1
        for resolution in resolutions
        if any(
            any(source in candidate.candidate_sources for source in _SEMANTIC_SOURCES)
            for candidate in resolution.candidates
        )
    )
    return EntityResolutionMetrics(
        resolved=counts["resolved"],
        unresolved=counts["unresolved"],
        ambiguous=counts["ambiguous"],
        low_confidence=low_confidence,
        exact_alias_used=exact_alias_used,
        semantic_candidate_used=semantic_candidate_used,
    )


_SEMANTIC_SOURCES = {
    "semantic_example",
    "semantic_description",
    "semantic_provider",
}


def _context_default_strategy_mentions(
    request: ReplyRequest,
    domain_context: DomainContext,
) -> tuple[EntityMention, ...]:
    if len(domain_context.strategies) != 1:
        return ()
    strategy = domain_context.strategies[0]
    return (
        EntityMention(
            mention_id=_mention_id(
                "request",
                request.context_id or "",
                "strategy",
                "",
                None,
            ),
            type="strategy",
            raw_text="",
            source="request",
            source_id=request.context_id or "",
            span=None,
            extraction_source="context_default",
            evidence={"strategy_id": strategy.id},
        ),
    )


def _dedupe_mentions(mentions: Sequence[EntityMention]) -> list[EntityMention]:
    ordered = sorted(
        mentions,
        key=lambda mention: (
            mention.source,
            mention.source_id,
            mention.type,
            mention.span[0] if mention.span else -1,
            -(mention.span[1] - mention.span[0]) if mention.span else 0,
        ),
    )
    selected: list[EntityMention] = []
    seen: set[tuple[str, str, EntityType, str, tuple[int, int] | None]] = set()
    for mention in ordered:
        key = (
            mention.source,
            mention.source_id,
            mention.type,
            _normalize(mention.raw_text),
            mention.span,
        )
        if key in seen:
            continue
        if _overlaps_selected(mention, selected):
            continue
        seen.add(key)
        selected.append(mention)
    selected.sort(
        key=lambda mention: (
            mention.source != "request",
            mention.span[0] if mention.span else 10_000,
            _TYPE_ORDER.index(mention.type) if mention.type in _TYPE_ORDER else 99,
        )
    )
    return selected


def _overlaps_selected(mention: EntityMention, selected: Sequence[EntityMention]) -> bool:
    if mention.span is None:
        return False
    for existing in selected:
        if existing.span is None:
            continue
        if existing.source != mention.source or existing.source_id != mention.source_id:
            continue
        if existing.type != mention.type:
            continue
        if _spans_overlap(mention.span, existing.span):
            return True
    return False


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _dedupe_raw_candidates(
    candidates: Sequence[_RawCandidate],
) -> tuple[_RawCandidate, ...]:
    best: dict[str, _RawCandidate] = {}
    for candidate in candidates:
        existing = best.get(candidate.entity.entity_id)
        if existing is None or candidate.base_score > existing.base_score:
            best[candidate.entity.entity_id] = candidate
            continue
        if candidate.base_score == existing.base_score:
            sources = tuple(
                _unique_sources(
                    [*existing.candidate_sources, *candidate.candidate_sources]
                )
            )
            best[candidate.entity.entity_id] = _RawCandidate(
                entity=existing.entity,
                candidate_sources=sources,
                base_score=existing.base_score,
                evidence={
                    **existing.evidence,
                    "candidate_source": list(sources),
                },
            )
    return tuple(best.values())


def _dedupe_scored_candidates(
    candidates: Sequence[EntityCandidateScore],
) -> tuple[EntityCandidateScore, ...]:
    best: dict[str, EntityCandidateScore] = {}
    for candidate in candidates:
        existing = best.get(candidate.entity_id)
        if existing is None or candidate.confidence > existing.confidence:
            best[candidate.entity_id] = candidate
    return tuple(best.values())


def _find_phrase_occurrences(
    text: str,
    phrase: str,
) -> tuple[tuple[tuple[int, int], str], ...]:
    if not phrase:
        return ()
    text_norm_map = _normalized_with_offsets(text)
    phrase_norm = _normalize(phrase)
    if not phrase_norm:
        return ()
    normalized_text = "".join(char for char, _ in text_norm_map)
    matches: list[tuple[tuple[int, int], str]] = []
    start = 0
    while start < len(normalized_text):
        index = normalized_text.find(phrase_norm, start)
        if index < 0:
            break
        end = index + len(phrase_norm)
        if _valid_phrase_boundary(normalized_text, index, end, phrase_norm):
            raw_start = text_norm_map[index][1]
            raw_end = text_norm_map[end - 1][1] + 1
            matches.append(((raw_start, raw_end), text[raw_start:raw_end]))
        start = index + 1
    return tuple(matches)


def _normalized_with_offsets(text: str) -> list[tuple[str, int]]:
    output: list[tuple[str, int]] = []
    for index, char in enumerate(text):
        normalized = _normalize_char(char)
        if normalized:
            output.append((normalized, index))
    return output


def _valid_phrase_boundary(
    normalized_text: str,
    start: int,
    end: int,
    phrase_norm: str,
) -> bool:
    if not phrase_norm.isascii() or not any(char.isalnum() for char in phrase_norm):
        return True
    before = normalized_text[start - 1] if start > 0 else ""
    after = normalized_text[end] if end < len(normalized_text) else ""
    if before and before.isascii() and before.isalnum():
        return False
    if after and after.isascii() and after.isalnum():
        return False
    if phrase_norm == "500" and before == "a":
        return False
    return True


def _structured_unknown_strategy_mentions(
    text: str,
    *,
    source: MentionSource,
    source_id: str,
    evidence: dict[str, object],
    existing: Sequence[EntityMention],
) -> tuple[EntityMention, ...]:
    mentions: list[EntityMention] = []
    prefix = "中证"
    start = 0
    while start < len(text):
        index = text.find(prefix, start)
        if index < 0:
            break
        end = index + len(prefix)
        while end < len(text) and _is_index_name_char(text[end]):
            end += 1
        if end > index + len(prefix):
            span = (index, end)
            raw_text = text[index:end]
            if not _span_already_covered(span, "strategy", existing):
                mentions.append(
                    EntityMention(
                        mention_id=_mention_id(
                            source,
                            source_id,
                            "strategy",
                            raw_text,
                            span,
                        ),
                        type="strategy",
                        raw_text=raw_text,
                        source=source,
                        source_id=source_id,
                        span=span,
                        extraction_source="structured_parser",
                        evidence=evidence,
                    )
                )
        start = max(end, index + 1)
    return tuple(mentions)


def _span_already_covered(
    span: tuple[int, int],
    entity_type: EntityType,
    mentions: Sequence[EntityMention],
) -> bool:
    return any(
        mention.type == entity_type
        and mention.span is not None
        and _spans_overlap(span, mention.span)
        for mention in mentions
    )


def _is_index_name_char(char: str) -> bool:
    return char.isascii() and char.isalnum()


def _user_visible_text(message: str) -> str:
    lines = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("[adapter_") and stripped.endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _mention_id(
    source: MentionSource,
    source_id: str,
    entity_type: EntityType,
    raw_text: str,
    span: tuple[int, int] | None,
) -> str:
    span_text = "" if span is None else f"{span[0]}:{span[1]}"
    raw = _normalize(raw_text)
    return f"{source}:{source_id}:{entity_type}:{span_text}:{raw}"


def _normalize(value: object) -> str:
    output = []
    for char in str(value or "").strip().lower():
        normalized = _normalize_char(char)
        if normalized:
            output.append(normalized)
    return "".join(output)


def _normalize_char(char: str) -> str:
    if char in {" ", "\t", "\n", "\r", "_", "-", "—", "–"}:
        return ""
    return char.lower()


def _unique_text(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = _normalize(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return tuple(output)


def _unique_sources(values: Sequence[CandidateSource]) -> tuple[CandidateSource, ...]:
    seen: set[CandidateSource] = set()
    output: list[CandidateSource] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)


def _unique_phrase_sources(
    values: Sequence[tuple[str, MentionExtractionSource]],
) -> tuple[tuple[str, MentionExtractionSource], ...]:
    seen: set[tuple[str, MentionExtractionSource]] = set()
    output: list[tuple[str, MentionExtractionSource]] = []
    for phrase, source in values:
        key = (_normalize(phrase), source)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        output.append((phrase, source))
    return tuple(output)


def _unique_entities(values: Sequence[DomainEntity]) -> tuple[DomainEntity, ...]:
    seen: set[tuple[EntityType, str, str | None]] = set()
    output: list[DomainEntity] = []
    for entity in values:
        key = (entity.type, entity.entity_id, entity.channel_id)
        if key in seen:
            continue
        seen.add(key)
        output.append(entity)
    return tuple(output)
