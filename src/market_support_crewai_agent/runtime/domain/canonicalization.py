from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from market_support_crewai_agent.runtime.domain.entity_resolution import (
    CandidateSource,
    CanonicalEntityResolver,
    EntityResolution,
    EntityResolutionMetrics,
    EntityResolutionResult,
    EntityType,
    EvidenceSnippet,
)
from market_support_crewai_agent.runtime.domain.ontology import (
    DomainContext,
    DomainContextBuilder,
)
from market_support_crewai_agent.schemas import ReplyRequest

CanonicalEntityType = EntityType
CanonicalEntitySource = CandidateSource
CanonicalStatus = Literal["resolved", "ambiguous", "unknown"]


@dataclass(frozen=True)
class CanonicalEntity:
    type: CanonicalEntityType
    raw_text: str
    canonical_name: str
    source: CanonicalEntitySource
    confidence: float = 1.0
    canonical_id: str | None = None
    resolver_stage: str = "disambiguation"
    rationale: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict:
        return {
            "type": self.type,
            "raw_text": self.raw_text,
            "canonical_name": self.canonical_name,
            "canonical_id": self.canonical_id,
            "source": self.source,
            "confidence": self.confidence,
            "resolver_stage": self.resolver_stage,
            "rationale": self.rationale,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CanonicalContext:
    strategy_status: CanonicalStatus = "unknown"
    selected_strategy: str | None = None
    strategy_candidates: tuple[str, ...] = ()
    entities: tuple[CanonicalEntity, ...] = ()
    ambiguities: tuple[str, ...] = ()
    resolutions: tuple[EntityResolution, ...] = ()
    resolution_metrics: EntityResolutionMetrics = field(
        default_factory=EntityResolutionMetrics
    )

    def to_prompt_dict(self) -> dict:
        return {
            "strategy_status": self.strategy_status,
            "selected_strategy": self.selected_strategy,
            "strategy_candidates": list(self.strategy_candidates),
            "entities": [entity.to_prompt_dict() for entity in self.entities],
            "ambiguities": list(self.ambiguities),
            "resolution_metrics": self.resolution_metrics.to_prompt_dict(),
            "resolutions": [
                _compact_resolution(resolution)
                for resolution in self.resolutions
            ],
        }


class CanonicalContextProjector:
    """Projects typed entity resolutions into the runtime CanonicalContext."""

    def to_context(
        self,
        result: EntityResolutionResult,
        request: ReplyRequest,
    ) -> CanonicalContext:
        strategy_resolutions = result.by_type("strategy")
        strategy_entities = tuple(
            _canonical_entity_from_resolution(resolution)
            for resolution in strategy_resolutions
            if resolution.status == "resolved"
        )
        strategy_candidates = _strategy_candidates(strategy_resolutions)
        selected_strategy = _selected_strategy(strategy_resolutions)
        strategy_status = _strategy_status(strategy_resolutions, selected_strategy)
        ambiguities = (
            ("multiple_strategy_candidates",)
            if strategy_status == "ambiguous"
            else ()
        )
        if strategy_status == "unknown":
            strategy_candidates = _clean_available_strategies(request.available_strategies)
        return CanonicalContext(
            strategy_status=strategy_status,
            selected_strategy=selected_strategy,
            strategy_candidates=strategy_candidates,
            entities=strategy_entities,
            ambiguities=ambiguities,
            resolutions=result.resolutions,
            resolution_metrics=result.metrics,
        )


def canonicalize_request(
    request: ReplyRequest,
    *,
    domain_context: DomainContext | None = None,
    evidence_snippets: tuple[EvidenceSnippet, ...] = (),
    resolver: CanonicalEntityResolver | None = None,
) -> CanonicalContext:
    domain_context = domain_context or DomainContextBuilder().build(request)
    result = (resolver or CanonicalEntityResolver()).resolve_request(
        request,
        domain_context=domain_context,
        evidence_snippets=evidence_snippets,
    )
    return CanonicalContextProjector().to_context(result, request)


def _canonical_entity_from_resolution(resolution: EntityResolution) -> CanonicalEntity:
    top = resolution.candidates[0] if resolution.candidates else None
    source: CanonicalEntitySource = (
        top.candidate_sources[0]
        if top is not None and top.candidate_sources
        else "semantic_provider"
    )
    return CanonicalEntity(
        type=resolution.type,
        raw_text=resolution.mention.raw_text,
        canonical_name=resolution.canonical_name or "",
        source=source,
        confidence=resolution.confidence,
        canonical_id=resolution.entity_id,
        resolver_stage=resolution.resolver_stage,
        rationale=resolution.rationale,
        evidence=resolution.evidence,
    )


def _compact_resolution(resolution: EntityResolution) -> dict:
    return {
        "status": resolution.status,
        "type": resolution.type,
        "mention": {
            "raw_text": resolution.mention.raw_text,
            "source": resolution.mention.source,
            "extraction_source": resolution.mention.extraction_source,
        },
        "entity_id": resolution.entity_id,
        "canonical_name": resolution.canonical_name,
        "confidence": round(resolution.confidence, 4),
        "resolver_stage": resolution.resolver_stage,
        "rationale": resolution.rationale,
        "candidates": [
            {
                "entity_id": candidate.entity_id,
                "canonical_name": candidate.canonical_name,
                "confidence": round(candidate.confidence, 4),
                "candidate_sources": list(candidate.candidate_sources),
            }
            for candidate in resolution.candidates[:3]
        ],
    }


def _strategy_status(
    resolutions: tuple[EntityResolution, ...],
    selected_strategy: str | None,
) -> CanonicalStatus:
    if any(resolution.status == "ambiguous" for resolution in resolutions):
        return "ambiguous"
    resolved_names = _unique(
        resolution.canonical_name or ""
        for resolution in resolutions
        if resolution.status == "resolved"
    )
    if len(resolved_names) > 1:
        return "ambiguous"
    if selected_strategy:
        return "resolved"
    return "unknown"


def _selected_strategy(resolutions: tuple[EntityResolution, ...]) -> str | None:
    resolved_names = _unique(
        resolution.canonical_name or ""
        for resolution in resolutions
        if resolution.status == "resolved"
    )
    return resolved_names[0] if len(resolved_names) == 1 else None


def _strategy_candidates(resolutions: tuple[EntityResolution, ...]) -> tuple[str, ...]:
    candidates: list[str] = []
    for resolution in resolutions:
        if resolution.status == "resolved" and resolution.canonical_name:
            candidates.append(resolution.canonical_name)
            continue
        candidates.extend(
            candidate.canonical_name
            for candidate in resolution.candidates
            if candidate.canonical_name
        )
    return _unique(candidates)


def _clean_available_strategies(strategies: list[str]) -> tuple[str, ...]:
    return _unique(strategy.strip() for strategy in strategies if strategy.strip())


def _unique(values) -> tuple[str, ...]:
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)
