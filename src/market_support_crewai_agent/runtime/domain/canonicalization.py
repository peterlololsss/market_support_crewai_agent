from __future__ import annotations

from dataclasses import dataclass, field

from market_support_crewai_agent.runtime.domain.entity_resolution import (
    CanonicalEntityResolver,
    EvidenceSnippet,
)
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.schemas import ReplyRequest


@dataclass(frozen=True)
class CanonicalResolutionMetrics:
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
class CanonicalContext:
    material_pack_options: tuple[str, ...] = ()
    entities: tuple[object, ...] = ()
    ambiguities: tuple[str, ...] = ()
    resolutions: tuple[object, ...] = ()
    resolution_metrics: CanonicalResolutionMetrics = field(
        default_factory=CanonicalResolutionMetrics
    )

    def to_prompt_dict(self) -> dict:
        return {
            "material_pack_options": list(self.material_pack_options),
            "entities": [],
            "ambiguities": list(self.ambiguities),
            "resolution_metrics": self.resolution_metrics.to_prompt_dict(),
            "resolutions": [],
        }


def canonicalize_request(
    request: ReplyRequest,
    *,
    domain_context: DomainContext | None = None,
    evidence_snippets: tuple[EvidenceSnippet, ...] = (),
    resolver: CanonicalEntityResolver | None = None,
) -> CanonicalContext:
    del domain_context, evidence_snippets, resolver
    return CanonicalContext(
        material_pack_options=_unique(
            option.strip() for option in request.material_pack_options
        )
    )


def _unique(values) -> tuple[str, ...]:
    seen = set()
    output = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)
