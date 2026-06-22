from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.schemas import ReplyRequest


@dataclass(frozen=True)
class CanonicalContext:
    material_pack_options: tuple[str, ...] = ()

    def to_prompt_dict(self) -> dict:
        return {
            "material_pack_options": list(self.material_pack_options),
        }


def canonicalize_request(
    request: ReplyRequest,
    *,
    domain_context: DomainContext | None = None,
) -> CanonicalContext:
    del domain_context
    return CanonicalContext(
        material_pack_options=_unique(
            option.strip()
            for artifact in request.available_artifacts
            if artifact.type == "material_pack"
            for option in artifact.options
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
