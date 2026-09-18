from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)


@dataclass(frozen=True, slots=True)
class GroundingProjectionContextV1:
    business_scope_authority: BusinessScopeAuthorityV1
    locator_safety: LocatorSafetyClassifierV1
    evaluation_epoch_seconds: int | None = None
