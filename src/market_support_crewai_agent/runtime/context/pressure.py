from __future__ import annotations

from math import ceil
from typing import Any

from market_support_crewai_agent.runtime.context.models import (
    ContextPressureEstimate,
    ContextProjectionPolicy,
    stable_json,
)


class ProjectionLimitError(RuntimeError):
    """Raised when projected context still exceeds the hard token budget."""


class ContextPressureEstimator:
    def token_estimate(self, value: Any) -> int:
        return ceil(len(stable_json(value)) / 4)

    def estimate(
        self,
        *,
        raw_context: Any,
        projected_context: Any,
        policy: ContextProjectionPolicy,
    ) -> ContextPressureEstimate:
        before = self.token_estimate(raw_context)
        after = self.token_estimate(projected_context)
        ratio = after / policy.token_budget if policy.token_budget else 1.0
        return ContextPressureEstimate(
            token_budget=policy.token_budget,
            estimated_tokens_before=before,
            estimated_tokens_after=after,
            pressure_ratio=round(ratio, 4),
            warning_threshold=policy.warning_threshold,
            hard_threshold=policy.hard_threshold,
            warning=ratio >= policy.warning_threshold,
            hard_blocked=ratio >= policy.hard_threshold,
        )
