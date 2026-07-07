from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.runtime.domain.planning.message_normalization import (
    normalize_compact_message,
)
from market_support_crewai_agent.runtime.domain.planning.models import (
    ExecutionPlan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.schemas import ReplyRequest

InputPolicyStatus = Literal["no_match", "guardrail_handoff"]

@dataclass(frozen=True, slots=True)
class InputPolicyResult:
    status: InputPolicyStatus
    plan: ExecutionPlan | None = None
    reason_code: str = ""
    rule_id: str = ""

    @property
    def matched(self) -> bool:
        return self.status != "no_match" and self.plan is not None


def match_input_policy(
    request: ReplyRequest,
    policy: PolicyManifest,
) -> InputPolicyResult:
    del policy
    normalized = normalize_compact_message(request.message)
    if not normalized:
        return InputPolicyResult(status="no_match", reason_code="empty_message")

    return InputPolicyResult(status="no_match", reason_code="no_match")
