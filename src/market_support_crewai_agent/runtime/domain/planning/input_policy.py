from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from market_support_crewai_agent.runtime.domain.capabilities import CapabilityName
from market_support_crewai_agent.runtime.domain.planning.message_normalization import (
    normalize_compact_message,
)
from market_support_crewai_agent.runtime.domain.planning.models import (
    AdapterResolveSpec,
    ComplianceDecision,
    ExecutionPlan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    HANDOFF_TEXT_METADATA_KEY,
    HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY,
    make_decision,
)
from market_support_crewai_agent.schemas import ReplyRequest

InputPolicyStatus = Literal["no_match", "guardrail_handoff"]

_T0_HANDOFF_RULE_ID: Final = "t0_handoff"
_T0_HANDOFF_REASON_CODE: Final = "t0_human_support_required"
_T0_HANDOFF_TEXT: Final = "这个问题需要老师您向群内请销售/支持同事确认哦。我帮您艾特ta~"
_T0_HANDOFF_UNAVAILABLE_TEXT: Final = "这个问题需要老师您向群内请销售/支持同事确认哦。"


@dataclass(frozen=True, slots=True)
class InputPolicyRule:
    rule_id: str
    reason_code: str
    contains: tuple[str, ...]
    user_need: str
    handoff_text: str
    handoff_unavailable_text: str
    human_reason: str


DEFAULT_INPUT_POLICY_RULES: Final = (
    InputPolicyRule(
        rule_id=_T0_HANDOFF_RULE_ID,
        reason_code=_T0_HANDOFF_REASON_CODE,
        contains=("t0",),
        user_need="T0 request requires human support",
        handoff_text=_T0_HANDOFF_TEXT,
        handoff_unavailable_text=_T0_HANDOFF_UNAVAILABLE_TEXT,
        human_reason="T0 request requires human support.",
    ),
)

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
    *,
    rules: tuple[InputPolicyRule, ...] = DEFAULT_INPUT_POLICY_RULES,
) -> InputPolicyResult:
    normalized = normalize_compact_message(request.message)
    if not normalized:
        return InputPolicyResult(status="no_match", reason_code="empty_message")
    for rule in rules:
        if _rule_matches(rule, normalized):
            return _handoff_result(rule, policy)

    return InputPolicyResult(status="no_match", reason_code="no_match")


def _rule_matches(rule: InputPolicyRule, normalized: str) -> bool:
    compact = _compact_policy_text(normalized)
    return any(_compact_policy_text(trigger) in compact for trigger in rule.contains)


def _compact_policy_text(value: str) -> str:
    return value.lower().replace("+", "")


def _handoff_result(
    rule: InputPolicyRule,
    policy: PolicyManifest,
) -> InputPolicyResult:
    capabilities: list[CapabilityName] = []
    adapter_resolves: list[AdapterResolveSpec] = []
    if (
        "sales_mention" in policy.allowed_capabilities
        and "sales_mention" in policy.allowed_adapter_resolves
    ):
        capabilities = ["sales_mention"]
        adapter_resolves = [AdapterResolveSpec(resolve_type="sales_mention")]

    return InputPolicyResult(
        status="guardrail_handoff",
        plan=ExecutionPlan(
            user_need=rule.user_need,
            artifact_kind="human_support",
            response_mode="handoff",
            compliance=ComplianceDecision(
                is_compliant=True,
                reason_code="customer_service_request",
                reason=rule.human_reason,
            ),
            capabilities=capabilities,
            adapter_resolves=adapter_resolves,
            guardrail_decisions=[
                make_decision(
                    "block",
                    "input",
                    rule.reason_code,
                    human_reason=rule.human_reason,
                    metadata={
                        HANDOFF_TEXT_METADATA_KEY: rule.handoff_text,
                        HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY: (
                            rule.handoff_unavailable_text
                        ),
                    },
                )
            ],
            confidence=1.0,
        ),
        reason_code=rule.reason_code,
        rule_id=rule.rule_id,
    )
