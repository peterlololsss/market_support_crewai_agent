from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.planning.message_normalization import (
    normalize_compact_message,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    HANDOFF_TEXT_METADATA_KEY,
    HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY,
    make_decision,
)

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
    plan: ExecutionPlanV2 | None = None
    reason_code: str = ""
    rule_id: str = ""

    @property
    def matched(self) -> bool:
        return self.status != "no_match" and self.plan is not None


def match_input_policy(
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
    *,
    rules: tuple[InputPolicyRule, ...] = DEFAULT_INPUT_POLICY_RULES,
) -> InputPolicyResult:
    normalized = normalize_compact_message(request.message)
    if not normalized:
        return InputPolicyResult(status="no_match", reason_code="empty_message")
    for rule in rules:
        if _rule_matches(rule, normalized):
            return _handoff_result(rule, policy, scope_authority)

    return InputPolicyResult(status="no_match", reason_code="no_match")


def _rule_matches(rule: InputPolicyRule, normalized: str) -> bool:
    compact = _compact_policy_text(normalized)
    return any(_compact_policy_text(trigger) in compact for trigger in rule.contains)


def _compact_policy_text(value: str) -> str:
    return value.lower().replace("+", "")


def _handoff_result(
    rule: InputPolicyRule,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
) -> InputPolicyResult:
    has_sales_handoff = (
        "sales.handoff" in {ref.manifest_id for ref in policy.eligible_capabilities}
        and "sales_mention" in policy.allowed_adapter_resolves
        and "sales" in policy.allowed_mention_types
    )
    is_direct = policy.scene == "direct"
    manifest_id = (
        "general.handoff"
        if is_direct
        else "sales.handoff"
        if has_sales_handoff
        else "general.abstention"
    )
    answerability = "handoff" if manifest_id != "general.abstention" else "abstain"
    metadata = {
        HANDOFF_TEXT_METADATA_KEY: (
            "如需人工协助，请联系您的客户经理或人工客服继续处理。"
            if is_direct
            else rule.handoff_text
            if has_sales_handoff
            else rule.handoff_unavailable_text
        ),
        HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY: rule.handoff_unavailable_text,
    }

    return InputPolicyResult(
        status="guardrail_handoff",
        plan=finalize_execution_plan_v2(
            DeterministicPlanOriginInputV1(
                user_need=rule.user_need,
                units=(
                    DeterministicPlanUnitV1(
                        unit_id=f"input-policy-{rule.rule_id}",
                        manifest_id=manifest_id,
                        answerability_policy=answerability,
                    ),
                ),
                compliance_reason_code="customer_service_request",
                guardrail_decisions=(
                    make_decision(
                        "block",
                        "input",
                        rule.reason_code,
                        human_reason=rule.human_reason,
                        metadata=metadata,
                    ),
                ),
            ),
            policy,
            scope_authority,
            origin="input_policy",
        ),
        reason_code=rule.reason_code,
        rule_id=rule.rule_id,
    )
