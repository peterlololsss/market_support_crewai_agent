from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.planning.plan_spec import AnswerabilityPolicy
from market_support_crewai_agent.runtime.policy.compliance import ComplianceReasonCode
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)


class ExecutionPlanCompilationError(ValueError):
    """Raised when PlanSpec cannot finalize into ExecutionPlanV2."""


@dataclass(frozen=True, slots=True)
class DeterministicPlanUnitV1:
    unit_id: str
    manifest_id: str
    answerability_policy: AnswerabilityPolicy
    material_pack_option: str | None = None
    evidence_query: str | None = None
    ambiguity_slots: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DeterministicPlanOriginInputV1:
    user_need: str
    units: tuple[DeterministicPlanUnitV1, ...]
    compliance_reason_code: ComplianceReasonCode = "unknown"
    guardrail_decisions: tuple[GuardrailDecision, ...] = ()
    confidence: float = 1.0
