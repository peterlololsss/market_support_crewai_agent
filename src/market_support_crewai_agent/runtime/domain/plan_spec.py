from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    EvidenceContract,
)
from market_support_crewai_agent.schemas import ChannelType, StrictModel

AnswerabilityPolicy = Literal[
    "answer",
    "send",
    "clarify",
    "abstain",
    "refuse",
    "handoff",
    "smalltalk",
    "no_reply",
]
ScopeMatchField = Literal[
    "channel_id",
    "channel_kind",
    "material_pack_option",
    "time_range",
    "product_id",
    "product_ids",
    "artifact_type",
]


class PlanTimeRange(StrictModel):
    period: str | None = None
    start: str | None = None
    end: str | None = None
    label: str | None = None


class PlanDomainScope(StrictModel):
    channel_id: str = Field(min_length=1)
    channel_kind: ChannelType
    material_pack_option: str | None = None
    product_ids: list[str] = Field(default_factory=list)
    time_range: PlanTimeRange | None = None


class PlanStep(StrictModel):
    step_id: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=300)
    uses_artifacts: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    allowed_artifacts: list[str] = Field(default_factory=list)
    forbidden_artifacts: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    evidence_query: str | None = Field(default=None, max_length=200)


class PlanUnit(StrictModel):
    unit_id: str = Field(min_length=1)
    selected_capability_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$",
    )
    domain_scope: PlanDomainScope
    required_artifacts: list[str] = Field(default_factory=list)
    allowed_artifacts: list[str] = Field(default_factory=list)
    forbidden_artifacts: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    answerability_policy: AnswerabilityPolicy
    output_schema_ref: str = Field(min_length=1)
    output_schema: dict[str, Any] | None = None
    evidence_contract_ref: str | None = None
    evidence_contract: EvidenceContract | None = None
    steps: list[PlanStep] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    abstention_cases: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact_sets(self):
        required = set(self.required_artifacts)
        allowed = set(self.allowed_artifacts)
        forbidden = set(self.forbidden_artifacts)
        if required & forbidden:
            raise ValueError("required_artifacts cannot also be forbidden")
        if allowed & forbidden:
            raise ValueError("allowed_artifacts cannot also be forbidden")
        if required and allowed and not required <= allowed:
            raise ValueError("required_artifacts must be included in allowed_artifacts")
        return self


class PlanSpec(StrictModel):
    contract_version: Literal["plan-spec"] = "plan-spec"
    plan_id: str = Field(min_length=1)
    user_intent_summary: str = Field(min_length=1, max_length=500)
    plan_units: list[PlanUnit] = Field(min_length=1, max_length=4)
    risk_flags: list[str] = Field(default_factory=list)
