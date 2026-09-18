from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import ChannelType


class _FrozenPlanModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DistributionPlanDomainScopeV2(_FrozenPlanModel):
    kind: Literal["distribution"] = "distribution"
    business_scope_ref: str = Field(pattern=r"^bsr:[0-9a-f]{32}$")
    channel_kind: ChannelType
    material_pack_option: str | None = Field(default=None, max_length=80)
    product_ids: tuple[str, ...] = Field(default=(), max_length=1_000)

    @model_validator(mode="after")
    def _validate_product_ids(self) -> DistributionPlanDomainScopeV2:
        if len(set(self.product_ids)) != len(self.product_ids):
            raise ValueError("duplicate_product_id")
        if any(
            not product_id or len(product_id) > 160 for product_id in self.product_ids
        ):
            raise ValueError("invalid_product_id")
        return self


class UnscopedPlanDomainScopeV2(_FrozenPlanModel):
    kind: Literal["unscoped"] = "unscoped"


PlanDomainScopeV2: TypeAlias = Annotated[
    Union[DistributionPlanDomainScopeV2, UnscopedPlanDomainScopeV2],
    Field(discriminator="kind"),
]

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


class PlanSpecEvidenceContractV1(StrictModel):
    required_fact_types: tuple[str, ...] = ()
    any_of_fact_types: tuple[str, ...] = ()
    allowed_source_types: tuple[str, ...] = ()
    forbidden_source_types: tuple[str, ...] = ()
    min_facts: int = Field(default=0, ge=0, le=16)

    @model_validator(mode="after")
    def validate_fact_sets(self) -> PlanSpecEvidenceContractV1:
        if set(self.required_fact_types) & set(self.any_of_fact_types):
            raise ValueError("required and any-of evidence facts must not overlap")
        if set(self.allowed_source_types) & set(self.forbidden_source_types):
            raise ValueError("allowed and forbidden evidence sources must not overlap")
        return self


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
    domain_scope: PlanDomainScopeV2
    required_artifacts: list[str] = Field(default_factory=list)
    allowed_artifacts: list[str] = Field(default_factory=list)
    forbidden_artifacts: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    answerability_policy: AnswerabilityPolicy
    output_schema_ref: str = Field(min_length=1)
    output_schema: dict[str, Any] | None = None
    evidence_contract_ref: str | None = None
    evidence_contract: PlanSpecEvidenceContractV1 | None = None
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
