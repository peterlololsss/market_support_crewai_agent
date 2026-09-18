from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import ConfigDict, Field, JsonValue, model_validator

from market_support_crewai_agent.runtime.planning.plan_hash import execution_plan_id_v2
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.policy.capabilities import (
    ArtifactKind,
    CapabilityName,
    ManifestRefV1,
    ResponseMode,
)
from market_support_crewai_agent.runtime.policy.compliance import (
    ComplianceReasonCode,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveType,
    OutboundActionType,
)


class ExecutionPlanValidationError(ValueError):
    """Raised when a canonical ExecutionPlanV2 invariant is violated."""


class _FrozenExecutionModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


ExecutionPlanOriginV1 = Literal[
    "planner",
    "input_policy",
    "direct_send",
    "approved_static_shortcut",
    "deterministic",
    "remediation",
]


class ExecutionPlanTimeRangeV1(_FrozenExecutionModel):
    period: str | None = Field(default=None, max_length=80)
    start: str | None = Field(default=None, max_length=80)
    end: str | None = Field(default=None, max_length=80)
    label: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def _validate_control_characters(self) -> ExecutionPlanTimeRangeV1:
        for value in (self.period, self.start, self.end, self.label):
            if value is not None and any(ord(character) < 32 for character in value):
                raise ExecutionPlanValidationError(
                    "execution_plan_time_range_control_character"
                )
        return self


class DistributionExecutionDomainScopeV2(_FrozenExecutionModel):
    kind: Literal["distribution"] = "distribution"
    business_scope_ref: str = Field(pattern=r"^bsr:[0-9a-f]{32}$")
    channel_kind: Literal["bank", "non_bank", "unknown"]
    material_pack_option: str | None = Field(default=None, max_length=80)
    product_ids: tuple[str, ...] = Field(default=(), max_length=1_000)
    time_range: ExecutionPlanTimeRangeV1 | None = None

    @model_validator(mode="after")
    def _validate_products(self) -> DistributionExecutionDomainScopeV2:
        if len(set(self.product_ids)) != len(self.product_ids):
            raise ExecutionPlanValidationError("execution_scope_duplicate_product_id")
        if any(
            not product_id or len(product_id) > 160 for product_id in self.product_ids
        ):
            raise ExecutionPlanValidationError("execution_scope_invalid_product_id")
        return self


class UnscopedExecutionDomainScopeV2(_FrozenExecutionModel):
    kind: Literal["unscoped"] = "unscoped"


ExecutionDomainScopeV2: TypeAlias = Annotated[
    DistributionExecutionDomainScopeV2 | UnscopedExecutionDomainScopeV2,
    Field(discriminator="kind"),
]


class CanonicalAdapterResolveV1(_FrozenExecutionModel):
    resolve_type: AdapterResolveType
    material_pack_option: str | None = Field(default=None, max_length=80)
    artifact_id: str | None = Field(default=None, max_length=160)


class CanonicalActionIntentV1(_FrozenExecutionModel):
    action_type: OutboundActionType
    capability: CapabilityName
    material_pack_option: str | None = Field(default=None, max_length=80)


class ExecutionPlanUnitV2(_FrozenExecutionModel):
    unit_id: str = Field(min_length=1, max_length=120)
    manifest_ref: ManifestRefV1
    answerability_policy: Literal[
        "answer",
        "send",
        "clarify",
        "abstain",
        "refuse",
        "handoff",
        "smalltalk",
        "no_reply",
    ]
    artifact_kind: ArtifactKind
    runtime_capabilities: tuple[CapabilityName, ...] = Field(default=(), max_length=8)
    answer_capabilities: tuple[CapabilityName, ...] = Field(default=(), max_length=4)
    adapter_resolves: tuple[CanonicalAdapterResolveV1, ...] = Field(
        default=(), max_length=8
    )
    action_intents: tuple[CanonicalActionIntentV1, ...] = Field(
        default=(), max_length=3
    )
    scope: ExecutionDomainScopeV2
    evidence_query: str | None = Field(default=None, max_length=200)
    ambiguity_slots: tuple[str, ...] = Field(default=(), max_length=8)
    risk_flags: tuple[Literal["weekly_report_rationale_required"], ...] = Field(
        default=()
    )

    @model_validator(mode="after")
    def _validate_unit_canonical(self) -> ExecutionPlanUnitV2:
        if tuple(sorted(set(self.runtime_capabilities))) != self.runtime_capabilities:
            raise ExecutionPlanValidationError(
                "execution_unit_runtime_capabilities_not_canonical"
            )
        if tuple(sorted(set(self.answer_capabilities))) != self.answer_capabilities:
            raise ExecutionPlanValidationError(
                "execution_unit_answer_capabilities_not_canonical"
            )
        if len(set(self.ambiguity_slots)) != len(self.ambiguity_slots):
            raise ExecutionPlanValidationError(
                "execution_unit_ambiguity_slots_not_canonical"
            )
        if any(not slot or len(slot) > 120 for slot in self.ambiguity_slots):
            raise ExecutionPlanValidationError("execution_unit_invalid_ambiguity_slot")
        return self


class ComplianceDecisionV1(_FrozenExecutionModel):
    is_compliant: bool | None = None
    reason_code: ComplianceReasonCode = "unknown"
    reason: str = Field(default="", max_length=400)


class ExecutionPlanV2(_FrozenExecutionModel):
    contract_version: Literal["execution-plan.v2"] = "execution-plan.v2"
    execution_plan_id: str = Field(pattern=r"^epl1:[0-9a-f]{64}$")
    origin: ExecutionPlanOriginV1
    user_need: str = Field(min_length=1, max_length=500)
    artifact_kind: ArtifactKind
    response_mode: ResponseMode
    compliance: ComplianceDecisionV1
    units: tuple[ExecutionPlanUnitV2, ...] = Field(min_length=1, max_length=4)
    selected_manifest_refs: tuple[ManifestRefV1, ...] = Field(
        min_length=1, max_length=4
    )
    adapter_resolves: tuple[CanonicalAdapterResolveV1, ...] = Field(
        default=(), max_length=8
    )
    action_intents: tuple[CanonicalActionIntentV1, ...] = Field(
        default=(), max_length=3
    )
    guardrail_decisions: tuple[GuardrailDecision, ...] = Field(
        default=(), max_length=32
    )
    confidence: float = Field(ge=0.0, le=1.0)
    plan_spec: PlanSpec | None = None

    @model_validator(mode="after")
    def _validate_aggregate_derivations(self) -> ExecutionPlanV2:
        if len({unit.unit_id for unit in self.units}) != len(self.units):
            raise ExecutionPlanValidationError("execution_plan_duplicate_unit_id")
        expected_refs = _selected_manifest_refs(self.units)
        if self.selected_manifest_refs != expected_refs:
            raise ExecutionPlanValidationError(
                "execution_plan_selected_manifest_refs_mismatch"
            )
        expected_resolves = _unique_resolves(
            tuple(resolve for unit in self.units for resolve in unit.adapter_resolves)
        )
        if self.adapter_resolves != expected_resolves:
            raise ExecutionPlanValidationError(
                "execution_plan_adapter_resolves_mismatch"
            )
        expected_actions = tuple(
            intent for unit in self.units for intent in unit.action_intents
        )
        if self.action_intents != expected_actions:
            raise ExecutionPlanValidationError("execution_plan_action_intents_mismatch")
        if (self.origin == "planner") != (self.plan_spec is not None):
            raise ExecutionPlanValidationError(
                "execution_plan_plan_spec_origin_mismatch"
            )
        if self.execution_plan_id != execution_plan_id_v2(self):
            raise ExecutionPlanValidationError("execution_plan_id_mismatch")
        return self


def _selected_manifest_refs(
    units: tuple[ExecutionPlanUnitV2, ...],
) -> tuple[ManifestRefV1, ...]:
    seen: set[str] = set()
    refs: list[ManifestRefV1] = []
    for unit in units:
        if unit.manifest_ref.manifest_id not in seen:
            seen.add(unit.manifest_ref.manifest_id)
            refs.append(unit.manifest_ref)
    return tuple(refs)


def _unique_resolves(
    values: tuple[CanonicalAdapterResolveV1, ...],
) -> tuple[CanonicalAdapterResolveV1, ...]:
    seen: set[tuple[AdapterResolveType, str | None, str | None]] = set()
    output: list[CanonicalAdapterResolveV1] = []
    for value in values:
        key = (value.resolve_type, value.material_pack_option, value.artifact_id)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return tuple(output)


PlanValidationSeverity = Literal["error", "fatal"]
PlanValidationCode = Literal[
    "response_mode_not_allowed",
    "capability_not_allowed",
    "adapter_resolve_not_allowed",
    "too_many_evidence_calls",
    "action_not_allowed",
    "action_capability_mismatch",
    "action_missing_required_resolve",
    "non_compliant_plan_has_actions",
    "non_compliant_plan_not_refusal",
    "unknown_compliance_has_actions",
    "ambiguous_plan_has_actions",
    "ambiguous_plan_not_clarification",
    "clarification_missing_supported_slot",
    "knowledge_answer_missing_capability",
    "material_pack_scope_not_allowed",
    "plan_spec_capability_not_found",
    "plan_spec_runtime_capability_not_allowed",
]


@dataclass(frozen=True, slots=True)
class PlanValidationIssue:
    code: PlanValidationCode
    message: str
    severity: PlanValidationSeverity = "error"
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanValidationResult:
    valid: bool
    issues: tuple[PlanValidationIssue, ...] = ()

    @property
    def fatal(self) -> bool:
        return any(issue.severity == "fatal" for issue in self.issues)
