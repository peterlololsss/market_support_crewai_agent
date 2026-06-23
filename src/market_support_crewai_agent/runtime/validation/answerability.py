from __future__ import annotations

from typing import Literal

from pydantic import Field

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.planning.clarification import (
    clarification_spec,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    select_evidence_for_plan,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    evidence_artifact_type,
    evidence_id,
    is_history_fact,
    is_missing,
    lookup_path,
    source_metadata_for_fact,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    EvidenceSelection,
    abstention_response_text,
)
from market_support_crewai_agent.schemas import ReplyRequest, StrictModel

AnswerabilityAmbiguity = Literal[
    "none",
    "missing_strategy",
    "multiple_channels",
    "multiple_products",
    "unknown_artifact",
    "other",
]
AnswerabilityResponseMode = Literal["answer", "abstain", "clarify"]


class DisallowedEvidence(StrictModel):
    evidence_id: str
    reason: str = Field(min_length=1)


class AnswerabilityAssessment(StrictModel):
    can_answer: bool
    capability_id: str
    required_artifacts: list[str] = Field(default_factory=list)
    available_matching_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    required_runtime_inputs: list[str] = Field(default_factory=list)
    missing_runtime_inputs: list[str] = Field(default_factory=list)
    allowed_evidence_ids: list[str] = Field(default_factory=list)
    disallowed_evidence_ids: list[DisallowedEvidence] = Field(default_factory=list)
    ambiguity: AnswerabilityAmbiguity = "none"
    recommended_response_mode: AnswerabilityResponseMode
    user_facing_reason: str = ""


class AnswerabilityGate:
    """Decide whether the current plan can answer from current evidence only."""

    def assess(
        self,
        *,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        domain_context: DomainContext,
        plan: ExecutionPlan,
        policy: PolicyManifest,
        evidence_facts: list[EvidenceFact],
    ) -> AnswerabilityAssessment:
        selected_capability_id = _selected_capability_id(plan)
        manifest = CAPABILITY_MANIFEST_REGISTRY.find(selected_capability_id)
        if manifest is None:
            return AnswerabilityAssessment(
                can_answer=False,
                capability_id=selected_capability_id,
                ambiguity="unknown_artifact",
                recommended_response_mode="abstain",
                user_facing_reason="老师，这个问题我这边暂时无法确认，先不回答避免信息不准确。",
            )

        selection = select_evidence_for_plan(
            plan=plan,
            policy=policy,
            evidence_facts=evidence_facts,
            domain_context=domain_context,
        )
        required_runtime_inputs = list(manifest.required_inputs)
        missing_runtime_inputs = _missing_runtime_inputs(
            required_runtime_inputs,
            request=request,
            canonical_context=canonical_context,
            plan=plan,
        )
        required_artifacts = list(manifest.required_artifacts)
        available_matching_artifacts = _matching_artifacts(selection.accepted)
        missing_artifacts = _missing_artifacts(required_artifacts, selection.accepted)
        evidence_missing = _required_evidence_missing(manifest, selection.accepted)
        if evidence_missing:
            for artifact in required_artifacts:
                if artifact not in missing_artifacts:
                    missing_artifacts.append(artifact)

        allowed_evidence_ids = [evidence_id(fact) for fact in selection.accepted]
        disallowed_evidence_ids = _disallowed_evidence(
            manifest,
            evidence_facts,
            selection,
        )

        ambiguity: AnswerabilityAmbiguity = "none"
        recommended: AnswerabilityResponseMode = "answer"
        reason = ""
        if plan.response_mode == "clarification" or plan.ambiguity_slots:
            ambiguity = "other"
            recommended = "clarify"
            reason = _clarification_reason(plan)
        elif missing_runtime_inputs:
            ambiguity = "other"
            recommended = "clarify"
            reason = "我需要先确认缺少的信息，再回答。"
        elif missing_artifacts or evidence_missing:
            ambiguity = "unknown_artifact"
            recommended = "abstain"
            reason = _missing_artifact_reason(manifest, missing_artifacts)

        return AnswerabilityAssessment(
            can_answer=recommended == "answer",
            capability_id=manifest.id,
            required_artifacts=required_artifacts,
            available_matching_artifacts=available_matching_artifacts,
            missing_artifacts=missing_artifacts,
            required_runtime_inputs=required_runtime_inputs,
            missing_runtime_inputs=missing_runtime_inputs,
            allowed_evidence_ids=allowed_evidence_ids,
            disallowed_evidence_ids=disallowed_evidence_ids,
            ambiguity=ambiguity,
            recommended_response_mode=recommended,
            user_facing_reason=reason,
        )


def _selected_capability_id(plan: ExecutionPlan) -> str:
    if plan.plan_spec is not None:
        if plan.answer_capabilities:
            answer_capabilities = set(plan.answer_capabilities)
            for unit in plan.plan_spec.plan_units:
                manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.selected_capability_id)
                if (
                    manifest is not None
                    and manifest.runtime_capability in answer_capabilities
                    and manifest.capability_type in {"answer", "summary"}
                ):
                    return manifest.id
        return plan.plan_spec.plan_units[0].selected_capability_id
    if plan.response_mode == "knowledge_answer" and plan.answer_capabilities:
        capability_name = str(plan.answer_capabilities[0])
        manifests = [
            manifest
            for manifest in CAPABILITY_MANIFEST_REGISTRY.list()
            if manifest.runtime_capability == capability_name
            and manifest.capability_type in {"answer", "summary"}
        ]
        if manifests:
            return manifests[0].id
    return "general.abstention"


def _missing_runtime_inputs(
    required_inputs: list[str],
    *,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    plan: ExecutionPlan,
) -> list[str]:
    runtime_inputs = {
        "request": request,
        "canonical_context": canonical_context,
        "plan": plan,
    }
    return [
        input_name
        for input_name in required_inputs
        if is_missing(lookup_path(runtime_inputs, input_name))
    ]


def _matching_artifacts(facts: tuple[EvidenceFact, ...]) -> list[str]:
    output: list[str] = []
    for fact in facts:
        value = fact.source_id or fact.artifact_type
        if value and value not in output:
            output.append(value)
    return output


def _missing_artifacts(
    required_artifacts: list[str],
    accepted_facts: tuple[EvidenceFact, ...],
) -> list[str]:
    accepted_artifacts = {
        evidence_artifact_type(fact)
        for fact in accepted_facts
        if bool(fact.value)
    }
    return [
        artifact
        for artifact in required_artifacts
        if artifact not in accepted_artifacts
    ]


def _required_evidence_missing(manifest, accepted_facts: tuple[EvidenceFact, ...]) -> bool:
    contract = manifest.evidence_contract
    truthy_fact_types = {str(fact.fact_type) for fact in accepted_facts if bool(fact.value)}
    required_types = contract.required_evidence_types or contract.required_fact_types
    if any(fact_type not in truthy_fact_types for fact_type in required_types):
        return True
    if contract.any_of_fact_types and not any(
        fact_type in truthy_fact_types for fact_type in contract.any_of_fact_types
    ):
        return True
    minimum = contract.minimum_evidence_count or contract.min_facts
    truthy_count = sum(1 for fact in accepted_facts if bool(fact.value))
    return minimum > 0 and truthy_count < minimum


def _disallowed_evidence(
    manifest,
    evidence_facts: list[EvidenceFact],
    selection: EvidenceSelection,
) -> list[DisallowedEvidence]:
    allowed = {evidence_id(fact) for fact in selection.accepted}
    decision_reasons = {
        evidence_id: decision.reason_code
        for decision in selection.decisions
        for evidence_id in decision.evidence_seen
        if evidence_id
    }
    output: list[DisallowedEvidence] = []
    seen: set[str] = set()
    for fact in evidence_facts:
        evidence_id_value = evidence_id(fact)
        if not evidence_id_value or evidence_id_value in allowed or evidence_id_value in seen:
            continue
        reason = decision_reasons.get(evidence_id_value) or _contract_disallowed_reason(
            manifest,
            fact,
        )
        if not reason:
            continue
        seen.add(evidence_id_value)
        output.append(DisallowedEvidence(evidence_id=evidence_id_value, reason=reason))
    return output


def _contract_disallowed_reason(manifest, fact: EvidenceFact) -> str:
    contract = manifest.evidence_contract
    source_metadata = source_metadata_for_fact(fact)
    if not contract.allow_history and is_history_fact(fact):
        return "history_source_not_current_artifact"
    if (
        source_metadata is not None
        and not source_metadata.evidence_allowed_by_default
        and not contract.allow_history
    ):
        return "source_not_evidence_by_default"
    forbidden_sources = set(contract.disallowed_source_types) | set(
        contract.forbidden_source_types
    )
    if fact.source_type in forbidden_sources:
        return "forbidden_source_type"
    if contract.allowed_source_types and fact.source_type not in contract.allowed_source_types:
        return "source_type_not_allowed"
    artifact_type = evidence_artifact_type(fact)
    if artifact_type in set(manifest.forbidden_artifacts):
        return "artifact_type_forbidden"
    allowed_artifacts = set(contract.allowed_artifact_types or manifest.allowed_artifacts)
    if allowed_artifacts and artifact_type not in allowed_artifacts:
        return "artifact_type_not_allowed"
    allowed_fact_types = set(contract.required_fact_types)
    allowed_fact_types.update(contract.required_evidence_types)
    allowed_fact_types.update(contract.any_of_fact_types)
    if allowed_fact_types and str(fact.fact_type) not in allowed_fact_types:
        return "fact_type_not_allowed"
    if not fact.value:
        return "evidence_value_empty"
    return ""

def _missing_artifact_reason(manifest, missing_artifacts: list[str]) -> str:
    label = _artifact_label(missing_artifacts[0] if missing_artifacts else "")
    if manifest.id == "material_pack.product_list":
        return "老师，材料包里的产品范围我这边暂时无法确认。"
    if label:
        return f"老师，这个问题需要以{label}里的准确信息为准，我这边暂时无法确认。"
    return "老师，这个信息我这边暂时无法确认，先不回答避免信息不准确。"


def _clarification_reason(plan: ExecutionPlan) -> str:
    spec = clarification_spec(plan.ambiguity_slots)
    if spec is not None:
        return spec.reason_text
    return abstention_response_text()


def _artifact_label(artifact: str) -> str:
    return {
        "material_pack": "材料包",
        "weekly_report": "周报",
        "monthly_report": "月报",
        "document_context": "文档",
        "history": "历史记录",
    }.get(artifact, artifact)
