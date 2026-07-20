from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities import CapabilityName
from market_support_crewai_agent.runtime.domain.planning import (
    ActionIntentSpec,
    ComplianceDecision,
    ExecutionPlan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence.models import EvidenceFact
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
)
from market_support_crewai_agent.runtime.orchestration.direct_actions import (
    DirectAdapterClient,
    DirectMaterialization,
    materialize_direct_output,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    evidence_id,
    trusted_document_context,
)
from market_support_crewai_agent.schemas import (
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
)


def build_direct_knowledge_plan(
    request: ReplyRequest,
    policy: PolicyManifest,
) -> ExecutionPlan:
    enabled = "document_context" in policy.allowed_capabilities
    capabilities: list[CapabilityName] = ["document_context"] if enabled else []
    return ExecutionPlan(
        user_need=request.message,
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer" if enabled else "unable",
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
        ),
        evidence_query=request.message,
        capabilities=capabilities,
        answer_capabilities=capabilities,
        confidence=1.0,
    )


def coerce_direct_output(result) -> DirectComposerOutput | None:
    value = getattr(result, "pydantic", None)
    if value is not None:
        try:
            return DirectComposerOutput.model_validate(value)
        except ValueError:
            return None
    try:
        return DirectComposerOutput.model_validate_json(getattr(result, "raw", ""))
    except ValueError:
        return None


async def materialize_allowed_direct_output(
    output: DirectComposerOutput,
    *,
    policy: PolicyManifest,
    evidence_facts: list[EvidenceFact],
    adapter_client: DirectAdapterClient,
    action_history: list[ActionLedgerRecord],
) -> DirectMaterialization:
    if output.response_mode == "answer_company_info":
        allowed_ids = {
            evidence_id(fact)
            for fact in evidence_facts
            if trusted_document_context(fact)
        }
        if (
            "document_context" not in policy.allowed_capabilities
            or not output.evidence_ids
            or not set(output.evidence_ids).issubset(allowed_ids)
        ):
            return unable_direct_materialization()
    if (
        output.response_mode
        in {
            "prepare_outbound_message",
            "execute_prepared_outbound_message",
        }
        and "outbound_message" not in policy.allowed_capabilities
    ):
        return unable_direct_materialization("当前无法使用外发功能，请稍后再试。")
    return await materialize_direct_output(
        output,
        adapter_client=adapter_client,
        action_history=action_history,
    )


def unable_direct_materialization(
    text: str = "这个信息我暂时无法确认，先不回答避免信息不准确。",
) -> DirectMaterialization:
    return DirectMaterialization(
        mode="unable",
        response=ReplyResponse(
            reply=PrimaryReply(kind="unable_to_answer", text=text, mentions=[]),
            actions=[],
        ),
    )


def plan_for_direct_materialization(
    request: ReplyRequest,
    materialization: DirectMaterialization,
    response: ReplyResponse,
) -> ExecutionPlan:
    action_intents = [
        ActionIntentSpec(
            action_type=action.type,
            capability="outbound_message",
        )
        for action in response.actions
    ]
    return ExecutionPlan(
        user_need=request.message,
        artifact_kind=(
            "knowledge_answer"
            if materialization.mode == "knowledge_answer"
            else "multi_action"
            if materialization.mode == "action"
            else "unclear"
        ),
        response_mode=materialization.mode,
        compliance=ComplianceDecision(
            is_compliant=True,
            reason_code="compliant_product_request",
        ),
        capabilities=[materialization.capability]
        if materialization.capability is not None
        else [],
        answer_capabilities=["document_context"]
        if materialization.mode == "knowledge_answer"
        else [],
        action_intents=action_intents,
        confidence=1.0,
    )


def direct_answerability(
    materialization: DirectMaterialization,
    evidence_facts: list[EvidenceFact],
) -> AnswerabilityAssessment:
    allowed_ids = [
        evidence_id(fact) for fact in evidence_facts if trusted_document_context(fact)
    ]
    can_answer = materialization.mode in {"knowledge_answer", "action"}
    return AnswerabilityAssessment(
        can_answer=can_answer,
        capability_id=materialization.capability or "direct.unavailable",
        allowed_evidence_ids=allowed_ids,
        ambiguity="other" if materialization.mode == "clarification" else "none",
        recommended_response_mode=(
            "answer"
            if can_answer
            else "clarify"
            if materialization.mode == "clarification"
            else "abstain"
        ),
    )
