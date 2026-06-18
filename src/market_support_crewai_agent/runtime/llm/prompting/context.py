from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from market_support_crewai_agent.runtime.context.models import (
    ModelVisibleContext,
    prompt_json,
)
from market_support_crewai_agent.runtime.context.projection import ContextProjectionManager
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.state.conversation_store import ConversationMessage
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.llm.prompting.profiles import (
    ModelFamily,
    PromptStage,
)
from market_support_crewai_agent.runtime.llm.prompting.registry import PromptLayer
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse, StrictModel


class IntentGateResult(StrictModel):
    contract_version: Literal["intent-gate"] = "intent-gate"
    artifact_hint: Literal[
        "material_pack",
        "weekly_report",
        "monthly_report",
        "knowledge_answer",
        "human_support",
        "refusal",
        "unclear",
        "smalltalk",
    ]
    side_effect_hint: bool = False
    material_pack_option_count: int = 0
    compliance_hint: Literal["clean", "risky", "blocked", "unknown"] = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass(frozen=True)
class PromptAssemblyContext:
    stage: PromptStage
    model_family: ModelFamily
    request: ReplyRequest
    canonical_context: CanonicalContext
    policy: PolicyManifest
    model_visible_context: ModelVisibleContext | None = None
    domain_context: DomainContext | None = None
    intent_gate: IntentGateResult | None = None
    execution_plan: ExecutionPlan | None = None
    plan_validation: PlanValidationResult | None = None
    preflight: AdapterPreflightSnapshot | None = None
    evidence_facts: list[EvidenceFact] = field(default_factory=list)
    business_facts: BusinessFacts | None = None
    answerability_assessment: AnswerabilityAssessment | None = None
    guardrail_decisions: list[GuardrailDecision] = field(default_factory=list)
    history: list[ConversationMessage] = field(default_factory=list)
    action_history: list[ActionLedgerRecord] = field(default_factory=list)
    candidate_response: ReplyResponse | None = None
    alignment_verdict: ReplyAlignmentVerdict | None = None
    alignment_attempt: int = 0


def render_prompt_context(ctx: PromptAssemblyContext) -> str:
    return "\n\n".join(
        text for text in render_prompt_context_layers(ctx).values() if text.strip()
    )


def render_prompt_context_layers(ctx: PromptAssemblyContext) -> dict[PromptLayer, str]:
    projection = ctx.model_visible_context or ContextProjectionManager().project_for_stage(
        stage=ctx.stage,
        request=ctx.request,
        canonical_context=ctx.canonical_context,
        domain_context=ctx.domain_context,
        policy=ctx.policy,
        intent_gate=ctx.intent_gate,
        execution_plan=ctx.execution_plan,
        plan_validation=ctx.plan_validation,
        preflight=ctx.preflight,
        evidence_facts=ctx.evidence_facts,
        business_facts=ctx.business_facts,
        answerability_assessment=ctx.answerability_assessment,
        guardrail_decisions=ctx.guardrail_decisions,
        history=ctx.history,
        action_history=ctx.action_history,
        candidate_response=ctx.candidate_response,
        alignment_verdict=ctx.alignment_verdict,
        alignment_attempt=ctx.alignment_attempt,
    )
    return _render_projected_layers(projection)


def _render_projected_layers(projection: ModelVisibleContext) -> dict[PromptLayer, str]:
    runtime_payload = projection.to_prompt_runtime_payload()
    runtime_payload["blocks"] = [
        block.to_prompt_dict()
        for block in projection.blocks
        if block.block_type not in {"current_task", "ephemeral"}
    ]

    task_parts: list[str] = []
    ephemeral_parts: list[str] = []
    for block in projection.blocks:
        if block.block_type == "current_task":
            task_parts.append(_render_task_block(block.title, block.payload))
        elif block.block_type == "ephemeral":
            ephemeral_parts.append(_render_ephemeral_block(block.title, block.payload))

    return {
        "stable": "",
        "domain": "",
        "runtime": "Runtime Capability & Evidence Boundary JSON:\n{}".format(
            prompt_json(runtime_payload)
        ),
        "task": "\n\n".join(part for part in task_parts if part.strip()),
        "ephemeral": "\n\n".join(part for part in ephemeral_parts if part.strip()),
    }


def _render_task_block(title: str, payload) -> str:
    if title == "Current user message":
        message = payload.get("message", "") if isinstance(payload, dict) else str(payload)
        return "Current user message:\n{}".format(message)
    if title == "Candidate ReplyResponse JSON":
        return "Candidate ReplyResponse JSON:\n{}".format(prompt_json(payload))
    return "{}:\n{}".format(title, prompt_json(payload))


def _render_ephemeral_block(title: str, payload) -> str:
    if title == "Previous alignment verdict JSON":
        return "Previous alignment verdict JSON:\n{}".format(prompt_json(payload))
    if title == "Alignment attempt":
        attempt = payload.get("attempt", 0) if isinstance(payload, dict) else payload
        return f"Alignment attempt: {attempt}"
    return "{}:\n{}".format(title, prompt_json(payload))
