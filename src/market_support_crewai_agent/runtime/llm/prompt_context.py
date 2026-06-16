from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.state.conversation_store import ConversationMessage
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.llm.prompt_profiles import (
    ModelFamily,
    PromptStage,
)
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
    named_strategy_count: int = 0
    compliance_hint: Literal["clean", "risky", "blocked", "unknown"] = "unknown"
    matched_keywords: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass(frozen=True)
class PromptAssemblyContext:
    stage: PromptStage
    model_family: ModelFamily
    request: ReplyRequest
    canonical_context: CanonicalContext
    policy: PolicyManifest
    intent_gate: IntentGateResult | None = None
    execution_plan: ExecutionPlan | None = None
    plan_validation: PlanValidationResult | None = None
    preflight: AdapterPreflightSnapshot | None = None
    evidence_facts: list[EvidenceFact] = field(default_factory=list)
    business_facts: BusinessFacts | None = None
    history: list[ConversationMessage] = field(default_factory=list)
    action_history: list[ActionLedgerRecord] = field(default_factory=list)
    candidate_response: ReplyResponse | None = None
    alignment_verdict: ReplyAlignmentVerdict | None = None
    alignment_attempt: int = 0


def render_prompt_context(ctx: PromptAssemblyContext) -> str:
    parts = [
        "Request metadata JSON:\n{}".format(
            _json(
                ctx.request.model_dump(
                    mode="json",
                    exclude={"message"},
                    exclude_none=True,
                )
            )
        ),
        "Canonical entities JSON:\n{}".format(
            _json(ctx.canonical_context.to_prompt_dict())
        ),
    ]
    if ctx.intent_gate is not None:
        parts.append(
            "IntentGate JSON:\n{}".format(
                _json(ctx.intent_gate.model_dump(mode="json", exclude_none=True))
            )
        )
    parts.extend(
        [
            "Policy JSON:\n{}".format(_json(_compact_policy(ctx.policy))),
            "Recent turns JSON:\n{}".format(_json(_compact_history(ctx.history))),
            "Recent executed adapter actions JSON:\n{}".format(
                _json(
                    [
                        _compact_action_record(record)
                        for record in ctx.action_history
                    ]
                )
            ),
        ]
    )
    if ctx.execution_plan is not None:
        parts.append(
            "ExecutionPlan JSON:\n{}".format(
                _json(ctx.execution_plan.model_dump(mode="json", exclude_none=True))
            )
        )
    if ctx.plan_validation is not None:
        parts.append(
            "Plan validation JSON:\n{}".format(
                _json(_compact_plan_validation(ctx.plan_validation))
            )
        )
    if ctx.preflight is not None:
        parts.append(
            "Adapter preflight JSON:\n{}".format(_json(_compact_preflight(ctx.preflight)))
        )
    if ctx.evidence_facts:
        parts.append(
            "EvidenceFacts JSON:\n{}".format(
                _json([_compact_evidence_fact(fact) for fact in ctx.evidence_facts])
            )
        )
    if ctx.business_facts is not None:
        parts.append(
            "BusinessFacts JSON:\n{}".format(_json(ctx.business_facts.to_prompt_dict()))
        )
    if ctx.alignment_verdict is not None:
        parts.append(
            "Previous alignment verdict JSON:\n{}".format(
                _json(ctx.alignment_verdict.model_dump(mode="json", exclude_none=True))
            )
        )
    if ctx.candidate_response is not None:
        parts.append(
            "Candidate ReplyResponse JSON:\n{}".format(
                _json(_compact_response(ctx.candidate_response))
            )
        )
    if ctx.alignment_attempt:
        parts.append(f"Alignment attempt: {ctx.alignment_attempt}")
    parts.append("Current user message:\n{}".format(ctx.request.message))
    return "\n\n".join(parts)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _compact_history(history: list[ConversationMessage]) -> list[dict]:
    return [
        {
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        for message in history
    ]


def _compact_action_record(record: ActionLedgerRecord) -> dict:
    execution = record.execution
    return {
        "context_id": record.context_id,
        "response_id": record.response_id,
        "action_id": execution.action_id,
        "action_type": execution.action_type,
        "status": execution.status,
        "resolve_ref_available": bool(execution.resolve_ref),
        "material_type": execution.material_type,
        "strategy": execution.strategy,
        "version": execution.version,
        "received_at": record.received_at.isoformat(),
    }


def _compact_policy(policy: PolicyManifest) -> dict:
    return {
        "policy_id": policy.policy_id,
        "allowed_reply_modes": sorted(policy.allowed_reply_modes),
        "allowed_capabilities": sorted(policy.allowed_capabilities),
        "allowed_side_effect_actions": sorted(policy.allowed_side_effect_actions),
        "allowed_read_capabilities": sorted(policy.allowed_read_capabilities),
        "allowed_adapter_resolves": sorted(policy.allowed_adapter_resolves),
        "ledger_summary": policy.ledger_summary.to_prompt_dict(),
        "evidence_call_limit": policy.evidence_call_limit,
    }


def _compact_plan_validation(validation: PlanValidationResult) -> dict:
    return {
        "valid": validation.valid,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "severity": issue.severity,
                "metadata": issue.metadata,
            }
            for issue in validation.issues
        ],
    }


def _compact_preflight(preflight: AdapterPreflightSnapshot) -> list[dict]:
    items = []
    for item in preflight.items:
        if item.result is None:
            items.append(
                {
                    "resolve_type": item.resolve_type,
                    "status": item.status,
                    "error": item.error,
                }
            )
            continue

        result = item.result
        items.append(
            {
                "resolve_type": result.resolve_type,
                "status": result.status,
                "display_name": result.display_name,
                "reason_code": result.reason_code,
                "candidates": result.candidates,
                "channel_type": result.channel_type,
                "available_materials": result.available_materials,
                "available_strategies": result.available_strategies,
                "strategy": result.strategy,
                "period": result.period,
                "report_date": result.report_date,
                "period_start": result.period_start,
                "period_end": result.period_end,
                "period_label": result.period_label,
                "contains_strategy": result.contains_strategy,
                "generated_strategies": result.generated_strategies,
                "scope_status": result.scope_status,
                "scope_complete": result.scope_complete,
                "expected_product_count": result.expected_product_count,
                "generated_product_count": result.generated_product_count,
                "missing_product_count": result.missing_product_count,
                "report_sections": [
                    section.model_dump(mode="json", exclude_none=True)
                    for section in result.report_sections
                ],
            }
        )
    return items


def _compact_evidence_fact(fact: EvidenceFact) -> dict:
    metadata = dict(fact.metadata)
    if "resolve_ref" in metadata:
        metadata["resolve_ref_available"] = bool(metadata.pop("resolve_ref"))
    return {
        "fact_type": fact.fact_type,
        "value": fact.value,
        "source_type": fact.source_type,
        "source_id": fact.source_id,
        "resolve_type": fact.resolve_type,
        "metadata": metadata,
    }


def _compact_response(response: ReplyResponse) -> dict:
    payload = response.model_dump(mode="json", exclude_none=True)
    for action in payload.get("actions", []):
        if "resolve_ref" in action:
            action["resolve_ref_available"] = bool(action.pop("resolve_ref"))
    return payload
