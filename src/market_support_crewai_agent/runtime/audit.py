from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Literal

from market_support_crewai_agent.runtime.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.adapter_preflight import AdapterPreflightSnapshot
from market_support_crewai_agent.runtime.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.canonicalization import (
    CanonicalContext,
    canonicalize_request,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.guardrails import ValidationResult
from market_support_crewai_agent.runtime.planning import PlanValidationResult, ReplyPlan
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse
from market_support_crewai_agent.settings import Settings

AuditTraceVersion = Literal["audit-trace.v1"]

AUDIT_TRACE_VERSION: AuditTraceVersion = "audit-trace.v1"
PLANNER_PROMPT_VERSION = "planner-prompt.v1"
COMPOSER_PROMPT_VERSION = "composer-prompt.v1"
PLAN_VALIDATOR_VERSION = "plan-validator.v1"
REPLY_VALIDATOR_VERSION = "reply-action-validator.v1"
BUSINESS_FACTS_VERSION = "business-facts.v1"


@dataclass(frozen=True)
class AuditTrace:
    """Replayable, adapter-safe trace for one /reply request."""

    trace_id: str
    contract_version: AuditTraceVersion
    created_at: datetime
    context_id: str | None
    conversation_key: str
    group_id: str
    sender_id: str
    request: dict
    model: dict
    llm_executions: list[dict]
    policy_id: str
    policy_hash: str
    policy: dict
    canonical_entities: dict
    planner_output: dict
    plan_validation: dict
    action_history: list[dict]
    adapter_preflight: list[dict]
    evidence_facts: list[dict]
    business_facts: dict
    reply_output: dict
    reply_validation: dict | None
    repair_attempts: list[dict]
    fallback_used: bool
    fallback_reason: str
    final_actions: list[dict]
    action_preconditions: list[dict]
    adapter_execution_status: str
    versions: dict

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["created_at"] = self.created_at.isoformat()
        return payload


class AuditStore:
    """Thread-safe in-memory audit trace store.

    This is deliberately bounded and internal-only. It gives tests, evals, and
    incident review a stable skeleton before durable storage is introduced.
    """

    def __init__(self, max_traces: int = 5000) -> None:
        if max_traces <= 0:
            raise ValueError("max_traces must be greater than zero")
        self._max_traces = max_traces
        self._traces: list[AuditTrace] = []
        self._lock = RLock()

    def record(self, trace: AuditTrace) -> AuditTrace:
        with self._lock:
            self._traces.append(trace)
            if len(self._traces) > self._max_traces:
                self._traces = self._traces[-self._max_traces:]
            return trace

    def recent_for_conversation(
            self,
            conversation_key: str,
            limit: int = 20,
    ) -> list[AuditTrace]:
        if limit <= 0:
            return []
        with self._lock:
            matches = [
                trace
                for trace in self._traces
                if trace.conversation_key == conversation_key
            ]
            return list(matches[-limit:])

    def by_context_id(self, context_id: str) -> list[AuditTrace]:
        with self._lock:
            return [
                trace
                for trace in self._traces
                if trace.context_id == context_id
            ]

    def latest(self) -> AuditTrace | None:
        with self._lock:
            if not self._traces:
                return None
            return self._traces[-1]

    def count(self) -> int:
        with self._lock:
            return len(self._traces)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


_DEFAULT_AUDIT_STORE = AuditStore()


def get_audit_store() -> AuditStore:
    return _DEFAULT_AUDIT_STORE


def build_audit_trace(
        *,
        request: ReplyRequest,
        settings: Settings,
        policy: PolicyManifest,
        plan: ReplyPlan,
        plan_validation: PlanValidationResult,
        action_history: list[ActionLedgerRecord],
        preflight: AdapterPreflightSnapshot,
        evidence_facts: list[EvidenceFact],
        business_facts: BusinessFacts,
        response: ReplyResponse,
        reply_validation: ValidationResult | None,
        fallback_used: bool,
        canonical_context: CanonicalContext | None = None,
        repair_attempts: list[dict] | None = None,
        llm_executions: list[dict] | None = None,
) -> AuditTrace:
    policy_payload = _compact_policy(policy)
    canonical_payload = _compact_canonical_context(
        canonical_context or canonicalize_request(request)
    )
    reply_validation_payload = (
        _compact_reply_validation(reply_validation)
        if reply_validation is not None
        else None
    )
    return AuditTrace(
        trace_id=f"trace-{uuid.uuid4().hex}",
        contract_version=AUDIT_TRACE_VERSION,
        created_at=datetime.now(timezone.utc),
        context_id=request.context_id,
        conversation_key=request.conversation_key,
        group_id=request.group_id,
        sender_id=request.sender_id,
        request=_compact_request(request),
        model=_compact_model(settings),
        llm_executions=llm_executions or [],
        policy_id=policy.policy_id,
        policy_hash=_stable_hash(policy_payload),
        policy=policy_payload,
        canonical_entities=canonical_payload,
        planner_output=plan.model_dump(mode="json", exclude_none=True),
        plan_validation=_compact_plan_validation(plan_validation),
        action_history=[
            _compact_action_record(record)
            for record in action_history
        ],
        adapter_preflight=_compact_preflight(preflight),
        evidence_facts=[
            _compact_evidence_fact(fact)
            for fact in evidence_facts
        ],
        business_facts=business_facts.to_prompt_dict(),
        reply_output=response.model_dump(mode="json", exclude_none=True),
        reply_validation=reply_validation_payload,
        repair_attempts=repair_attempts or [],
        fallback_used=fallback_used,
        fallback_reason=_fallback_reason(reply_validation_payload, fallback_used),
        final_actions=[
            action.model_dump(mode="json", exclude_none=True)
            for action in response.actions
        ],
        action_preconditions=_compact_action_preconditions(response, plan, preflight),
        adapter_execution_status=(
            "pending_adapter_execution" if response.actions else "no_actions"
        ),
        versions={
            "audit_trace": AUDIT_TRACE_VERSION,
            "planner_prompt": PLANNER_PROMPT_VERSION,
            "composer_prompt": COMPOSER_PROMPT_VERSION,
            "policy": policy.policy_id.split(":", 1)[0],
            "plan_validator": PLAN_VALIDATOR_VERSION,
            "reply_validator": REPLY_VALIDATOR_VERSION,
            "business_facts": BUSINESS_FACTS_VERSION,
        },
    )


def _compact_request(request: ReplyRequest) -> dict:
    return {
        "context_id": request.context_id,
        "conversation_key": request.conversation_key,
        "group_id": request.group_id,
        "sender_id": request.sender_id,
        "message": request.message,
        "is_group": request.is_group,
        "group_name": request.group_name,
        "dist_channel_name": request.dist_channel_name,
        "sender_nickname": request.sender_nickname,
        "available_materials": list(request.available_materials),
        "available_strategies": list(request.available_strategies),
        "channel_type": request.channel_type,
    }


def _compact_model(settings: Settings) -> dict:
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }


def _compact_canonical_context(canonical_context: CanonicalContext) -> dict:
    return canonical_context.to_prompt_dict()


def _compact_policy(policy: PolicyManifest) -> dict:
    return {
        "policy_id": policy.policy_id,
        "allowed_reply_kinds": sorted(policy.allowed_reply_kinds),
        "allowed_side_effect_actions": sorted(policy.allowed_side_effect_actions),
        "required_adapter_resolves": sorted(policy.required_adapter_resolves),
        "allowed_read_capabilities": sorted(policy.allowed_read_capabilities),
        "allowed_business_checks": sorted(policy.allowed_business_checks),
        "forbidden_claim_categories": sorted(policy.forbidden_claim_categories),
        "ledger_summary": policy.ledger_summary.to_prompt_dict(),
        "evidence_call_limit": policy.evidence_call_limit,
        "repair_policy": {
            "allow_repair": policy.repair_policy.allow_repair,
            "max_attempts": policy.repair_policy.max_attempts,
            "fallback_reply_kind": policy.repair_policy.fallback_reply_kind,
        },
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


def _compact_reply_validation(validation: ValidationResult) -> dict:
    return {
        "valid": validation.valid,
        "severity": validation.severity,
        "repairable": validation.repairable,
        "fallback_reply_kind": validation.fallback_reply_kind,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "severity": issue.severity,
                "repairable": issue.repairable,
                "fallback_reply_kind": issue.fallback_reply_kind,
                "metadata": issue.metadata,
            }
            for issue in validation.issues
        ],
    }


def _compact_action_record(record: ActionLedgerRecord) -> dict:
    execution = record.execution
    return {
        "context_id": record.context_id,
        "response_id": record.response_id,
        "action_id": execution.action_id,
        "action_type": execution.action_type,
        "status": execution.status,
        "material_type": execution.material_type,
        "strategy": execution.strategy,
        "version": execution.version,
        "received_at": record.received_at.isoformat(),
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
                "contains_strategy": result.contains_strategy,
                "generated_strategies": result.generated_strategies,
                "scope_status": result.scope_status,
            }
        )
    return items


def _compact_action_preconditions(
        response: ReplyResponse,
        plan: ReplyPlan,
        preflight: AdapterPreflightSnapshot,
) -> list[dict]:
    return [
        _compact_action_precondition(action, index, plan, preflight)
        for index, action in enumerate(response.actions, start=1)
    ]


def _compact_action_precondition(
        action,
        index: int,
        plan: ReplyPlan,
        preflight: AdapterPreflightSnapshot,
) -> dict:
    resolve_type = _resolve_type_for_action(action.type)
    candidate = _matching_plan_candidate(action, plan)
    item = _preflight_item(preflight, resolve_type)
    result = item.result if item is not None else None
    return {
        "action_index": index,
        "action_id": action.action_id,
        "action_type": action.type,
        "resolve_type": resolve_type,
        "resolve_status": item.status if item is not None else "missing_preflight",
        "plan_report_scope": getattr(candidate, "report_scope", None),
        "plan_strategy": getattr(candidate, "strategy", None),
        "action_strategy": getattr(action, "strategy", None),
        "adapter_strategy": result.strategy if result is not None else None,
        "adapter_ref_available": bool(result.card_ref) if result is not None else False,
        "contains_strategy": (
            result.contains_strategy if result is not None else None
        ),
        "scope_status": result.scope_status if result is not None else None,
        "period": result.period if result is not None else None,
        "report_date": result.report_date if result is not None else None,
    }


def _resolve_type_for_action(action_type: str) -> str:
    if action_type == "send_material_pack":
        return "material_pack"
    if action_type == "send_weekly_report":
        return "weekly_report"
    if action_type == "send_monthly_report":
        return "monthly_report"
    return ""


def _matching_plan_candidate(action, plan: ReplyPlan):
    for candidate in plan.candidate_actions:
        if candidate.type != action.type:
            continue
        candidate_strategy = getattr(candidate, "strategy", None)
        action_strategy = getattr(action, "strategy", None)
        if candidate_strategy and action_strategy and candidate_strategy != action_strategy:
            continue
        return candidate
    return None


def _preflight_item(
        preflight: AdapterPreflightSnapshot,
        resolve_type: str,
):
    for item in preflight.items:
        if item.resolve_type == resolve_type:
            return item
    return None


def _compact_evidence_fact(fact: EvidenceFact) -> dict:
    return {
        "fact_type": fact.fact_type,
        "value": fact.value,
        "source_type": fact.source_type,
        "source_id": fact.source_id,
        "resolve_type": fact.resolve_type,
        "metadata": dict(fact.metadata),
    }


def _fallback_reason(
        reply_validation_payload: dict | None,
        fallback_used: bool,
) -> str:
    if not fallback_used:
        return ""
    if not reply_validation_payload:
        return "plan_or_contract_fallback"
    issues = reply_validation_payload.get("issues") or []
    if not issues:
        return "reply_guardrail_fallback"
    return str(issues[0].get("code") or "reply_guardrail_fallback")


def _stable_hash(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
