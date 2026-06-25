from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Literal

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.runtime_trace import RUNTIME_TRACE_VERSION
from market_support_crewai_agent.runtime.evidence.adapter_preflight import AdapterPreflightSnapshot
from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.sources.metadata import (
    SourceMetadata,
    source_metadata_prompt_dict,
)
from market_support_crewai_agent.runtime.validation.reply_validator import ValidationResult
from market_support_crewai_agent.runtime.validation.guardrail_types import GuardrailDecision
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan, PlanValidationResult
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.domain.capabilities import (
    capability_registry_hash,
    resolve_type_for_action,
)
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.context import IntentGateResult
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse
from market_support_crewai_agent.settings import Settings

AuditTraceVersion = Literal["audit-trace"]

AUDIT_TRACE_VERSION: AuditTraceVersion = "audit-trace"
PLAN_VALIDATOR_VERSION = "plan-validator"
REPLY_VALIDATOR_VERSION = "reply-action-validator"
BUSINESS_FACTS_VERSION = "business-facts"
PROMPT_PROGRAM_SCHEMA_VERSION = "prompt-program"


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
    intent_gate: dict
    prompt_programs: list[dict]
    llm_executions: list[dict]
    policy_id: str
    policy_hash: str
    policy: dict
    policy_scope: dict
    domain_context: dict
    planner_output: dict
    response_directive: dict
    plan_validation: dict
    action_history: list[dict]
    adapter_preflight: list[dict]
    evidence_facts: list[dict]
    business_facts: dict
    answerability_assessment: dict
    reply_output: dict
    reply_validation: dict | None
    guardrail_decisions: list[dict]
    alignment_verdicts: list[dict]
    alignment_remediations: list[dict]
    final_actions: list[dict]
    action_preconditions: list[dict]
    adapter_execution_status: str
    versions: dict
    runtime_trace: dict = field(default_factory=dict)

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
        plan: ExecutionPlan,
        directive: ResponseDirective,
        plan_validation: PlanValidationResult,
        action_history: list[ActionLedgerRecord],
        preflight: AdapterPreflightSnapshot,
        evidence_facts: list[EvidenceFact],
        business_facts: BusinessFacts,
        response: ReplyResponse,
        reply_validation: ValidationResult | None,
        guardrail_decisions: list[GuardrailDecision] | None = None,
        domain_context: DomainContext | None = None,
        intent_gate: IntentGateResult | None = None,
        prompt_programs: list[PromptProgram] | None = None,
        llm_executions: list[dict] | None = None,
        alignment_verdicts: list[ReplyAlignmentVerdict] | None = None,
        alignment_remediations: list[dict] | None = None,
        answerability_assessment: AnswerabilityAssessment | None = None,
        runtime_trace: dict | None = None,
) -> AuditTrace:
    policy_payload = _compact_policy(policy)
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
        intent_gate=(
            intent_gate.model_dump(mode="json", exclude_none=True)
            if intent_gate is not None
            else {}
        ),
        prompt_programs=[
            _compact_prompt_program(program)
            for program in prompt_programs or []
        ],
        llm_executions=llm_executions or [],
        policy_id=policy.policy_id,
        policy_hash=_stable_hash(policy_payload),
        policy=policy_payload,
        policy_scope={"material_pack_options": list(policy.material_pack_options)},
        domain_context=(
            domain_context.to_prompt_dict()
            if domain_context is not None
            else {}
        ),
        planner_output=plan.model_dump(mode="json", exclude_none=True),
        response_directive=_compact_response_directive(directive),
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
        answerability_assessment=(
            answerability_assessment.model_dump(mode="json", exclude_none=True)
            if answerability_assessment is not None
            else {}
        ),
        reply_output=_compact_response(response),
        reply_validation=reply_validation_payload,
        guardrail_decisions=[
            _compact_guardrail_decision(decision)
            for decision in guardrail_decisions or []
        ],
        alignment_verdicts=[
            _compact_alignment_verdict(verdict)
            for verdict in alignment_verdicts or []
        ],
        alignment_remediations=[
            _compact_alignment_remediation(item)
            for item in alignment_remediations or []
        ],
        final_actions=[
            _compact_action(action)
            for action in response.actions
        ],
        action_preconditions=_compact_action_preconditions(response, plan, preflight),
        adapter_execution_status=(
            "pending_adapter_execution" if response.actions else "no_actions"
        ),
        versions={
            "audit_trace": AUDIT_TRACE_VERSION,
            "runtime_trace": RUNTIME_TRACE_VERSION,
            "prompt_program_schema": PROMPT_PROGRAM_SCHEMA_VERSION,
            "prompt_profile_ids": _prompt_profile_ids(llm_executions or []),
            "capability_registry_hash": capability_registry_hash(),
            "adapter_contract": "adapter-resolve",
            "action_contract": "adapter-action",
            "policy": policy.policy_id.split(":", 1)[0],
            "plan_validator": PLAN_VALIDATOR_VERSION,
            "reply_validator": REPLY_VALIDATOR_VERSION,
            "business_facts": BUSINESS_FACTS_VERSION,
        },
        runtime_trace=runtime_trace or {},
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
        "available_artifacts": [
            artifact.model_dump(mode="json", exclude_none=True)
            for artifact in request.available_artifacts
        ],
        "channel_type": request.channel_type,
    }


def _compact_response(response: ReplyResponse) -> dict:
    payload = response.model_dump(mode="json", exclude_none=True)
    for action in payload.get("actions", []):
        if "resolve_ref" in action:
            action["resolve_ref_available"] = bool(action.pop("resolve_ref"))
    return payload


def _compact_response_directive(directive: ResponseDirective) -> dict:
    return directive.model_dump(mode="json", exclude_none=True)


def _compact_action(action) -> dict:
    payload = action.model_dump(mode="json", exclude_none=True)
    if "resolve_ref" in payload:
        payload["resolve_ref_available"] = bool(payload.pop("resolve_ref"))
    return payload


def _compact_model(settings: Settings) -> dict:
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "planner_provider": settings.planner_llm_provider,
        "planner_model": settings.planner_llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }


def _compact_prompt_program(program: PromptProgram) -> dict:
    payload = {
        "stage": program.profile.stage,
        "profile_id": program.profile.id,
        "model_family": program.profile.model_family,
        "fragment_ids": list(program.fragment_ids),
        "layers": list(program.layers),
        "prompt_hash": program.prompt_hash,
        "fragment_hashes": dict(program.fragment_hashes),
    }
    if program.projection_id:
        payload.update(
            {
                "projection_id": program.projection_id,
                "projection_pressure": program.projection_pressure,
                "projection_decision_count": program.projection_decision_count,
                "model_visible_context_hash": program.model_visible_context_hash,
            }
        )
    return payload


def _compact_policy(policy: PolicyManifest) -> dict:
    return {
        "policy_id": policy.policy_id,
        "allowed_reply_modes": sorted(policy.allowed_reply_modes),
        "allowed_capabilities": sorted(policy.allowed_capabilities),
        "allowed_outbound_actions": sorted(policy.allowed_outbound_actions),
        "allowed_read_capabilities": sorted(policy.allowed_read_capabilities),
        "allowed_adapter_resolves": sorted(policy.allowed_adapter_resolves),
        "material_pack_options": list(policy.material_pack_options),
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


def _compact_reply_validation(validation: ValidationResult) -> dict:
    return {
        "valid": validation.valid,
        "severity": validation.severity,
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


def _compact_guardrail_decision(decision: GuardrailDecision) -> dict:
    payload = decision.model_dump(mode="json", exclude_none=True)
    metadata = dict(payload.get("metadata") or {})
    if "resolve_ref" in metadata:
        metadata["resolve_ref_available"] = bool(metadata.pop("resolve_ref"))
    payload["metadata"] = metadata
    return payload


def _compact_alignment_verdict(verdict: ReplyAlignmentVerdict) -> dict:
    return verdict.model_dump(mode="json", exclude_none=True)


def _compact_alignment_remediation(item: dict) -> dict:
    output: dict = {}
    for key, value in item.items():
        if key == "resolve_ref":
            output["resolve_ref_available"] = bool(value)
            continue
        if str(key).lower().endswith(("secret", "api_key", "token")):
            continue
        output[str(key)] = value
    return output


def _prompt_profile_ids(llm_executions: list[dict]) -> list[str]:
    return sorted(
        {
            str(execution.get("prompt_profile_id") or "")
            for execution in llm_executions
            if execution.get("prompt_profile_id")
        }
    )


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
        "material_pack_option": execution.material_pack_option,
        "version": execution.version,
        "received_at": record.received_at.isoformat(),
        "source_metadata": SourceMetadata(
            source_id=record.response_id or record.context_id or execution.action_id,
            source_type="tool_result",
            artifact_type="history",
            channel_id=record.group_id,
            created_at=record.received_at,
            observed_at=record.received_at,
            provenance="adapter_action_ledger",
            evidence_allowed_by_default=False,
        ).to_prompt_dict(),
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
                "available_artifacts": [
                    artifact.model_dump(mode="json", exclude_none=True)
                    for artifact in result.available_artifacts
                ],
                "material_pack_option": result.material_pack_option,
                "period": result.period,
                "report_date": result.report_date,
            }
        )
    return items


def _compact_action_preconditions(
        response: ReplyResponse,
        plan: ExecutionPlan,
        preflight: AdapterPreflightSnapshot,
) -> list[dict]:
    return [
        _compact_action_precondition(action, index, plan, preflight)
        for index, action in enumerate(response.actions, start=1)
    ]


def _compact_action_precondition(
        action,
        index: int,
        plan: ExecutionPlan,
        preflight: AdapterPreflightSnapshot,
) -> dict:
    resolve_type = resolve_type_for_action(action.type) or ""
    candidate = _matching_plan_candidate(action, plan)
    item = _preflight_item(preflight, resolve_type)
    result = item.result if item is not None else None
    return {
        "action_index": index,
        "action_id": action.action_id,
        "action_type": action.type,
        "resolve_type": resolve_type,
        "resolve_status": item.status if item is not None else "missing_preflight",
        "plan_material_pack_option": getattr(candidate, "material_pack_option", None),
        "action_material_pack_option": getattr(action, "material_pack_option", None),
        "adapter_material_pack_option": (
            result.material_pack_option if result is not None else None
        ),
        "adapter_ref_available": bool(result.resolve_ref) if result is not None else False,
        "action_ref_available": bool(getattr(action, "resolve_ref", None)),
        "period": result.period if result is not None else None,
        "report_date": result.report_date if result is not None else None,
    }


def _matching_plan_candidate(action, plan: ExecutionPlan):
    for candidate in plan.action_intents:
        if candidate.action_type != action.type:
            continue
        candidate_option = getattr(candidate, "material_pack_option", None)
        action_option = getattr(action, "material_pack_option", None)
        if candidate_option and action_option and candidate_option != action_option:
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
    metadata = dict(fact.metadata)
    if "resolve_ref" in metadata:
        metadata["resolve_ref_available"] = bool(metadata.pop("resolve_ref"))
    metadata = {
        key: _compact_large_audit_value(value)
        for key, value in metadata.items()
    }
    return {
        "evidence_id": ":".join(
            item
            for item in (fact.source_type, fact.source_id, fact.fact_type)
            if item
        ),
        "fact_type": fact.fact_type,
        "value": _compact_large_audit_value(fact.value),
        "source_type": fact.source_type,
        "source_id": fact.source_id,
        "resolve_type": fact.resolve_type,
        "artifact_type": fact.artifact_type,
        "scope": fact.scope.to_prompt_dict(),
        "source_metadata": source_metadata_prompt_dict(fact.source_metadata),
        "metadata": metadata,
    }


def _compact_large_audit_value(value):
    if isinstance(value, str) and len(value) > 1000:
        return {
            "preview": value[:200],
            "original_char_count": len(value),
            "truncated_for_audit": True,
        }
    if isinstance(value, dict):
        return {
            str(key): _compact_large_audit_value(item)
            for key, item in value.items()
            if not str(key).lower().endswith(("secret", "token", "api_key"))
        }
    if isinstance(value, list):
        return [_compact_large_audit_value(item) for item in value[:50]]
    return value

def _stable_hash(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
