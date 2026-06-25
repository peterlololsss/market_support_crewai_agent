from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from typing import Any

from market_support_crewai_agent.runtime.context.models import (
    CompactedSpanSummary,
    ContextBlock,
    ContextBlockType,
    ContextProjectionPolicy,
    LargeResultPreview,
    ModelVisibleContext,
    ProjectionDecision,
    RuntimeAppState,
    stable_json,
)
from market_support_crewai_agent.runtime.context.payload_store import ContextPayloadStore
from market_support_crewai_agent.runtime.context.pressure import (
    ContextPressureEstimator,
    ProjectionLimitError,
)
from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.capabilities.adapters import (
    planner_capability_cards,
)
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.domain.sources.metadata import (
    SourceMetadata,
    source_metadata_prompt_dict,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.conversation_store import ConversationMessage
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    select_evidence_for_plan,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    evidence_id,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    EvidenceSelection,
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse

_LARGE_FIELD_NAMES = {"text", "content", "body", "document_text", "stdout", "stderr", "raw"}
# Fact types whose value is the answer material the composer must read in full
# (bounded upstream by the Document MCP per-document cap), rather than incidental
# evidence that can be previewed.
_ANSWER_EVIDENCE_FACT_TYPES = frozenset({"document_context"})


class ContextProjectionManager:
    def __init__(
        self,
        policy: ContextProjectionPolicy | None = None,
        payload_store: ContextPayloadStore | None = None,
        pressure_estimator: ContextPressureEstimator | None = None,
    ) -> None:
        self.policy = policy or ContextProjectionPolicy()
        self.payload_store = payload_store or ContextPayloadStore()
        self.pressure_estimator = pressure_estimator or ContextPressureEstimator()

    @classmethod
    def from_settings(cls, settings: object | None = None) -> "ContextProjectionManager":
        return cls(policy=ContextProjectionPolicy.from_settings(settings))

    def project_for_planner(self, **kwargs: Any) -> ModelVisibleContext:
        return self.project_for_stage(stage="planner_intent", **kwargs)

    def project_for_composer(self, **kwargs: Any) -> ModelVisibleContext:
        return self.project_for_stage(stage=kwargs.pop("stage", "knowledge_composer"), **kwargs)

    def project_for_verifier(self, **kwargs: Any) -> ModelVisibleContext:
        return self.project_for_stage(stage="alignment_verifier", **kwargs)

    def project_for_stage(
        self,
        *,
        stage: str,
        request: ReplyRequest,
        policy: PolicyManifest,
        domain_context: DomainContext | None = None,
        intent_gate: object | None = None,
        execution_plan: ExecutionPlan | None = None,
        plan_validation: PlanValidationResult | None = None,
        preflight: AdapterPreflightSnapshot | None = None,
        evidence_facts: list[EvidenceFact] | None = None,
        business_facts: BusinessFacts | None = None,
        answerability_assessment: AnswerabilityAssessment | None = None,
        guardrail_decisions: list[GuardrailDecision] | None = None,
        history: list[ConversationMessage] | None = None,
        action_history: list[ActionLedgerRecord] | None = None,
        candidate_response: ReplyResponse | None = None,
        alignment_verdict: ReplyAlignmentVerdict | None = None,
        alignment_attempt: int = 0,
    ) -> ModelVisibleContext:
        history = list(history or [])
        action_history = list(action_history or [])
        evidence_facts = list(evidence_facts or [])
        guardrail_decisions = list(guardrail_decisions or [])
        decisions: list[ProjectionDecision] = []
        blocks: list[ContextBlock] = []

        app_state = RuntimeAppState(
            stage=stage,
            request_metadata=request.model_dump(
                mode="json",
                exclude={"message"},
                exclude_none=True,
            ),
            current_user_message=request.message,
            domain_context=_compact_domain_context(domain_context),
            policy=_compact_policy(policy),
            intent_gate=_model_dump(intent_gate),
            execution_plan=_model_dump(execution_plan),
            plan_validation=_compact_plan_validation(plan_validation),
            preflight=_compact_preflight(preflight) if preflight else None,
            business_facts=business_facts.to_prompt_dict() if business_facts else None,
            answerability_assessment=(
                answerability_assessment.model_dump(mode="json", exclude_none=True)
                if answerability_assessment is not None and self.policy.preserve_answerability
                else None
            ),
            guardrail_decisions=_compact_guardrail_decisions(guardrail_decisions),
            candidate_response=_compact_response(candidate_response),
            alignment_attempt=alignment_attempt,
            alignment_verdict=_model_dump(alignment_verdict),
        )
        app_payload = _app_state_payload(app_state)
        if stage == "planner_intent":
            app_payload["Capability registry JSON"] = _compact_planner_capabilities(
                request,
                policy,
            )
        blocks.append(
            self._block(
                "app_state",
                "Runtime app state",
                app_payload,
                included_reason="preserve_runtime_state",
            )
        )
        decisions.append(
            ProjectionDecision(
                source_id="runtime_app_state",
                decision="include",
                block_type="app_state",
                reason="preserve_runtime_state",
                projected_char_count=len(stable_json(app_payload)),
            )
        )

        pending = _pending_clarification_from_history(history)
        if stage == "planner_intent" and pending is not None:
            blocks.append(
                self._block(
                    "pending_clarification",
                    "Pending clarification context JSON",
                    pending,
                    block_type="context_only",
                    source_ids=["conversation:pending_clarification"],
                    included_reason="prior_clarification_pending_user_answer",
                )
            )
            decisions.append(
                ProjectionDecision(
                    source_id="conversation:pending_clarification",
                    decision="include",
                    block_type="context_only",
                    reason="prior_clarification_pending_user_answer",
                    projected_char_count=len(stable_json(pending)),
                )
            )

        history_blocks, history_decisions = self._project_history(history)
        blocks.extend(history_blocks)
        decisions.extend(history_decisions)

        if action_history:
            for record in action_history:
                decisions.append(
                    ProjectionDecision(
                        source_id=_action_source_id(record),
                        decision="include",
                        block_type="context_only",
                        reason="adapter_confirmed_action_summary_only",
                    )
                )
            blocks.append(
                self._block(
                    "action_history",
                    "Adapter action history context JSON",
                    [_compact_action_record(record) for record in action_history],
                    block_type="context_only",
                    source_ids=[_action_source_id(record) for record in action_history],
                    included_reason="adapter_confirmed_action_summary_only",
                )
            )

        evidence_blocks, evidence_decisions, allowed_ids, disallowed_ids = self._project_evidence(
            evidence_facts=evidence_facts,
            execution_plan=execution_plan,
            policy=policy,
            domain_context=domain_context,
            answerability_assessment=answerability_assessment,
        )
        blocks.extend(evidence_blocks)
        decisions.extend(evidence_decisions)

        if candidate_response is not None:
            payload = _compact_response(candidate_response)
            blocks.append(
                self._block(
                    "candidate_response",
                    "Candidate ReplyResponse JSON",
                    payload,
                    block_type="current_task",
                    included_reason="verifier_candidate_response",
                )
            )
            decisions.append(
                ProjectionDecision(
                    source_id="candidate_response",
                    decision="include",
                    block_type="current_task",
                    reason="verifier_candidate_response",
                    projected_char_count=len(stable_json(payload)),
                )
            )

        if self.policy.preserve_current_user_message:
            payload = {"message": request.message}
            blocks.append(
                self._block(
                    "current_user_message",
                    "Current user message",
                    payload,
                    block_type="current_task",
                    source_ids=["current_user_message"],
                    included_reason="preserve_current_user_message",
                )
            )
            decisions.append(
                ProjectionDecision(
                    source_id="current_user_message",
                    decision="include",
                    block_type="current_task",
                    reason="preserve_current_user_message",
                    original_char_count=len(request.message),
                    projected_char_count=len(request.message),
                )
            )

        if alignment_verdict is not None:
            payload = alignment_verdict.model_dump(mode="json", exclude_none=True)
            blocks.append(
                self._block(
                    "alignment_verdict",
                    "Previous alignment verdict JSON",
                    payload,
                    block_type="ephemeral",
                    included_reason="alignment_retry_feedback",
                )
            )
            decisions.append(
                ProjectionDecision(
                    source_id="alignment_verdict",
                    decision="include",
                    block_type="ephemeral",
                    reason="alignment_retry_feedback",
                    projected_char_count=len(stable_json(payload)),
                )
            )
        if alignment_attempt:
            blocks.append(
                self._block(
                    "alignment_attempt",
                    "Alignment attempt",
                    {"attempt": alignment_attempt},
                    block_type="ephemeral",
                    included_reason="alignment_retry_feedback",
                )
            )

        raw_context = _raw_context(
            request=request,
            history=history,
            action_history=action_history,
            domain_context=domain_context,
            policy=policy,
            intent_gate=intent_gate,
            execution_plan=execution_plan,
            plan_validation=plan_validation,
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            answerability_assessment=answerability_assessment,
            guardrail_decisions=guardrail_decisions,
            candidate_response=candidate_response,
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
        )
        metadata = {
            "policy": self.policy.__dict__,
            "block_count": len(blocks),
        }
        shell = ModelVisibleContext(
            projection_id="pending",
            stage=stage,
            blocks=blocks,
            allowed_evidence_ids=allowed_ids,
            disallowed_evidence_ids=disallowed_ids,
            context_only_source_ids=_context_only_source_ids(blocks),
            decisions=decisions,
            metadata=metadata,
        )
        pressure = self.pressure_estimator.estimate(
            raw_context=raw_context,
            projected_context=shell.to_prompt_runtime_payload(),
            policy=self.policy,
        )
        projection = replace(
            shell,
            projection_id=_projection_id(shell.to_prompt_runtime_payload()),
            pressure=pressure,
        )
        if pressure.hard_blocked:
            raise ProjectionLimitError(
                "projected context exceeds hard threshold: {:.2f}".format(
                    pressure.pressure_ratio
                )
            )
        return projection

    def _project_history(
        self,
        history: list[ConversationMessage],
    ) -> tuple[list[ContextBlock], list[ProjectionDecision]]:
        if not history:
            return [], []
        blocks: list[ContextBlock] = []
        decisions: list[ProjectionDecision] = []
        recent_count = max(0, self.policy.recent_turns_verbatim_count)
        older = history[:-recent_count] if recent_count else history
        recent = history[-recent_count:] if recent_count else []
        if older:
            summary = _summarize_history_span(older, self.policy.max_history_message_chars_inline)
            payload = summary.to_prompt_dict()
            blocks.append(
                self._block(
                    summary.span_id,
                    "Older conversation compacted summary",
                    payload,
                    block_type="compacted_summary",
                    source_ids=summary.source_ids,
                    included_reason="older_history_summarized",
                )
            )
            decisions.append(
                ProjectionDecision(
                    source_id=summary.span_id,
                    decision="summarize",
                    block_type="compacted_summary",
                    reason="older_history_summarized",
                    original_char_count=sum(len(message.content) for message in older),
                    projected_char_count=len(stable_json(payload)),
                )
            )
        for index, message in enumerate(recent, start=1):
            payload = _compact_message(message)
            source_id = _message_source_id(message)
            original_chars = len(message.content)
            if original_chars > self.policy.max_history_message_chars_inline:
                preview = self._large_preview(
                    result_id=f"history:{source_id}",
                    source_id=source_id,
                    source_type="conversation_history",
                    text=message.content,
                    truncation_reason="history_message_exceeds_inline_limit",
                    artifact_type="history",
                )
                payload["content"] = preview.to_prompt_dict()
                decision = "preview"
                block_type: ContextBlockType = "large_result_preview"
            else:
                decision = "include"
                block_type = "recent_verbatim"
            blocks.append(
                self._block(
                    f"history_recent_{index}",
                    "Recent conversation turn",
                    payload,
                    block_type=block_type,
                    source_ids=[source_id],
                    included_reason="recent_history_context_only",
                    reload_handle=payload.get("content", {}).get("reload_handle")
                    if isinstance(payload.get("content"), dict)
                    else None,
                )
            )
            decisions.append(
                ProjectionDecision(
                    source_id=source_id,
                    decision=decision,  # type: ignore[arg-type]
                    block_type=block_type,
                    reason="recent_history_context_only",
                    original_char_count=original_chars,
                    projected_char_count=len(stable_json(payload)),
                )
            )
        return blocks, decisions

    def _project_evidence(
        self,
        *,
        evidence_facts: list[EvidenceFact],
        execution_plan: ExecutionPlan | None,
        policy: PolicyManifest,
        domain_context: DomainContext | None,
        answerability_assessment: AnswerabilityAssessment | None,
    ) -> tuple[list[ContextBlock], list[ProjectionDecision], list[str], list[str]]:
        if not evidence_facts:
            return [], [], [], []
        if execution_plan is None:
            return self._project_evidence_inventory(evidence_facts)

        selection = select_evidence_for_plan(
            plan=execution_plan,
            policy=policy,
            evidence_facts=evidence_facts,
            domain_context=domain_context,
        )
        accepted_ids = {evidence_id(fact) for fact in selection.accepted}
        answer_allowed = list(answerability_assessment.allowed_evidence_ids) if answerability_assessment else []
        allowed_ids = answer_allowed or sorted(accepted_ids)
        answer_disallowed = {
            item.evidence_id: item.reason
            for item in (answerability_assessment.disallowed_evidence_ids if answerability_assessment else [])
        }
        reason_by_id = _selection_reason_by_id(selection)
        blocks: list[ContextBlock] = []
        decisions: list[ProjectionDecision] = []
        disallowed_ids: list[str] = []

        for fact in evidence_facts:
            evidence_id_value = evidence_id(fact)
            if evidence_id_value in accepted_ids:
                payload = _compact_evidence_fact(fact, include_content=True)
                payload, previews = self._replace_large_evidence(payload, fact, evidence_id_value)
                blocks.append(
                    self._block(
                        evidence_id_value,
                        "Allowed evidence JSON",
                        payload,
                        block_type="allowed_evidence",
                        source_ids=[fact.source_id],
                        evidence_ids=[evidence_id_value],
                        included_reason="selected_by_evidence_guard",
                    )
                )
                decisions.append(
                    ProjectionDecision(
                        source_id=evidence_id_value,
                        decision="include",
                        block_type="allowed_evidence",
                        reason="selected_by_evidence_guard",
                        original_char_count=len(stable_json(_compact_evidence_fact(fact, True))),
                        projected_char_count=len(stable_json(payload)),
                    )
                )
                for preview in previews:
                    blocks.append(
                        self._block(
                            preview.result_id,
                            "Large result preview",
                            preview.to_prompt_dict(),
                            block_type="large_result_preview",
                            source_ids=[fact.source_id],
                            evidence_ids=[evidence_id_value],
                            included_reason="large_evidence_preview",
                            reload_handle=preview.reload_handle,
                        )
                    )
                    decisions.append(
                        ProjectionDecision(
                            source_id=evidence_id_value,
                            decision="preview",
                            block_type="large_result_preview",
                            reason=preview.truncation_reason,
                            original_char_count=preview.original_char_count,
                            projected_char_count=preview.preview_char_count,
                        )
                    )
                continue

            reason = (
                answer_disallowed.get(evidence_id_value)
                or reason_by_id.get(evidence_id_value)
                or "not_allowed_for_current_plan"
            )
            payload = _compact_evidence_fact(fact, include_content=False)
            payload["rejection_reason"] = reason
            payload["content_redacted"] = True
            blocks.append(
                self._block(
                    evidence_id_value,
                    "Disallowed context JSON",
                    payload,
                    block_type="disallowed_evidence",
                    source_ids=[fact.source_id],
                    evidence_ids=[evidence_id_value],
                    included_reason=reason,
                    redacted=True,
                )
            )
            decisions.append(
                ProjectionDecision(
                    source_id=evidence_id_value,
                    decision="redact",
                    block_type="disallowed_evidence",
                    reason=reason,
                    original_char_count=len(stable_json(_compact_evidence_fact(fact, True))),
                    projected_char_count=len(stable_json(payload)),
                )
            )
            disallowed_ids.append(evidence_id_value)
        return blocks, decisions, allowed_ids, disallowed_ids

    def _project_evidence_inventory(
        self,
        evidence_facts: list[EvidenceFact],
    ) -> tuple[list[ContextBlock], list[ProjectionDecision], list[str], list[str]]:
        inventory = [_compact_evidence_inventory(fact) for fact in evidence_facts]
        blocks = [
            self._block(
                "evidence_inventory",
                "Evidence inventory JSON",
                inventory,
                block_type="context_only",
                source_ids=[fact.source_id for fact in evidence_facts if fact.source_id],
                included_reason="planner_inventory_without_execution_plan",
            )
        ]
        decisions = [
            ProjectionDecision(
                source_id=evidence_id(fact),
                decision="summarize",
                block_type="context_only",
                reason="planner_inventory_without_execution_plan",
                original_char_count=len(stable_json(_compact_evidence_fact(fact, True))),
                projected_char_count=len(stable_json(_compact_evidence_inventory(fact))),
            )
            for fact in evidence_facts
        ]
        return blocks, decisions, [], []

    def _inline_value_limit(self, fact: EvidenceFact) -> int:
        """Inline-char budget for an evidence value. Answer-bearing facts (a
        selected knowledge document) are delivered in full up to the larger
        answer budget; everything else uses the compact evidence limit."""
        if fact.fact_type in _ANSWER_EVIDENCE_FACT_TYPES:
            return max(
                self.policy.max_evidence_chars_inline,
                self.policy.max_answer_evidence_chars_inline,
            )
        return self.policy.max_evidence_chars_inline

    def _replace_large_evidence(
        self,
        payload: dict[str, Any],
        fact: EvidenceFact,
        evidence_id: str,
    ) -> tuple[dict[str, Any], list[LargeResultPreview]]:
        previews: list[LargeResultPreview] = []
        output = dict(payload)
        value_inline_limit = self._inline_value_limit(fact)
        if isinstance(output.get("value"), str) and len(output["value"]) > value_inline_limit:
            preview = self._large_preview(
                result_id=f"evidence:{evidence_id}:value",
                source_id=fact.source_id,
                source_type=fact.source_type,
                text=output["value"],
                truncation_reason="evidence_value_exceeds_inline_limit",
                artifact_type=fact.artifact_type,
                fact_type=fact.fact_type,
                status=str(fact.metadata.get("status") or ""),
            )
            output["value"] = preview.to_prompt_dict()
            previews.append(preview)
        metadata = dict(output.get("metadata") or {})
        for key, value in list(metadata.items()):
            if key not in _LARGE_FIELD_NAMES or not isinstance(value, str):
                continue
            if len(value) <= self.policy.large_result_preview_chars:
                continue
            preview = self._large_preview(
                result_id=f"evidence:{evidence_id}:metadata:{key}",
                source_id=fact.source_id,
                source_type=fact.source_type,
                text=value,
                truncation_reason=f"metadata.{key}_exceeds_inline_limit",
                artifact_type=fact.artifact_type,
                fact_type=fact.fact_type,
                status=str(fact.metadata.get("status") or ""),
            )
            metadata[key] = preview.to_prompt_dict()
            previews.append(preview)
        output["metadata"] = metadata
        return output, previews

    def _large_preview(
        self,
        *,
        result_id: str,
        source_id: str,
        source_type: str,
        text: str,
        truncation_reason: str,
        artifact_type: str | None = None,
        fact_type: str | None = None,
        status: str | None = None,
    ) -> LargeResultPreview:
        limit = max(1, self.policy.large_result_preview_chars)
        preview_text = text[:limit]
        handle = self.payload_store.put(
            text,
            metadata={
                "result_id": result_id,
                "source_id": source_id,
                "source_type": source_type,
                "artifact_type": artifact_type,
                "fact_type": fact_type,
            },
        )
        return LargeResultPreview(
            result_id=result_id,
            source_id=source_id,
            source_type=source_type,
            artifact_type=artifact_type,
            fact_type=fact_type,
            status=status or None,
            preview=preview_text,
            original_char_count=len(text),
            preview_char_count=len(preview_text),
            truncation_reason=truncation_reason,
            reload_handle=handle,
        )

    def _block(
        self,
        block_id: str,
        title: str,
        payload: Any,
        *,
        block_type: ContextBlockType = "app_state",
        source_ids: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        included_reason: str = "",
        redacted: bool = False,
        reload_handle: str | None = None,
    ) -> ContextBlock:
        return ContextBlock(
            block_id=_safe_block_id(block_type, block_id),
            block_type=block_type,
            title=title,
            payload=payload,
            source_ids=source_ids or [],
            evidence_ids=evidence_ids or [],
            token_estimate=self.pressure_estimator.token_estimate(payload),
            included_reason=included_reason,
            redacted=redacted,
            reload_handle=reload_handle,
        )


def _app_state_payload(app_state: RuntimeAppState) -> dict[str, Any]:
    payload = app_state.to_prompt_dict()
    payload.pop("current_user_message", None)
    request_payload = payload.pop("request_metadata", {})
    domain_payload = payload.pop("domain_context", {})
    policy_payload = payload.pop("policy", {})
    _rename(payload, "intent_gate", "IntentGate JSON")
    _rename(payload, "execution_plan", "ExecutionPlan JSON")
    _rename(payload, "plan_validation", "Plan validation JSON")
    _rename(payload, "guardrail_decisions", "Guardrail decisions JSON")
    _rename(payload, "preflight", "Adapter preflight JSON")
    _rename(payload, "business_facts", "BusinessFacts JSON")
    return {
        "Request metadata JSON": request_payload,
        "DomainContext JSON": domain_payload,
        "Policy JSON": policy_payload,
        "current_channel": _current_channel(domain_payload, request_payload),
        "material_pack_routing": _material_pack_routing(policy_payload),
        "available_artifacts": _available_artifacts(domain_payload, request_payload),
        **payload,
    }


def _rename(payload: dict[str, Any], old: str, new: str) -> None:
    if old in payload:
        payload[new] = payload.pop(old)


def _compact_domain_context(domain_context: DomainContext | None) -> dict[str, Any]:
    if domain_context is None:
        return {}
    payload = domain_context.to_prompt_dict()
    if payload.get("products"):
        payload["products"] = [
            {
                "id": product.get("id"),
                "channel_id": product.get("channel_id"),
                "strategy_ids": product.get("strategy_ids", []),
                "source_id": product.get("source_id"),
                "provenance": product.get("provenance"),
                "name_redacted": True,
            }
            for product in payload.get("products", [])
            if isinstance(product, dict)
        ]
    return payload


def _current_channel(domain_payload: object, request_payload: object) -> dict[str, Any]:
    if isinstance(domain_payload, dict) and isinstance(domain_payload.get("channel"), dict):
        return domain_payload["channel"]
    request_payload = request_payload if isinstance(request_payload, dict) else {}
    return {
        "id": "unknown",
        "name": request_payload.get("dist_channel_name") or request_payload.get("group_name") or "",
        "kind": request_payload.get("channel_type") or "unknown",
        "provenance": "request",
    }


def _material_pack_routing(policy_payload: object) -> dict[str, Any]:
    policy = policy_payload if isinstance(policy_payload, dict) else {}
    payload: dict[str, Any] = {
        "material_pack_options": policy.get("material_pack_options"),
    }
    return {key: value for key, value in payload.items() if value}


def _available_artifacts(domain_payload: object, request_payload: object) -> list[dict[str, Any]]:
    if isinstance(domain_payload, dict) and domain_payload.get("artifacts"):
        return list(domain_payload.get("artifacts") or [])
    request_payload = request_payload if isinstance(request_payload, dict) else {}
    return [
        {
            "artifact_type": artifact.get("type"),
            "source_type": "request.available_artifacts",
            "options": artifact.get("options", []),
        }
        for artifact in request_payload.get("available_artifacts", []) or []
        if isinstance(artifact, dict)
    ]


def _summarize_history_span(
    messages: list[ConversationMessage],
    inline_limit: int,
) -> CompactedSpanSummary:
    role_counts: dict[str, int] = {}
    snippets: list[str] = []
    for message in messages:
        role_counts[message.role] = role_counts.get(message.role, 0) + 1
        snippets.append("{}: {}".format(message.role, _clip(message.content, min(180, inline_limit))))
    start = messages[0].created_at.isoformat() if messages else None
    end = messages[-1].created_at.isoformat() if messages else None
    source_ids = [_message_source_id(message) for message in messages]
    return CompactedSpanSummary(
        span_id=_safe_block_id("compacted_summary", "history:{}:{}".format(start, end)),
        original_message_count=len(messages),
        role_counts=role_counts,
        start_time=start,
        end_time=end,
        summary="Older conversation context only; not claim evidence. " + " | ".join(snippets),
        unresolved_items=[],
        source_ids=source_ids,
    )


def _pending_clarification_from_history(
    history: list[ConversationMessage],
) -> dict[str, Any] | None:
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message.role != "assistant":
            continue
        payload = _assistant_runtime_history_payload(message.content)
        reply = payload.get("reply_response", {}).get("reply", {})
        pending_plan = payload.get("pending_plan")
        if reply.get("kind") != "clarification" or not isinstance(pending_plan, dict):
            continue
        return {
            "status": "awaiting_user_answer",
            "assistant_question": reply.get("text", ""),
            "pending_plan": pending_plan,
            "user_messages_after_question": [
                item.content for item in history[index + 1 :] if item.role == "user"
            ],
            "instruction": (
                "If the current user message answers this clarification, reuse the "
                "pending plan intent and do not ask the same clarification again. "
                "If adapter evidence later shows unavailable content, return unable "
                "or handoff instead of clarifying."
            ),
        }
    return None


def _assistant_runtime_history_payload(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("contract_version") != "reply-runtime-history":
        return {}
    return payload


def _compact_message(message: ConversationMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
        "source_metadata": source_metadata_prompt_dict(
            message.source_metadata
            or SourceMetadata(
                source_id=_message_source_id(message),
                source_type="user_message" if message.role == "user" else "assistant_message",
                artifact_type="history",
                created_at=message.created_at,
                observed_at=message.created_at,
                provenance="conversation_store",
                evidence_allowed_by_default=False,
            )
        ),
        "context_only": True,
    }


def _compact_action_record(record: ActionLedgerRecord) -> dict[str, Any]:
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
            source_id=_action_source_id(record),
            source_type="tool_result",
            artifact_type="history",
            channel_id=record.group_id,
            created_at=record.received_at,
            observed_at=record.received_at,
            provenance="adapter_action_ledger",
            evidence_allowed_by_default=False,
        ).to_prompt_dict(),
    }


def _compact_policy(policy: PolicyManifest) -> dict[str, Any]:
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


def _compact_planner_capabilities(request: ReplyRequest, policy: PolicyManifest) -> dict[str, Any]:
    cards = planner_capability_cards(
        request,
        {"policy_allowed_capabilities": sorted(policy.allowed_capabilities)},
    )
    return {
        "candidate_count": len(cards),
        "capability_cards": [_planner_capability_guidance_line(card) for card in cards],
    }


def _planner_capability_guidance_line(card: dict[str, object]) -> str:
    evidence = card.get("evidence") if isinstance(card.get("evidence"), dict) else {}
    return (
        "{id}|t={type}|rt={runtime}|req={required_artifacts}|forbid={forbidden_artifacts}|"
        "need={required_facts}|any={any_of_facts}|no_src={forbidden_sources}|"
        "guidance={guidance}|pos={positive}|neg={negative}"
    ).format(
        id=card.get("id"),
        type=card.get("capability_type"),
        runtime=card.get("runtime_capability"),
        required_artifacts=",".join(card.get("required_artifacts", [])),
        forbidden_artifacts=",".join(card.get("forbidden_artifacts", [])),
        required_facts=",".join(evidence.get("required_fact_types", [])),
        any_of_facts=",".join(evidence.get("any_of_fact_types", [])),
        forbidden_sources=",".join(evidence.get("forbidden_source_types", [])),
        guidance=_clip(card.get("planner_guidance"), 220),
        positive=_clip(_first_value(card.get("examples_positive", [])), 80),
        negative=_clip(_first_value(card.get("examples_negative", [])), 80),
    )


def _compact_plan_validation(validation: PlanValidationResult | None) -> dict[str, Any] | None:
    if validation is None:
        return None
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


def _compact_guardrail_decisions(decisions: list[GuardrailDecision]) -> list[dict[str, Any]]:
    return [decision.model_dump(mode="json", exclude_none=True) for decision in decisions]


def _compact_preflight(preflight: AdapterPreflightSnapshot) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in preflight.items:
        if item.result is None:
            items.append({"resolve_type": item.resolve_type, "status": item.status, "error": item.error})
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
                "period_start": result.period_start,
                "period_end": result.period_end,
                "period_label": result.period_label,
                "scope_complete": result.scope_complete,
                "expected_product_count": result.expected_product_count,
                "generated_product_count": result.generated_product_count,
                "missing_product_count": result.missing_product_count,
                "report_sections": [section.model_dump(mode="json", exclude_none=True) for section in result.report_sections],
                "resolve_ref_available": bool(result.resolve_ref),
            }
        )
    return items


def _compact_evidence_fact(fact: EvidenceFact, include_content: bool) -> dict[str, Any]:
    metadata = dict(fact.metadata)
    if "resolve_ref" in metadata:
        metadata["resolve_ref_available"] = bool(metadata.pop("resolve_ref"))
    if not include_content:
        metadata = {
            key: value
            for key, value in metadata.items()
            if key
            in {
                "status",
                "reason_code",
                "material_pack_option",
                "period",
                "report_date",
            }
        }
    payload = {
        "evidence_id": evidence_id(fact),
        "fact_type": fact.fact_type,
        "source_type": fact.source_type,
        "source_id": fact.source_id,
        "resolve_type": fact.resolve_type,
        "artifact_type": fact.artifact_type,
        "scope": fact.scope.to_prompt_dict(),
        "source_metadata": source_metadata_prompt_dict(fact.source_metadata),
        "metadata": metadata,
    }
    if include_content:
        payload["value"] = fact.value
    return payload


def _compact_evidence_inventory(fact: EvidenceFact) -> dict[str, Any]:
    metadata = dict(fact.metadata)
    resolve_ref = metadata.pop("resolve_ref", None)
    return {
        "evidence_id": evidence_id(fact),
        "fact_type": fact.fact_type,
        "source_type": fact.source_type,
        "source_id": fact.source_id,
        "resolve_type": fact.resolve_type,
        "artifact_type": fact.artifact_type,
        "value_present": fact.value is not None and fact.value is not False,
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key
            in {
                "status",
                "reason_code",
                "material_pack_option",
                "period",
                "report_date",
            }
        }
        | ({"resolve_ref_available": bool(resolve_ref)} if resolve_ref is not None else {}),
        "source_metadata": source_metadata_prompt_dict(fact.source_metadata),
    }


def _compact_response(response: ReplyResponse | None) -> dict[str, Any] | None:
    if response is None:
        return None
    payload = response.model_dump(mode="json", exclude_none=True)
    for action in payload.get("actions", []):
        if "resolve_ref" in action:
            action["resolve_ref_available"] = bool(action.pop("resolve_ref"))
    return payload


def _selection_reason_by_id(selection: EvidenceSelection) -> dict[str, str]:
    return {
        evidence_id: decision.reason_code
        for decision in selection.decisions
        for evidence_id in decision.evidence_seen
        if evidence_id
    }


def _message_source_id(message: ConversationMessage) -> str:
    if message.source_metadata is not None:
        return message.source_metadata.source_id
    return f"{message.role}:{message.created_at.isoformat()}"


def _action_source_id(record: ActionLedgerRecord) -> str:
    return record.response_id or record.context_id or record.execution.action_id or "action-ledger"


def _context_only_source_ids(blocks: list[ContextBlock]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for block in blocks:
        if block.block_type not in {"recent_verbatim", "compacted_summary", "context_only"}:
            if not (block.block_type == "large_result_preview" and not block.evidence_ids):
                continue
        if block.block_type == "large_result_preview" and block.evidence_ids:
            continue
        for source_id in block.source_ids:
            if source_id in seen:
                continue
            seen.add(source_id)
            output.append(source_id)
    return output


def _raw_context(**kwargs: Any) -> dict[str, Any]:
    return {key: _model_dump(value) for key, value in kwargs.items()}


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "to_prompt_dict"):
        return value.to_prompt_dict()
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    if isinstance(value, tuple):
        return [_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: _model_dump(item) for key, item in value.__dict__.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _projection_id(payload: dict[str, Any]) -> str:
    return "ctx-proj:" + hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:16]


def _safe_block_id(block_type: str, source: str) -> str:
    return f"{block_type}:{hashlib.sha256(str(source).encode('utf-8')).hexdigest()[:12]}"


def _first_value(value: object) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return ""
    return str(value[0])


def _clip(value: object, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
