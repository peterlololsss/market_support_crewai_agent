from __future__ import annotations

from typing import Any

from market_support_crewai_agent.runtime.orchestration.alignment_refetch import (
    plan_can_refetch_report_scope,
    report_scope_refetch_query,
    report_scope_refetch_requested,
)
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.orchestration.crewai_io import safe_short_text
from market_support_crewai_agent.runtime.state.runtime_trace import trace_event, trace_span
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    abstention_response_text,
)


async def ensure_aligned_response(
    runtime: Any,
    *,
    request,
    canonical_context,
    domain_context,
    policy,
    model_family,
    intent_gate,
    history,
    action_history,
    prompt_programs,
    llm_executions,
    alignment_verdicts,
    alignment_remediations,
    candidate,
):
    replan_count = 0
    refetch_count = 0
    recompose_count = 0
    total_remediations = 0

    while True:
        try:
            verdict = await runtime._verify_reply_alignment(
                request=request,
                canonical_context=canonical_context,
                policy=policy,
                model_family=model_family,
                intent_gate=intent_gate,
                history=history,
                action_history=action_history,
                prompt_programs=prompt_programs,
                llm_executions=llm_executions,
                candidate=candidate,
                attempt=len(alignment_verdicts),
            )
        except Exception as exc:
            trace_event("alignment.verifier_failed", error=safe_short_text(exc))
            alignment_remediations.append(
                {
                    "remediation": "return_unable",
                    "reason": "verifier_failed",
                    "error": safe_short_text(exc),
                }
            )
            return runtime._fallback_attempt(
                candidate,
                policy,
                kind="unable",
                reason_code="alignment_verifier_failed",
            )

        alignment_verdicts.append(verdict)
        if verdict.aligned and verdict.safe_to_return:
            trace_event("alignment.accepted", attempt=len(alignment_verdicts) - 1)
            return candidate

        if total_remediations >= runtime.settings.reply_alignment_max_total_remediations:
            trace_event(
                "alignment.remediation_limit",
                failure_code=verdict.failure_code,
            )
            alignment_remediations.append(
                {
                    "remediation": "return_unable",
                    "reason": "remediation_limit_exceeded",
                    "failure_code": verdict.failure_code,
                }
            )
            return runtime._fallback_attempt(
                candidate,
                policy,
                kind="unable",
                reason_code="alignment_remediation_limit",
            )

        if (
            verdict.remediation == "replan"
            and replan_count < runtime.settings.reply_alignment_max_replans
        ):
            replan_count += 1
            total_remediations += 1
            alignment_remediations.append(
                {
                    "remediation": "replan",
                    "failure_code": verdict.failure_code,
                }
            )
            try:
                with trace_span("alignment.remediate.replan"):
                    candidate = await runtime._build_candidate_response(
                        request=request,
                        canonical_context=canonical_context,
                        domain_context=domain_context,
                        policy=policy,
                        model_family=model_family,
                        intent_gate=intent_gate,
                        history=history,
                        action_history=action_history,
                        prompt_programs=prompt_programs,
                        llm_executions=llm_executions,
                        alignment_verdict=verdict,
                        alignment_attempt=len(alignment_verdicts),
                    )
            except Exception as exc:
                trace_event("alignment.replan_failed", error=safe_short_text(exc))
                alignment_remediations.append(
                    {
                        "remediation": "return_unable",
                        "reason": "replan_failed",
                        "error": safe_short_text(exc),
                    }
                )
                return runtime._fallback_attempt(
                    candidate,
                    policy,
                    kind="unable",
                    reason_code="alignment_replan_failed",
                )
            if not candidate.reply_validation.valid:
                return candidate
            continue

        if (
            report_scope_refetch_requested(verdict, candidate.plan)
            and refetch_count < runtime.settings.reply_alignment_max_evidence_refetches
            and candidate.plan.response_mode == "knowledge_answer"
            and plan_can_refetch_report_scope(candidate.plan, policy)
            and report_scope_refetch_query(verdict)
        ):
            refetch_count += 1
            total_remediations += 1
            refined_query = report_scope_refetch_query(verdict)
            alignment_remediations.append(
                {
                    "remediation": "refetch_report_scope",
                    "failure_code": verdict.failure_code,
                    "refined_evidence_query": refined_query,
                }
            )
            with trace_span("alignment.remediate.refetch_report_scope"):
                candidate = await runtime._build_candidate_from_plan(
                    request=request,
                    canonical_context=canonical_context,
                    domain_context=candidate.domain_context,
                    policy=policy,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    plan=candidate.plan.model_copy(update={"evidence_query": refined_query}),
                    alignment_verdict=verdict,
                    alignment_attempt=len(alignment_verdicts),
                )
            if not candidate.reply_validation.valid:
                return candidate
            continue

        if (
            verdict.remediation == "refetch_document_context"
            and refetch_count < runtime.settings.reply_alignment_max_evidence_refetches
            and candidate.plan.response_mode == "knowledge_answer"
            and "document_context" in candidate.plan.capabilities
            and "document_context" in policy.allowed_capabilities
            and (verdict.refined_evidence_query or "").strip()
        ):
            refetch_count += 1
            total_remediations += 1
            alignment_remediations.append(
                {
                    "remediation": "refetch_document_context",
                    "failure_code": verdict.failure_code,
                }
            )
            with trace_span("alignment.remediate.refetch_document_context"):
                candidate = await runtime._build_candidate_from_plan(
                    request=request,
                    canonical_context=canonical_context,
                    domain_context=candidate.domain_context,
                    policy=policy,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    plan=candidate.plan.model_copy(
                        update={"evidence_query": verdict.refined_evidence_query}
                    ),
                    alignment_verdict=verdict,
                    alignment_attempt=len(alignment_verdicts),
                )
            if not candidate.reply_validation.valid:
                return candidate
            continue

        if (
            verdict.remediation == "recompose"
            and recompose_count < runtime.settings.reply_alignment_max_recomposes
            and candidate.directive.requires_knowledge_composer
        ):
            recompose_count += 1
            total_remediations += 1
            alignment_remediations.append(
                {
                    "remediation": "recompose",
                    "failure_code": verdict.failure_code,
                }
            )
            with trace_span("alignment.remediate.recompose"):
                response, composer_output = await runtime._compose_or_render_response(
                    request=request,
                    canonical_context=canonical_context,
                    domain_context=candidate.domain_context,
                    policy=policy,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    plan=candidate.plan,
                    plan_validation=candidate.plan_validation,
                    preflight=candidate.preflight,
                    evidence_facts=candidate.evidence_facts,
                    business_facts=candidate.business_facts,
                    answerability=candidate.answerability,
                    directive=candidate.directive,
                    alignment_verdict=verdict,
                    alignment_attempt=len(alignment_verdicts),
                    guardrail_decisions=candidate.guardrail_decisions,
                )
                candidate = runtime._validated_attempt(
                    plan=candidate.plan,
                    plan_validation=candidate.plan_validation,
                    preflight=candidate.preflight,
                    evidence_facts=candidate.evidence_facts,
                    business_facts=candidate.business_facts,
                    domain_context=candidate.domain_context,
                    answerability=candidate.answerability,
                    directive=candidate.directive,
                    response=response,
                    policy=policy,
                    guardrail_decisions=candidate.guardrail_decisions,
                    composer_output=composer_output,
                )
            if not candidate.reply_validation.valid:
                return candidate
            continue

        if verdict.remediation == "return_clarification":
            alignment_remediations.append(
                {
                    "remediation": "return_clarification",
                    "failure_code": verdict.failure_code,
                }
            )
            directive = ResponseDirective(
                mode="clarification",
                reply_kind="clarification",
                text=verdict.composer_feedback
                or verdict.rationale
                or abstention_response_text(),
                requires_knowledge_composer=True,
                composer_stage="knowledge_composer",
                reason_code="alignment_return_clarification",
            )
            answerability = candidate.answerability.model_copy(
                update={
                    "can_answer": False,
                    "ambiguity": "other",
                    "recommended_response_mode": "clarify",
                    "user_facing_reason": directive.text,
                }
            )
            try:
                with trace_span("alignment.remediate.return_clarification"):
                    response, composer_output = await runtime._compose_or_render_response(
                        request=request,
                        canonical_context=canonical_context,
                        domain_context=candidate.domain_context,
                        policy=policy,
                        model_family=model_family,
                        intent_gate=intent_gate,
                        history=history,
                        action_history=action_history,
                        prompt_programs=prompt_programs,
                        llm_executions=llm_executions,
                        plan=candidate.plan,
                        plan_validation=candidate.plan_validation,
                        preflight=candidate.preflight,
                        evidence_facts=candidate.evidence_facts,
                        business_facts=candidate.business_facts,
                        answerability=answerability,
                        directive=directive,
                        alignment_verdict=verdict,
                        alignment_attempt=len(alignment_verdicts),
                        guardrail_decisions=candidate.guardrail_decisions,
                    )
                    return runtime._validated_attempt(
                        plan=candidate.plan,
                        plan_validation=candidate.plan_validation,
                        preflight=candidate.preflight,
                        evidence_facts=candidate.evidence_facts,
                        business_facts=candidate.business_facts,
                        domain_context=candidate.domain_context,
                        answerability=answerability,
                        directive=directive,
                        response=response,
                        policy=policy,
                        guardrail_decisions=candidate.guardrail_decisions,
                        composer_output=composer_output,
                    )
            except Exception as exc:
                trace_event(
                    "alignment.return_clarification_failed",
                    error=safe_short_text(exc),
                )
                return runtime._fallback_attempt(
                    candidate,
                    policy,
                    kind="unable",
                    reason_code="alignment_return_clarification_failed",
                )

        alignment_remediations.append(
            {
                "remediation": "return_unable",
                "reason": "unsupported_or_exhausted_remediation",
                "suggested_remediation": verdict.remediation,
                "failure_code": verdict.failure_code,
            }
        )
        return runtime._fallback_attempt(
            candidate,
            policy,
            kind="unable",
            reason_code="alignment_return_unable",
        )
