from __future__ import annotations

from typing import final

from market_support_crewai_agent.runtime.context.retry_view_models import (
    ComposerRetryOverlayV1,
)
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime import planning_flow, v2_attempt
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.alignment import (
    AlignmentVerifierInvocationV1,
    InternalAlignmentVerifierSourceV1,
    build_runtime_alignment_verifier_input_v1,
    project_alignment_history_v1,
    verify_reply_alignment,
)
from market_support_crewai_agent.runtime.validation.alignment_loop import (
    AlignmentActionAttemptV1,
    AlignmentAttemptV1,
    AlignmentFallbackV1,
    AlignmentRefetchOutcomeV1,
)
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    build_alignment_refetch_request_v1,
)
from market_support_crewai_agent.runtime.validation.alignment_runtime_contracts import (
    AlignmentPlanningFlowV1,
    AlignmentRuntimeV1,
    RuntimeAlignmentContextV1,
    V2AttemptTransportV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)

if not isinstance(planning_flow, AlignmentPlanningFlowV1):
    raise ContextViewInvariantError("alignment_planning_flow_contract_missing")
if not isinstance(v2_attempt, V2AttemptTransportV1):
    raise ContextViewInvariantError("v2_attempt_transport_contract_missing")
_PLANNING_FLOW: AlignmentPlanningFlowV1 = planning_flow
_V2_ATTEMPT: V2AttemptTransportV1 = v2_attempt


@final
class RuntimeAlignmentRemediatorV1:
    def __init__(
        self,
        runtime: AlignmentRuntimeV1,
        context: RuntimeAlignmentContextV1,
    ) -> None:
        self._runtime = runtime
        self._context = context

    async def verdict_for(
        self,
        candidate: V2AttemptResult,
        attempt: AlignmentAttemptV1,
    ) -> ReplyAlignmentVerdict:
        input_value = build_runtime_alignment_verifier_input_v1(
            AlignmentVerifierInvocationV1(
                request=self._context.request,
                policy=self._context.policy,
                scope_authority=self._context.scope_authority,
                plan=candidate.plan,
                evidence=candidate.evidence,
                response=candidate.response,
                reason_code=candidate.reason_code,
                history=self._context.history,
                attempt=attempt,
                now=self._context.now,
                locator_safety=LocatorSafetyClassifierV1.from_settings(
                    self._runtime.settings
                ),
            )
        )
        verdict = await verify_reply_alignment(
            self._runtime,
            input_value,
            InternalAlignmentVerifierSourceV1(
                model_family=self._context.model_family,
                prompt_programs=self._context.prompt_programs,
                llm_executions=self._context.llm_executions,
            ),
        )
        self._context.alignment_verdicts.append(verdict)
        return verdict

    async def replan(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: AlignmentActionAttemptV1,
    ) -> V2AttemptResult:
        recall_state = candidate.recall_state
        if recall_state is None:
            raise AgentRuntimeError("alignment_replan_missing_recall_state")
        self._record_remediation(verdict, "replan", attempt)
        return await _PLANNING_FLOW.build_candidate_via_planner(
            self._runtime,
            request=self._context.request,
            domain_context=candidate.evidence.domain_context,
            policy=self._context.policy,
            model_family=self._context.model_family,
            intent_gate=self._context.intent_gate,
            history=list(self._context.history),
            action_history=self._context.action_history,
            prompt_programs=self._context.prompt_programs,
            llm_executions=self._context.llm_executions,
            scope_authority=self._context.scope_authority,
            state_key_ref=self._context.state_key_ref,
            alignment_verdict=verdict,
            alignment_attempt=attempt,
            recall_state=recall_state,
        )

    async def refetch(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: AlignmentActionAttemptV1,
    ) -> AlignmentRefetchOutcomeV1:
        request = build_alignment_refetch_request_v1(
            candidate.plan,
            verdict,
            attempt,
        )
        self._record_remediation(verdict, verdict.remediation, attempt)
        refreshed = await _V2_ATTEMPT.build_candidate_from_plan_v2(
            self._runtime,
            request=self._context.request,
            policy=self._context.policy,
            scope_authority=self._context.scope_authority,
            state_key_ref=self._context.state_key_ref,
            plan=candidate.plan,
            recall_state=candidate.recall_state,
            alignment_refetch_request=request,
        )
        return AlignmentRefetchOutcomeV1(request=request, candidate=refreshed)

    async def recompose(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict,
        attempt: AlignmentActionAttemptV1,
    ) -> V2AttemptResult:
        self._record_remediation(verdict, "recompose", attempt)
        feedback = " ".join(
            (
                verdict.composer_feedback or verdict.rationale or verdict.failure_code
            ).split()
        )
        return await _V2_ATTEMPT.recompose_candidate_v2(
            self._runtime,
            candidate=candidate,
            request=self._context.request,
            policy=self._context.policy,
            scope_authority=self._context.scope_authority,
            history=project_alignment_history_v1(
                self._context.history,
                self._context.now,
            ),
            retry_overlay=ComposerRetryOverlayV1(
                attempt=attempt,
                feedback=feedback[:300],
            ),
        )

    async def replace(
        self,
        candidate: V2AttemptResult,
        verdict: ReplyAlignmentVerdict | None,
        remediation: AlignmentFallbackV1,
    ) -> V2AttemptResult:
        manifest_id = (
            "general.clarification"
            if remediation == "return_clarification"
            else "general.abstention"
        )
        answerability = (
            "clarify" if remediation == "return_clarification" else "abstain"
        )
        plan = finalize_execution_plan_v2(
            DeterministicPlanOriginInputV1(
                user_need=candidate.plan.user_need,
                units=(
                    DeterministicPlanUnitV1(
                        unit_id="alignment-remediation",
                        manifest_id=manifest_id,
                        answerability_policy=answerability,
                    ),
                ),
                compliance_reason_code=candidate.plan.compliance.reason_code,
                confidence=1.0,
            ),
            self._context.policy,
            self._context.scope_authority,
            origin="remediation",
        )
        self._context.alignment_remediations.append(
            {
                "remediation": remediation,
                "failure_code": "none" if verdict is None else verdict.failure_code,
            }
        )
        return await _V2_ATTEMPT.build_candidate_from_plan_v2(
            self._runtime,
            request=self._context.request,
            policy=self._context.policy,
            scope_authority=self._context.scope_authority,
            state_key_ref=self._context.state_key_ref,
            plan=plan,
            recall_state=candidate.recall_state,
        )

    def _record_remediation(
        self,
        verdict: ReplyAlignmentVerdict,
        remediation: str,
        attempt: AlignmentActionAttemptV1,
    ) -> None:
        self._context.alignment_remediations.append(
            {
                "remediation": remediation,
                "failure_code": verdict.failure_code,
                "attempt": attempt,
            }
        )


__all__ = [
    "RuntimeAlignmentContextV1",
    "RuntimeAlignmentRemediatorV1",
]
