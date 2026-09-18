from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning import (
    PlanSpec,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.recall.outcome_models import (
    RecallBranchOutcomeV1,
    RecallOutcomeV1,
)
from market_support_crewai_agent.runtime.recall.turn_state import (
    RecallTurnStateV1,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    build_composer_prompt_input_v1,
)
from market_support_crewai_agent.runtime.v2_attempt import (
    V2AttemptResult,
    V2ReplyValidationResult,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.unit.llm._composer_stage_contract_fixtures import (
    composer_scenario,
    with_static_facts,
)


@dataclass(frozen=True, slots=True)
class PlannerFinalizerControl:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    scope: BusinessScopeAuthorityV1
    neutral: PlanSpec
    widening: PlanSpec


@dataclass(frozen=True, slots=True)
class ComposerFinalizerControl:
    input_value: KnowledgeComposerPromptInputV1
    neutral: ComposerReplyOutput
    evidence_widening: ComposerReplyOutput
    ceiling_widening: ComposerReplyOutput


def planner_finalizer_control() -> PlannerFinalizerControl:
    request = make_v2_envelope(
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:stage-output",
            "direct_thread_ref": "direct:stage-output",
            "principal_ref": "principal:stage-output",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "tester",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    scope = business_scope_authority_v1(request.business_scope)
    policy = PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(request, scope),
        policy_ledger_summary_v1((), 0),
    )
    neutral = PlanSpec.model_validate(
        {
            "plan_id": "neutral-company-answer",
            "user_intent_summary": "answer a company question",
            "plan_units": [
                {
                    "unit_id": "company-answer",
                    "selected_capability_id": "answer_internal_company_knowledge",
                    "domain_scope": {"kind": "unscoped"},
                    "answerability_policy": "answer",
                    "output_schema_ref": (
                        "answer_internal_company_knowledge:output_schema"
                    ),
                }
            ],
        }
    )
    widening_unit = neutral.plan_units[0].model_copy(
        update={
            "selected_capability_id": "material_pack.send",
            "output_schema_ref": "material_pack.send:output_schema",
        }
    )
    widening = neutral.model_copy(update={"plan_units": (widening_unit,)})
    return PlannerFinalizerControl(request, policy, scope, neutral, widening)


def composer_finalizer_control() -> ComposerFinalizerControl:
    scenario = with_static_facts(
        composer_scenario("knowledge_answer", direct=True),
        ("company_shareholders",),
    )
    input_value = build_composer_prompt_input_v1(scenario.invocation)
    if not isinstance(input_value, KnowledgeComposerPromptInputV1):
        raise AssertionError("knowledge scenario must build a knowledge composer input")
    admitted_id = scenario.groundings[0].allowed_evidence_ids[0]
    forged_id = "eid1:" + "f" * 64
    neutral = ComposerReplyOutput(
        response_mode="answer",
        evidence_ids=[admitted_id],
        reply=PrimaryReply(kind="answer", text="grounded answer"),
    )
    evidence_widening = neutral.model_copy(
        update={"evidence_ids": [admitted_id, forged_id]}
    )
    ceiling_widening = neutral.model_copy(
        update={
            "reply": PrimaryReply(kind="answer", text="界" * 4_001),
        }
    )
    return ComposerFinalizerControl(
        input_value,
        neutral,
        evidence_widening,
        ceiling_widening,
    )


def selected_refs(plan: ExecutionPlanV2) -> frozenset[str]:
    return frozenset(ref.manifest_id for ref in plan.selected_manifest_refs)


def verifier_candidate() -> V2AttemptResult:
    control = planner_finalizer_control()
    plan = finalize_execution_plan_v2(
        control.neutral,
        control.policy,
        control.scope,
        origin="planner",
    )
    outcome = RecallOutcomeV1(
        mode="off",
        decision="disabled",
        candidates=(),
        branch_outcomes=(
            RecallBranchOutcomeV1(
                source_class="approved_static",
                status="not_called",
                accepted_count=0,
                rejected_count=0,
                reason_code="disabled",
            ),
            RecallBranchOutcomeV1(
                source_class="document_mcp",
                status="not_called",
                accepted_count=0,
                rejected_count=0,
                reason_code="disabled",
            ),
        ),
        shortcut_summary=None,
        trace_hash="rch1:" + "0" * 64,
    )
    evidence = CanonicalEvidenceExecutionResultV1(
        preflight=AdapterPreflightSnapshot.empty(),
        canonical_facts=(),
        resolve_bindings=(),
        groundings=(),
        domain_context=DomainContextV1Builder().build(
            control.request,
            scope_authority=control.scope,
        ),
    )
    return V2AttemptResult(
        plan=plan,
        evidence=evidence,
        response=ReplyResponse(
            reply=PrimaryReply(kind="answer", text="neutral answer")
        ),
        reply_validation=V2ReplyValidationResult(valid=True),
        reason_code="question_recall_hit",
        recall_state=RecallTurnStateV1(outcome=outcome, shortcut_payload=None),
    )
