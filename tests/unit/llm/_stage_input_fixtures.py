from __future__ import annotations

from typing import Final

from market_support_crewai_agent.runtime.context.common_view_models import (
    RelativeYearsViewV1,
)
from market_support_crewai_agent.runtime.context.models import (
    CandidateReplyViewV1,
    ComposerRetryOverlayV1,
    CurrentMessageViewV1,
    EffectiveOutputCeilingsViewV1,
    EffectivePolicyViewV1,
    HistoryTurnViewV1,
    IntentGateViewV1,
    MaterialPackOptionSummaryViewV1,
    PendingClarificationViewV1,
    ResponseDirectiveViewV1,
    RuntimeClockViewV1,
    ScenePresentationViewV1,
    UnitGroundingViewV1,
    UnscopedPlanScopeViewV1,
    ValidatedPlanUnitViewV1,
    ValidatedPlanViewV1,
)
from market_support_crewai_agent.runtime.context.policy_view_models import (
    UnscopedBusinessScopeViewV1,
)
from market_support_crewai_agent.runtime.context.stage_inputs import (
    AlignmentVerifierPromptInputSourceV1,
    KnowledgeComposerPromptInputSourceV1,
    PlannerPromptInputSourceV1,
    SmalltalkComposerPromptInputSourceV1,
)
from tests.unit.llm._stage_input_fixture_components import (
    KNOWLEDGE_REF,
    SMALLTALK_REF,
    business_facts,
    composer_capabilities,
    planner_capabilities,
    recall,
    verifier_capabilities,
)


DEFAULT_TEXT: Final = "请介绍公司的投研策略"


def stage_sources(
    text: str = DEFAULT_TEXT,
) -> tuple[
    PlannerPromptInputSourceV1,
    KnowledgeComposerPromptInputSourceV1,
    SmalltalkComposerPromptInputSourceV1,
    AlignmentVerifierPromptInputSourceV1,
]:
    message = CurrentMessageViewV1(text=text)
    history = (HistoryTurnViewV1(role="user", text=text[:1_200], age_seconds=1),)
    clock = RuntimeClockViewV1(
        current_date="2026-07-18",
        current_datetime="2026-07-18T14:20:00+08:00",
        weekday=6,
        relative_years=RelativeYearsViewV1(
            current=2026,
            last=2025,
            two_years_ago=2024,
        ),
    )
    presentation = ScenePresentationViewV1(
        scene="group",
        audience="group",
        conversation_name=text[:120],
        principal_name=text[:80],
        redacted=True,
    )
    policy = EffectivePolicyViewV1(
        policy_id="pol1:" + "b" * 64,
        scene="group",
        eligible_capabilities=(KNOWLEDGE_REF,),
        read_capabilities=("query_internal_company_info",),
        internal_company_knowledge_enabled=True,
        outbound_actions=(),
        mention_types=(),
        adapter_resolves=(),
        allowed_reply_modes=("knowledge_answer", "smalltalk"),
        recall_mode="off",
        evidence_call_limit=1,
        actions_allowed=False,
        mentions_allowed=False,
    )
    query = text[:190]
    scope = UnscopedPlanScopeViewV1()
    units = tuple(
        ValidatedPlanUnitViewV1(
            unit_id=f"unit-{index}",
            manifest_ref=KNOWLEDGE_REF,
            answerability="answer",
            scope=scope,
            evidence_query=f"{query}-{index}",
        )
        for index in (1, 2)
    )
    plan = ValidatedPlanViewV1(
        execution_plan_id="epl1:" + "d" * 64,
        user_need=text[:500],
        response_mode="knowledge_answer",
        artifact_kind="knowledge_answer",
        units=units,
        selected_manifest_refs=(KNOWLEDGE_REF,),
        is_compliant=True,
        compliance_reason_code="compliant_product_request",
        compliance_reason=text[:300],
        confidence=0.9,
    )
    groundings = tuple(
        UnitGroundingViewV1(
            unit_id=unit.unit_id,
            manifest_ref=unit.manifest_ref,
            answerability=unit.answerability,
            scope=unit.scope,
            evidence_query=unit.evidence_query,
            business_facts=business_facts(),
        )
        for unit in units
    )
    ceilings = EffectiveOutputCeilingsViewV1(
        allowed_reply_kinds=("answer", "clarification", "unable_to_answer"),
        mentions_allowed=False,
        max_mentions=0,
        max_reply_chars=4_000,
    )
    knowledge_directive = ResponseDirectiveViewV1(
        mode="knowledge_answer",
        reply_kind="answer",
        reason_code="grounded_answer",
        requires_composer=True,
        composer_stage="knowledge_composer",
        mentions_requested=False,
        action_intent_count=0,
    )
    smalltalk_directive = ResponseDirectiveViewV1(
        mode="smalltalk",
        reply_kind="answer",
        reason_code="smalltalk_reply",
        requires_composer=True,
        composer_stage="smalltalk_composer",
        mentions_requested=False,
        action_intent_count=0,
    )
    planner = PlannerPromptInputSourceV1(
        scene="group",
        message=message,
        history=history,
        runtime_clock=clock,
        pending_clarification=PendingClarificationViewV1(
            kind="other",
            slots=("scope",),
            question=text[:400],
            age_turns=1,
        ),
        recent_executed_actions=(),
        material_pack_options=MaterialPackOptionSummaryViewV1(
            total_count=100_000,
            exact_requested_option=text[:80],
            exact_match=False,
            bounded_page_available=True,
        ),
        presentation=presentation,
        business_scope=UnscopedBusinessScopeViewV1(),
        effective_policy=policy,
        intent_gate=IntentGateViewV1(
            artifact_hint="knowledge_answer",
            outbound_action_hint=False,
            material_pack_option_count=100_000,
            compliance_hint="clean",
            confidence=0.9,
        ),
        eligible_capabilities=planner_capabilities(),
        recall=recall(),
        retry_overlay=None,
    )
    knowledge = KnowledgeComposerPromptInputSourceV1(
        scene="group",
        message=message,
        history=history,
        runtime_clock=clock,
        presentation=presentation,
        validated_plan=plan,
        directive=knowledge_directive,
        output_ceilings=ceilings,
        selected_capabilities=composer_capabilities(
            "knowledge_composer", KNOWLEDGE_REF
        ),
        preflight=(),
        unit_groundings=groundings,
        retry_overlay=None,
    )
    smalltalk = SmalltalkComposerPromptInputSourceV1(
        scene="group",
        message=message,
        history=history,
        presentation=presentation,
        directive=smalltalk_directive,
        output_ceilings=ceilings,
        selected_capabilities=composer_capabilities(
            "smalltalk_composer", SMALLTALK_REF
        ),
        guardrails=(),
        retry_overlay=ComposerRetryOverlayV1(attempt=1, feedback=text[:300]),
    )
    verifier = AlignmentVerifierPromptInputSourceV1(
        scene="group",
        message=message,
        history=history,
        presentation=presentation,
        validated_plan=plan,
        directive=knowledge_directive,
        selected_capabilities=verifier_capabilities(),
        unit_groundings=groundings,
        candidate=CandidateReplyViewV1(reply_kind="answer", text=text[:4_000]),
        attempt=0,
    )
    return planner, knowledge, smalltalk, verifier
