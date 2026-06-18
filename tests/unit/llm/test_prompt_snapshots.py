from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.llm.prompting.assembler import (
    assembleCanonicalizationPrompt,
    assembleGuardrailPrompt,
)
from market_support_crewai_agent.runtime.llm.prompting.context import PromptAssemblyContext
from market_support_crewai_agent.runtime.llm.prompting.router import (
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
    DisallowedEvidence,
)
from market_support_crewai_agent.schemas import (
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
    SendWeeklyReportAction,
)

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "snapshots" / "prompts"


def make_request(message: str = "发一下中证1000材料", **overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": message,
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_materials": ["material", "weekly", "monthly"],
        "material_pack_options": ["中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_context(
    message: str = "发一下中证1000材料",
    *,
    stage: str = "planner_intent",
) -> PromptAssemblyContext:
    request = make_request(message)
    canonical_context = canonicalize_request(request)
    policy = compile_policy(request, doc_mcp_enabled=True)
    return PromptAssemblyContext(
        stage=stage,  # type: ignore[arg-type]
        model_family="ds_v4pro",
        request=request,
        canonical_context=canonical_context,
        policy=policy,
        intent_gate=route_intent(request, canonical_context, policy),
    )


def test_planner_prompt_snapshot():
    program = select_prompt_program(make_context())

    assert_snapshot("planner_intent_ds_v4pro.txt", program.prompt_text)


def test_knowledge_composer_prompt_snapshot():
    ctx = make_context(stage="knowledge_composer")
    assessment = AnswerabilityAssessment(
        can_answer=False,
        capability_id="material_pack.product_list",
        required_artifacts=["material_pack"],
        missing_artifacts=["material_pack"],
        required_runtime_inputs=["request.dist_channel_name"],
        disallowed_evidence_ids=[
            DisallowedEvidence(
                evidence_id="adapter_report_scope:weekly_report:report_scope_products",
                reason="source_type_not_allowed",
            )
        ],
        ambiguity="unknown_artifact",
        recommended_response_mode="abstain",
        user_facing_reason="当前上下文没有可用于回答产品列表的材料包内容，我不能用周报、月报或历史记录替代判断。",
    )
    program = select_prompt_program(
        replace(ctx, answerability_assessment=assessment)
    )

    assert_snapshot("knowledge_composer_boundary.txt", program.prompt_text)


def test_alignment_verifier_prompt_snapshot():
    candidate = ReplyResponse(
        response_id="resp-1",
        reply=PrimaryReply(kind="answer", text=""),
        actions=[
            SendWeeklyReportAction(
                type="send_weekly_report",
                action_id="act-1",
                resolve_type="weekly_report",
                resolve_ref="weekly:resolve-ref",
                period="20260529",
                report_date="2026-05-29",
            )
        ],
    )
    ctx = make_context(stage="alignment_verifier")
    program = select_prompt_program(replace(ctx, candidate_response=candidate))

    assert_snapshot("alignment_verifier.txt", program.prompt_text)


def test_canonicalization_and_guardrail_prompt_snapshots():
    selector_prompt = assembleCanonicalizationPrompt(
        "canonicalization.report_scope_selector",
        stage="report_scope_selector",
        selector_input_json='{"query":"report_scope_products","candidate_products":[]}',
    )
    guardrail_prompt = assembleGuardrailPrompt(
        "guardrail.image_alignment_verifier",
        stage="image_alignment_verifier",
        verifier_input_json='{"reply_image_filenames":["company_shareholders.png"]}',
    )

    assert_snapshot(
        "canonicalization_report_scope_selector.txt",
        selector_prompt,
    )
    assert_snapshot("guardrail_image_alignment_verifier.txt", guardrail_prompt)


def assert_snapshot(name: str, actual: str) -> None:
    path = SNAPSHOT_DIR / name
    if os.getenv("UPDATE_PROMPT_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
    assert actual == path.read_text(encoding="utf-8")
