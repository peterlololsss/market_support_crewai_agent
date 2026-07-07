from __future__ import annotations

from dataclasses import replace

from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    DomainContextBuilder,
)
from market_support_crewai_agent.runtime.domain.sources.metadata import SourceMetadata
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
    DisallowedEvidence,
)
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.llm.prompting.assembler import assemble_prompt_program
from market_support_crewai_agent.runtime.llm.prompting.context import PromptAssemblyContext
from market_support_crewai_agent.runtime.llm.prompting.profiles import (
    PromptStage,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.runtime.llm.prompting.router import (
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.schemas import (
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
    SendWeeklyReportAction,
)
from tests.helpers.planning import compile_test_plan


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
        "available_artifacts": [
            {"type": "material_pack", "options": ["中证1000"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_ctx(message: str = "发一下中证1000材料", stage: PromptStage = "planner_intent"):
    request = make_request(message)
    policy = compile_policy(request, doc_mcp_enabled=True)
    return PromptAssemblyContext(
        stage=stage,
        model_family="ds_v4pro",
        request=request,
        policy=policy,
        intent_gate=route_intent(request, policy),
    )


def test_prompt_contains_ordered_fragment_sections():
    ctx = make_ctx()
    profile = prompt_profile_by_stage("planner_intent", "ds_v4pro")
    program = assemble_prompt_program(
        ctx,
        profile,
        (
            "base.planner_intent",
            "model.ds_v4pro.structured",
            "output.plan_spec_schema",
        ),
    )

    first = program.prompt_text.index('<prompt_fragment id="base.planner_intent">')
    second = program.prompt_text.index('<prompt_fragment id="model.ds_v4pro.structured">')
    third = program.prompt_text.index('<prompt_fragment id="output.plan_spec_schema">')
    assert first < second < third


def test_duplicate_fragment_ids_are_deduped():
    ctx = make_ctx()
    profile = prompt_profile_by_stage("planner_intent", "ds_v4pro")
    program = assemble_prompt_program(
        ctx,
        profile,
        ("base.planner_intent", "base.planner_intent"),
    )

    assert program.fragment_ids == ("base.planner_intent",)
    assert program.prompt_text.count('<prompt_fragment id="base.planner_intent">') == 1


def test_prompt_hash_changes_when_fragment_content_changes(monkeypatch):
    import market_support_crewai_agent.runtime.llm.prompting.assembler as assembler

    ctx = make_ctx()
    profile = prompt_profile_by_stage("planner_intent", "ds_v4pro")
    fragment_ids = ("base.planner_intent", "model.ds_v4pro.structured")
    original = assemble_prompt_program(ctx, profile, fragment_ids)
    original_render = assembler.render_prompt_fragment

    def changed_render(fragment_id, stage, **context):
        text = original_render(fragment_id, stage, **context)
        if fragment_id == "base.planner_intent":
            return text + "\nChanged fragment text."
        return text

    monkeypatch.setattr(assembler, "render_prompt_fragment", changed_render)
    changed = assemble_prompt_program(ctx, profile, fragment_ids)

    assert changed.prompt_hash != original.prompt_hash
    assert (
        changed.fragment_hashes["base.planner_intent"]
        != original.fragment_hashes["base.planner_intent"]
    )


def test_fragment_hashes_include_every_fragment_id():
    _, program = route_and_select("发下这个渠道的周报")

    assert set(program.fragment_ids) == set(program.fragment_hashes)


def test_raw_resolve_ref_is_not_present_in_assembled_prompt():
    ctx = make_ctx(stage="knowledge_composer")
    ctx.evidence_facts.append(
        EvidenceFact(
            fact_type="weekly_report_resolvable",
            value=True,
            resolve_type="weekly_report",
            metadata={"resolve_ref": "weekly:resolve-ref", "status": "resolved"},
        )
    )
    profile = prompt_profile_by_stage("knowledge_composer", "ds_v4pro")
    program = assemble_prompt_program(
        ctx,
        profile,
        (
            "base.knowledge_composer",
            "model.ds_v4pro.structured",
            "output.reply_response_no_actions",
        ),
    )

    assert "weekly:resolve-ref" not in program.prompt_text
    assert "resolve_ref_available" in program.prompt_text


def test_planner_prompt_uses_compact_schema_skeleton_not_full_schema_dump():
    _, program = route_and_select("发下这个渠道的周报")

    assert "PlanSpec compact schema:" in program.prompt_text
    assert '"plan_units": [' in program.prompt_text
    assert '"selected_capability_id": "one id from Capability registry JSON"' in program.prompt_text
    assert '"answerability_policy": "answer|send|clarify|abstain|refuse|handoff|smalltalk|no_reply"' in program.prompt_text
    assert "Canonical JSON schema:" not in program.prompt_text
    assert '"$defs"' not in program.prompt_text
    assert '"properties"' not in program.prompt_text


def test_composer_prompt_uses_compact_no_action_skeleton_not_full_schema_dump():
    ctx = make_ctx(stage="knowledge_composer")
    profile = prompt_profile_by_stage("knowledge_composer", "ds_v4pro")
    program = assemble_prompt_program(
        ctx,
        profile,
        (
            "base.knowledge_composer",
            "model.ds_v4pro.structured",
            "output.reply_response_no_actions",
        ),
    )

    assert "ComposerReplyOutput compact no-action schema:" in program.prompt_text
    assert '"response_mode": "answer|abstain|clarify"' in program.prompt_text
    assert '"evidence_ids": [' in program.prompt_text
    assert '"actions": []' in program.prompt_text
    assert "never say an action has already been sent" in program.prompt_text
    assert "Canonical JSON schema:" not in program.prompt_text
    assert '"$defs"' not in program.prompt_text
    assert '"properties"' not in program.prompt_text


def test_knowledge_composer_prompt_says_existing_data_not_knowledge_base():
    ctx = make_ctx(stage="knowledge_composer")
    profile = prompt_profile_by_stage("knowledge_composer", "ds_v4pro")
    program = assemble_prompt_program(
        ctx,
        profile,
        (
            "base.knowledge_composer",
            "model.ds_v4pro.structured",
            "output.reply_response_no_actions",
        ),
    )

    assert "知识库" not in program.prompt_text
    assert "根据我已有的数据" in program.prompt_text


def test_knowledge_composer_prompt_includes_runtime_boundary_block_snapshot():
    assessment = AnswerabilityAssessment(
        can_answer=False,
        capability_id="channel.strategy_summary",
        required_artifacts=["document_context"],
        available_matching_artifacts=[],
        missing_artifacts=["document_context"],
        required_runtime_inputs=["request.dist_channel_name"],
        missing_runtime_inputs=[],
        allowed_evidence_ids=[],
        disallowed_evidence_ids=[
            DisallowedEvidence(
                evidence_id="adapter_report_scope:weekly_report:report_scope_products",
                reason="source_type_not_allowed",
            )
        ],
        ambiguity="unknown_artifact",
        recommended_response_mode="abstain",
        user_facing_reason="document evidence missing",
    )
    ctx = replace(
        make_ctx(stage="knowledge_composer"),
        answerability_assessment=assessment,
    )

    program = select_prompt_program(ctx)

    assert program.prompt_text.count("Runtime Capability & Evidence Boundary JSON:") == 1
    assert '"answerability_assessment": {' in program.prompt_text
    assert '"current_channel": {' in program.prompt_text
    assert '"material_pack_routing": {' in program.prompt_text
    assert '"available_artifacts": [' in program.prompt_text
    assert '"capability_id": "channel.strategy_summary"' in program.prompt_text
    assert (
        '"evidence_id": "adapter_report_scope:weekly_report:report_scope_products"'
        in program.prompt_text
    )
    assert (
        '"user_facing_reason": "document evidence missing"'
        in program.prompt_text
    )


def test_knowledge_composer_prompt_separates_allowed_evidence_from_disallowed_context():
    request = make_request(
        "材料包里有哪些产品",
        available_artifacts=[{"type": "material_pack", "options": ["指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
    )
    policy = compile_policy(request, doc_mcp_enabled=True)
    plan = compile_test_plan(
        request,
        policy=policy,
        user_need="answer strategy summary",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["document_context"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal material question",
        },
        confidence=0.9,
    )
    allowed_fact = EvidenceFact(
        fact_type="document_context",
        value=True,
        source_type="document_mcp",
        source_id="doc-current",
        metadata={"documents": [{"text": "Current Product"}]},
        artifact_type="document_context",
        scope=ArtifactScope(channel_id="unknown"),
    )
    stale_history_fact = EvidenceFact(
        fact_type="document_context",
        value=True,
        source_type="conversation_history",
        source_id="old-doc",
        metadata={"documents": [{"text": "Old Product"}]},
        artifact_type="document_context",
        source_metadata=SourceMetadata(
            source_id="old-doc",
            source_type="assistant_message",
            artifact_type="document_context",
            provenance="conversation_store",
            evidence_allowed_by_default=False,
        ),
    )
    facts = [allowed_fact, stale_history_fact]
    domain_context = DomainContextBuilder().build(request, available_artifacts=facts)
    program = select_prompt_program(
        PromptAssemblyContext(
            stage="knowledge_composer",
            model_family="ds_v4pro",
            request=request,
            domain_context=domain_context,
            policy=policy,
            intent_gate=route_intent(request, policy),
            execution_plan=plan,
            evidence_facts=facts,
            business_facts=derive_business_facts(facts, request),
        )
    )

    assert '"title": "Allowed evidence JSON"' in program.prompt_text
    assert '"title": "Disallowed context JSON"' in program.prompt_text
    assert "Current Product" in program.prompt_text
    assert "Old Product" not in program.prompt_text
    assert '"content_redacted": true' in program.prompt_text
    assert '"source_type": "assistant_message"' in program.prompt_text


def test_alignment_verifier_prompt_uses_compact_schema_and_candidate_response():
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
    ctx = replace(make_ctx(stage="alignment_verifier"), candidate_response=candidate)

    program = select_prompt_program(ctx)

    assert "base.alignment_verifier" in program.fragment_ids
    assert "output.reply_alignment_verdict_schema" in program.fragment_ids
    assert "ReplyAlignmentVerdict compact schema:" in program.prompt_text
    assert "Candidate ReplyResponse JSON" in program.prompt_text
    assert "weekly:resolve-ref" not in program.prompt_text
    assert "resolve_ref_available" in program.prompt_text
    assert "Canonical JSON schema:" not in program.prompt_text
    assert '"$defs"' not in program.prompt_text


def test_knowledge_composer_never_receives_action_fragments():
    request = make_request("这个策略怎么样")
    policy = compile_policy(request, doc_mcp_enabled=True)
    program = select_prompt_program(
        PromptAssemblyContext(
            stage="knowledge_composer",
            model_family="ds_v4pro",
            request=request,
            policy=policy,
            intent_gate=route_intent(request, policy),
        )
    )

    assert "capability.material_pack" not in program.fragment_ids
    assert "capability.weekly_report" not in program.fragment_ids
    assert "capability.monthly_report" not in program.fragment_ids
    assert "examples.material_pack" not in program.fragment_ids
    assert "examples.report_scope" not in program.fragment_ids


def route_and_select(message: str):
    request = make_request(message)
    policy = compile_policy(request, doc_mcp_enabled=True)
    gate = route_intent(request, policy)
    program = select_prompt_program(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            policy=policy,
            intent_gate=gate,
        )
    )
    return gate, program
