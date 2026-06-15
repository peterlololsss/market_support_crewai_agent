from __future__ import annotations

from dataclasses import replace

from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.llm.prompt_assembler import assemble_prompt_program
from market_support_crewai_agent.runtime.llm.prompt_context import PromptAssemblyContext
from market_support_crewai_agent.runtime.llm.prompt_profiles import prompt_profile_by_stage
from market_support_crewai_agent.runtime.llm.prompt_router import (
    route_intent,
    select_prompt_program,
)
from market_support_crewai_agent.schemas import (
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
    SendWeeklyReportAction,
)


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
        "available_strategies": ["中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_ctx(message: str = "发一下中证1000材料", stage: str = "planner_intent"):
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


def test_prompt_contains_ordered_fragment_sections():
    ctx = make_ctx()
    profile = prompt_profile_by_stage("planner_intent", "ds_v4pro")
    program = assemble_prompt_program(
        ctx,
        profile,
        (
            "base.planner_intent",
            "model.ds_v4pro.structured",
            "output.intent_frame_schema",
        ),
    )

    first = program.prompt_text.index('<prompt_fragment id="base.planner_intent">')
    second = program.prompt_text.index('<prompt_fragment id="model.ds_v4pro.structured">')
    third = program.prompt_text.index('<prompt_fragment id="output.intent_frame_schema">')
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
    import market_support_crewai_agent.runtime.llm.prompt_assembler as assembler

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

    assert "IntentFrame compact schema:" in program.prompt_text
    assert '"artifact_kind": "material_pack|weekly_report|monthly_report|knowledge_answer|human_support|refusal|unclear|smalltalk"' in program.prompt_text
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

    assert "ReplyResponse compact no-action schema:" in program.prompt_text
    assert '"actions": []' in program.prompt_text
    assert "Canonical JSON schema:" not in program.prompt_text
    assert '"$defs"' not in program.prompt_text
    assert '"properties"' not in program.prompt_text


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
                report_scope="channel_all",
                strategy=None,
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
    canonical_context = canonicalize_request(request)
    policy = compile_policy(request, doc_mcp_enabled=True)
    program = select_prompt_program(
        PromptAssemblyContext(
            stage="knowledge_composer",
            model_family="ds_v4pro",
            request=request,
            canonical_context=canonical_context,
            policy=policy,
            intent_gate=route_intent(request, canonical_context, policy),
        )
    )

    assert "capability.material_pack" not in program.fragment_ids
    assert "capability.weekly_report" not in program.fragment_ids
    assert "capability.monthly_report" not in program.fragment_ids
    assert "examples.material_pack" not in program.fragment_ids
    assert "examples.report_scope" not in program.fragment_ids


def route_and_select(message: str):
    request = make_request(message)
    canonical_context = canonicalize_request(request)
    policy = compile_policy(request, doc_mcp_enabled=True)
    gate = route_intent(request, canonical_context, policy)
    program = select_prompt_program(
        PromptAssemblyContext(
            stage="planner_intent",
            model_family="ds_v4pro",
            request=request,
            canonical_context=canonical_context,
            policy=policy,
            intent_gate=gate,
        )
    )
    return gate, program
