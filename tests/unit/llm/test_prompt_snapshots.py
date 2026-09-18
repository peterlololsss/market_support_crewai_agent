from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from pydantic import TypeAdapter

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.context.stage_inputs import (
    AlignmentVerifierPromptInputV1,
    KnowledgeComposerPromptInputV1,
    PlannerPromptInputV1,
    SmalltalkComposerPromptInputV1,
    build_alignment_verifier_prompt_input_v1,
    build_knowledge_composer_prompt_input_v1,
    build_planner_prompt_input_v1,
    build_smalltalk_composer_prompt_input_v1,
)
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.policy.manifest import (
    compile_policy_authority_core_v1,
)
from market_support_crewai_agent.runtime.prompts.registry import PROMPT_REGISTRY
from market_support_crewai_agent.runtime.prompts.router import (
    select_stage_input_prompt_program,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    image_marker_filenames,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.unit.llm._stage_input_fixtures import stage_sources


ROOT = Path(__file__).resolve().parents[3]


class _LlmInvocationV1(TypedDict):
    stage_kind: str


class _LlmInvocationInventoryV1(TypedDict):
    invocations: list[_LlmInvocationV1]


def make_stage_inputs(
    message: str = "send the requested material",
) -> tuple[
    PlannerPromptInputV1,
    KnowledgeComposerPromptInputV1,
    SmalltalkComposerPromptInputV1,
    AlignmentVerifierPromptInputV1,
]:
    planner, knowledge, smalltalk, verifier = stage_sources(message)
    return (
        build_planner_prompt_input_v1(planner),
        build_knowledge_composer_prompt_input_v1(knowledge),
        build_smalltalk_composer_prompt_input_v1(smalltalk),
        build_alignment_verifier_prompt_input_v1(verifier),
    )


def test_planner_prompt_snapshot() -> None:
    # Given: a group planner registration and its immutable static budget.
    planner, _, _, _ = make_stage_inputs()
    program = select_stage_input_prompt_program(planner, "ds_v4pro")

    # When/Then: structural registration replaces rendered-prose snapshots.
    assert program.program_id == "planner_intent.wecom_group.v1@1"
    assert program.scene_contract_id == "scene.wecom_group.planner_intent.v1"
    assert program.profile.response_model is PlanSpec
    assert program.layers == ("stable", "domain", "runtime", "task")
    assert program.static_bytes == program.baseline_bytes
    assert program.static_bytes <= program.allowed_max_bytes


def test_knowledge_composer_prompt_snapshot() -> None:
    # Given: the registered group knowledge-composer program.
    _, knowledge, _, _ = make_stage_inputs()
    program = select_stage_input_prompt_program(knowledge, "ds_v4pro")

    # When/Then: one scene contract and the typed output schema are bound.
    assert program.program_id == "knowledge_composer.wecom_group.v1@1"
    assert program.profile.response_model is ComposerReplyOutput
    assert tuple(
        fragment_id
        for fragment_id in program.fragment_ids
        if fragment_id.startswith("scene.")
    ) == ("scene.wecom_group.knowledge_composer.v1",)
    assert program.static_bytes == program.baseline_bytes


def test_prompt_rules_route_company_intro_and_allow_public_urls() -> None:
    # Given: direct internal-knowledge authority and the locator safety boundary.
    request = make_v2_envelope(
        message="介绍下你们公司",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:test",
            "direct_thread_ref": "direct:thread-1",
            "principal_ref": "principal:sender-1",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "test user",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request
    policy = compile_policy_authority_core_v1(
        request,
        business_scope_authority_v1(request.business_scope),
    )
    locator = LocatorSafetyClassifierV1(internal_origins=frozenset(), secrets=())

    # When/Then: company knowledge remains eligible and public evidence URLs survive.
    assert "answer_internal_company_knowledge" in {
        ref.manifest_id for ref in policy.eligible_capabilities
    }
    assert locator.public_url("https://example.com/public/company") == (
        "https://example.com/public/company"
    )
    assert locator.public_url("http://127.0.0.1/internal") is None


def test_alignment_verifier_prompt_snapshot() -> None:
    # Given: the registered group alignment-verifier program.
    _, _, _, verifier = make_stage_inputs()
    program = select_stage_input_prompt_program(verifier, "ds_v4pro")

    # When/Then: verifier schema, scene contract, and sealed budget are structural.
    assert program.program_id == "alignment_verifier.wecom_group.v1@1"
    assert program.profile.response_model is ReplyAlignmentVerdict
    assert program.scene_contract_id == "scene.wecom_group.alignment_verifier.v1"
    assert program.static_bytes == program.baseline_bytes


def test_guardrail_prompt_snapshot() -> None:
    # Given: deterministic marker parsing and the real invocation inventory.
    inventory = TypeAdapter(_LlmInvocationInventoryV1).validate_json(
        (ROOT / "tests/fixtures/real_llm_invocations.v1.json").read_bytes()
    )

    # When/Then: marker validation remains local and no image LLM program exists.
    assert image_marker_filenames("资料：%%company_shareholders.png%%") == [
        "company_shareholders.png"
    ]
    assert "guardrail.image_alignment_verifier" not in PROMPT_REGISTRY.prompt_ids()
    assert "agent.image_alignment_verifier" not in PROMPT_REGISTRY.agent_spec_ids()
    assert "image_alignment_verifier" not in {
        row["stage_kind"] for row in inventory["invocations"]
    }
