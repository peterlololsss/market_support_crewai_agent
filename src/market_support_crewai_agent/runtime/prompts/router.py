from __future__ import annotations

from market_support_crewai_agent.runtime.context.stage_inputs import (
    AlignmentVerifierPromptInputV1,
    KnowledgeComposerPromptInputV1,
    PlannerPromptInputV1,
    SmalltalkComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.prompts.assembler import (
    PromptProgram,
    assemble_prompt_program,
)
from market_support_crewai_agent.runtime.prompts.context import (
    IntentGateResult,
)
from market_support_crewai_agent.runtime.prompts.profiles import (
    ModelFamily,
    PromptScene,
    PromptProfileError,
    SceneKeyV1,
    UserFacingPromptStage,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    PromptProgramV2,
    active_prompt_source_ids_v2,
    resolve_active_prompt_program_v2,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.settings_model import Settings

_SCENE_KEY_BY_PROMPT_SCENE: dict[PromptScene, SceneKeyV1] = {
    "direct": "wecom_direct.v1",
    "group": "wecom_group.v1",
}
_USER_FACING_STAGE_BY_INPUT_TYPE: dict[
    type[
        PlannerPromptInputV1
        | KnowledgeComposerPromptInputV1
        | SmalltalkComposerPromptInputV1
        | AlignmentVerifierPromptInputV1
    ],
    UserFacingPromptStage,
] = {
    PlannerPromptInputV1: "planner_intent",
    KnowledgeComposerPromptInputV1: "knowledge_composer",
    SmalltalkComposerPromptInputV1: "smalltalk_composer",
    AlignmentVerifierPromptInputV1: "alignment_verifier",
}


def model_family_from_settings(
    settings: Settings,
    *,
    stage: str | None = None,
) -> ModelFamily:
    model = (
        settings.planner_llm_model if stage == "planner_intent" else settings.llm_model
    ).lower()
    if "deepseek-v4-pro" in model or "ds-v4pro" in model or "v4-pro" in model:
        return "ds_v4pro"
    if "deepseek" in model:
        return "deepseek"
    if "gpt" in model:
        return "gpt"
    if "claude" in model:
        return "claude"
    return "generic"


def route_intent(
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    history: list[ConversationMessage] | None = None,
) -> IntentGateResult:
    """Audit hint only.

    The planner LLM is the semantic router. This function must not infer artifact
    kind, compliance status, or outbound-action intent from message substrings.
    """
    del history
    return IntentGateResult(
        artifact_hint="unclear",
        outbound_action_hint=False,
        material_pack_option_count=_material_pack_option_count(request, policy),
        compliance_hint="unknown",
        confidence=0.0,
    )


def select_stage_input_prompt_program(
    input_value: (
        PlannerPromptInputV1
        | KnowledgeComposerPromptInputV1
        | SmalltalkComposerPromptInputV1
        | AlignmentVerifierPromptInputV1
    ),
    model_family: ModelFamily,
) -> PromptProgram:
    stage = _USER_FACING_STAGE_BY_INPUT_TYPE[type(input_value)]
    program, _execution_spec = resolve_active_prompt_program_v2(
        stage=stage,
        scene_key=_scene_key(input_value.scene),
    )
    return assemble_prompt_program(
        input_value,
        prompt_profile_by_stage(stage, model_family),
        _fragment_ids_for_program(program, model_family),
        program,
    )


def user_facing_fragment_ids(
    stage: UserFacingPromptStage,
    model_family: ModelFamily,
    scene: PromptScene,
) -> tuple[str, ...]:
    program, _execution_spec = resolve_active_prompt_program_v2(
        stage=stage,
        scene_key=_scene_key(scene),
    )
    return _fragment_ids_for_program(program, model_family)


def _material_pack_option_count(
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
) -> int:
    del request
    return len(policy.material_pack_options)


def _scene_key(scene: PromptScene) -> SceneKeyV1:
    return _SCENE_KEY_BY_PROMPT_SCENE[scene]


def _fragment_ids_for_program(
    program: PromptProgramV2,
    model_family: ModelFamily,
) -> tuple[str, ...]:
    sources = active_prompt_source_ids_v2(program, model_family)
    precedence = tuple(
        source_id for source_id in sources if not source_id.startswith("scene.")
    )
    scene = tuple(source_id for source_id in sources if source_id.startswith("scene."))
    model_overlay = ("model.ds_v4pro.structured",) if model_family == "ds_v4pro" else ()
    if program.stage == "planner_intent":
        return (
            "base.planner_intent",
            *model_overlay,
            *precedence,
            "planner.intent_taxonomy",
            "output.plan_spec_schema",
            "compliance.reason_codes",
            *scene,
        )
    if program.stage == "knowledge_composer":
        return (
            "base.knowledge_composer",
            *model_overlay,
            *precedence,
            "output.reply_response_no_actions",
            "evidence.document_grounding",
            "style.wecom_concise_zh",
            *scene,
        )
    if program.stage == "smalltalk_composer":
        return (
            "base.smalltalk_composer",
            *model_overlay,
            *precedence,
            "output.reply_response_no_actions",
            "style.wecom_concise_zh",
            *scene,
        )
    if program.stage == "alignment_verifier":
        return (
            "base.alignment_verifier",
            *model_overlay,
            *precedence,
            "output.reply_alignment_verdict_schema",
            *scene,
        )
    raise PromptProfileError("unsupported_user_facing_prompt_stage")
