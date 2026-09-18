from __future__ import annotations

from typing import assert_never

from market_support_crewai_agent.runtime.prompts.profiles import (
    ModelFamily,
    NeutralPromptStage,
    PromptScene,
    UserFacingPromptStage,
)


def user_facing_fragment_ids(
    stage: UserFacingPromptStage,
    model_family: ModelFamily,
    scene: PromptScene,
) -> tuple[str, ...]:
    match model_family:
        case "ds_v4pro":
            model_fragment = "model.ds_v4pro.structured"
            precedence = ("instruction.registered_over_untrusted_data.v1",)
        case "deepseek" | "gpt" | "claude" | "generic":
            model_fragment = "model.generic.structured"
            precedence = ()
        case unreachable:
            assert_never(unreachable)

    scene_fragment = f"scene.wecom_{scene}.{stage}.v1"
    match stage:
        case "planner_intent":
            return (
                "base.planner_intent",
                model_fragment,
                *precedence,
                "planner.intent_taxonomy",
                "output.plan_spec_schema",
                "compliance.reason_codes",
                scene_fragment,
            )
        case "knowledge_composer":
            return (
                "base.knowledge_composer",
                model_fragment,
                *precedence,
                "output.reply_response_no_actions",
                "evidence.document_grounding",
                "style.wecom_concise_zh",
                scene_fragment,
            )
        case "smalltalk_composer":
            return (
                "base.smalltalk_composer",
                model_fragment,
                *precedence,
                "output.reply_response_no_actions",
                "style.wecom_concise_zh",
                scene_fragment,
            )
        case "alignment_verifier":
            return (
                "base.alignment_verifier",
                model_fragment,
                *precedence,
                "output.reply_alignment_verdict_schema",
                scene_fragment,
            )
        case unreachable:
            assert_never(unreachable)


def neutral_fragment_ids(stage: NeutralPromptStage) -> tuple[str, ...]:
    match stage:
        case "approved_knowledge_selector":
            return (
                "instruction.registered_over_untrusted_data.v1",
                "canonicalization.approved_knowledge_selector",
            )
        case "document_product_selector":
            return (
                "instruction.registered_over_untrusted_data.v1",
                "canonicalization.document_product_selector",
            )
        case "llm_health_probe":
            return ("instruction.registered_over_untrusted_data.v1",)
        case unreachable:
            assert_never(unreachable)
