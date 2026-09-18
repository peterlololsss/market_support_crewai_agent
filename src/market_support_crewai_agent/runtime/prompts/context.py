from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field

from market_support_crewai_agent.runtime.context.models import prompt_json
from market_support_crewai_agent.runtime.context.stage_inputs import (
    AlignmentVerifierPromptInputV1,
    KnowledgeComposerPromptInputV1,
    PlannerPromptInputV1,
    SmalltalkComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.prompts.registry import PromptLayer
from market_support_crewai_agent.schemas.base import StrictModel

StrictStageInputV1: TypeAlias = (
    PlannerPromptInputV1
    | KnowledgeComposerPromptInputV1
    | SmalltalkComposerPromptInputV1
    | AlignmentVerifierPromptInputV1
)


class IntentGateResult(StrictModel):
    contract_version: Literal["intent-gate"] = "intent-gate"
    artifact_hint: Literal[
        "material_pack",
        "weekly_report",
        "monthly_report",
        "knowledge_answer",
        "human_support",
        "refusal",
        "unclear",
        "smalltalk",
    ]
    outbound_action_hint: bool = False
    material_pack_option_count: int = 0
    compliance_hint: Literal["clean", "risky", "blocked", "unknown"] = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def render_prompt_context(ctx: StrictStageInputV1) -> str:
    return "\n\n".join(
        text for text in render_prompt_context_layers(ctx).values() if text.strip()
    )


def render_prompt_context_layers(
    ctx: StrictStageInputV1,
) -> dict[PromptLayer, str]:
    return {
        "stable": "",
        "domain": "",
        "runtime": "Strict Stage Input JSON:\n{}".format(
            prompt_json(ctx.model_dump(mode="json"))
        ),
        "task": "",
        "ephemeral": "",
    }
