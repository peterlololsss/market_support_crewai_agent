from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import BaseModel

from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)

UserFacingPromptStage = Literal[
    "planner_intent",
    "knowledge_composer",
    "smalltalk_composer",
    "alignment_verifier",
]
NeutralPromptStage = Literal[
    "document_product_selector",
    "approved_knowledge_selector",
    "llm_health_probe",
]
PromptStage: TypeAlias = UserFacingPromptStage | NeutralPromptStage
PromptScene = Literal["group", "direct"]
SceneKeyV1 = Literal["wecom_group.v1", "wecom_direct.v1", "scene_neutral.v1"]
ModelFamily = Literal["ds_v4pro", "deepseek", "gpt", "claude", "generic"]


class PromptProfileError(ValueError):
    def __init__(self, code: str) -> None:
        self.code: str = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PromptProfile:
    id: str
    stage: PromptStage
    base_template_name: str
    response_model: type[BaseModel]
    model_family: ModelFamily
    temperature: float | None = None
    max_tokens: int | None = None


def _planner_profile(model_family: ModelFamily) -> PromptProfile:
    return PromptProfile(
        id=f"planner_intent.{model_family}",
        stage="planner_intent",
        base_template_name="base.planner_intent",
        response_model=PlanSpec,
        model_family=model_family,
    )


def _composer_profile(model_family: ModelFamily) -> PromptProfile:
    return PromptProfile(
        id=f"knowledge_composer.{model_family}",
        stage="knowledge_composer",
        base_template_name="base.knowledge_composer",
        response_model=ComposerReplyOutput,
        model_family=model_family,
    )


def _smalltalk_composer_profile(model_family: ModelFamily) -> PromptProfile:
    return PromptProfile(
        id=f"smalltalk_composer.{model_family}",
        stage="smalltalk_composer",
        base_template_name="base.smalltalk_composer",
        response_model=ComposerReplyOutput,
        model_family=model_family,
        temperature=0.2,
        max_tokens=300,
    )


def _alignment_verifier_profile(model_family: ModelFamily) -> PromptProfile:
    return PromptProfile(
        id=f"alignment_verifier.{model_family}",
        stage="alignment_verifier",
        base_template_name="base.alignment_verifier",
        response_model=ReplyAlignmentVerdict,
        model_family=model_family,
        temperature=0.0,
        max_tokens=1200,
    )


PROMPT_PROFILES: tuple[PromptProfile, ...] = tuple(
    profile
    for model_family in ("ds_v4pro", "deepseek", "gpt", "claude", "generic")
    for profile in (
        _planner_profile(model_family),
        _composer_profile(model_family),
        _smalltalk_composer_profile(model_family),
        _alignment_verifier_profile(model_family),
    )
)


def prompt_profile_by_stage(
    stage: PromptStage,
    model_family: ModelFamily = "generic",
) -> PromptProfile:
    for profile in PROMPT_PROFILES:
        if profile.stage == stage and profile.model_family == model_family:
            return profile
    if model_family != "generic":
        return prompt_profile_by_stage(stage, "generic")
    raise PromptProfileError("unknown_prompt_profile_stage")


def prompt_profile_by_id(profile_id: str) -> PromptProfile:
    for profile in PROMPT_PROFILES:
        if profile.id == profile_id:
            return profile
    raise PromptProfileError("unknown_prompt_profile_id")
