from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from market_support_crewai_agent.runtime.domain.planning import IntentFrame
from market_support_crewai_agent.schemas import ReplyResponse

PromptStage = Literal["planner_intent", "knowledge_composer", "smalltalk_composer"]
ModelFamily = Literal["ds_v4pro", "deepseek", "gpt", "claude", "generic"]


@dataclass(frozen=True)
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
        response_model=IntentFrame,
        model_family=model_family,
    )


def _composer_profile(model_family: ModelFamily) -> PromptProfile:
    return PromptProfile(
        id=f"knowledge_composer.{model_family}",
        stage="knowledge_composer",
        base_template_name="base.knowledge_composer",
        response_model=ReplyResponse,
        model_family=model_family,
    )


def _smalltalk_composer_profile(model_family: ModelFamily) -> PromptProfile:
    return PromptProfile(
        id=f"smalltalk_composer.{model_family}",
        stage="smalltalk_composer",
        base_template_name="base.smalltalk_composer",
        response_model=ReplyResponse,
        model_family=model_family,
        temperature=0.2,
        max_tokens=300,
    )


PROMPT_PROFILES: tuple[PromptProfile, ...] = tuple(
    profile
    for model_family in ("ds_v4pro", "deepseek", "gpt", "claude", "generic")
    for profile in (
        _planner_profile(model_family),
        _composer_profile(model_family),
        _smalltalk_composer_profile(model_family),
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
    raise ValueError(f"Unknown prompt profile stage: {stage}")


def prompt_profile_by_id(profile_id: str) -> PromptProfile:
    for profile in PROMPT_PROFILES:
        if profile.id == profile_id:
            return profile
    raise ValueError(f"Unknown prompt profile id: {profile_id}")
