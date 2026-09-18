from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from market_support_crewai_agent.runtime.prompts.profiles import (
    PromptStage,
)

PromptLayer = Literal["stable", "domain", "runtime", "task", "ephemeral"]
PresentationField = Literal["conversation_name", "principal_name"]
PresentationRuleId = Literal[
    "address_group_audience",
    "use_optional_principal_name",
    "allow_current_conversation_label",
    "address_individual",
    "forbid_group_addressing",
]


@dataclass(frozen=True, slots=True)
class PromptFragment:
    id: str
    stage: PromptStage
    layer: PromptLayer
    priority: int
    template_name: str
    required: bool = False
    conflict_tags: frozenset[str] = frozenset()
    token_budget_hint: int | None = None


@dataclass(frozen=True, slots=True)
class PromptAgentSpec:
    id: str
    role: str
    goal: str
    backstory: str
