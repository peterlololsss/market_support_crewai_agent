from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from string import Template
from typing import Final


_PROMPT_PACKAGE: Final = "market_support_crewai_agent.runtime.prompts"


class PromptTemplateName:
    PLANNER_INTENT: Final = "planner_intent"
    KNOWLEDGE_COMPOSER: Final = "knowledge_composer"


@lru_cache(maxsize=None)
def _load_prompt_template(name: str) -> Template:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid prompt template name: {name}")

    resource = files(_PROMPT_PACKAGE).joinpath(f"{name}.md")
    try:
        return Template(resource.read_text(encoding="utf-8").strip())
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown prompt template: {name}") from exc


def render_prompt_template(name: str, **context: str) -> str:
    return _load_prompt_template(name).substitute(context)
