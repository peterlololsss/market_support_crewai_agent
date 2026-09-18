from __future__ import annotations

import hashlib

from pydantic import BaseModel

from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.budgets import (
    load_prompt_static_budgets,
)
from market_support_crewai_agent.runtime.prompts.profiles import (
    NeutralPromptStage,
    PromptProfile,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    require_active_prompt_program_v2,
    resolve_agent_execution_spec_v1,
)
from market_support_crewai_agent.runtime.prompts.registry import render_prompt_fragment


def assemble_neutral_prompt_program(
    *,
    stage: NeutralPromptStage,
    response_model: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> PromptProgram:
    program_id = f"{stage}.scene_neutral.v1@1"
    active_program = require_active_prompt_program_v2(
        program_id=program_id,
        stage=stage,
        scene_key="scene_neutral.v1",
        scene_contract_id=None,
        scene_contract_version=None,
    )
    fragment_ids = tuple(source.fragment_id for source in active_program.sources)
    fragment_texts = tuple(
        render_prompt_fragment(fragment_id, stage) for fragment_id in fragment_ids
    )
    prompt_text = "\n\n".join(fragment_texts)
    budget = load_prompt_static_budgets().row_for(
        program_id,
        "generic",
        "scene_neutral.v1",
    )
    static_bytes = len(prompt_text.encode("utf-8"))
    budget.assert_within_budget(static_bytes)
    if static_bytes != budget.baseline_bytes:
        from market_support_crewai_agent.runtime.prompts.registry import (
            PromptRegistryInvariantError,
        )

        raise PromptRegistryInvariantError("prompt_static_budget_baseline_mismatch")
    return PromptProgram(
        profile=PromptProfile(
            id=(
                "llm_health_probe.generic@1"
                if stage == "llm_health_probe"
                else f"{stage}.generic"
            ),
            stage=stage,
            base_template_name="",
            response_model=response_model,
            model_family="generic",
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        program_id=program_id,
        program_version=active_program.program_version,
        agent_execution_spec=resolve_agent_execution_spec_v1(program_id),
        scene_key="scene_neutral.v1",
        scene_contract_id=None,
        scene_contract_version=None,
        fragment_ids=fragment_ids,
        prompt_text=prompt_text,
        prompt_hash=_sha256(prompt_text),
        fragment_hashes={
            fragment_id: _sha256(fragment_text)
            for fragment_id, fragment_text in zip(
                fragment_ids,
                fragment_texts,
                strict=True,
            )
        },
        layers=("stable",),
        static_bytes=static_bytes,
        baseline_bytes=budget.baseline_bytes,
        allowed_max_bytes=budget.allowed_max_bytes,
    )


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
