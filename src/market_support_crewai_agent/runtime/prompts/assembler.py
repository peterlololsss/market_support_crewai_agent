from __future__ import annotations

import hashlib
from dataclasses import dataclass

from market_support_crewai_agent.runtime.hashing import (
    hash_canonical_model,
)
from market_support_crewai_agent.runtime.hashing import (
    hph1 as prompt_hph1,
)
from market_support_crewai_agent.runtime.policy.compliance import (
    compliance_policy_prompt_lines,
)
from market_support_crewai_agent.runtime.prompts.budgets import (
    load_prompt_static_budgets,
)
from market_support_crewai_agent.runtime.prompts.context import (
    StrictStageInputV1,
    render_prompt_context_layers,
)
from market_support_crewai_agent.runtime.prompts.profiles import (
    NeutralPromptStage,
    PromptProfile,
    PromptScene,
    SceneKeyV1,
    UserFacingPromptStage,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    AgentExecutionSpecV1,
    PromptProgramV2,
    active_prompt_source_ids_v2,
    resolve_agent_execution_spec_v1,
)
from market_support_crewai_agent.runtime.prompts.registry import (
    PROMPT_LAYER_ORDER,
    PromptLayer,
    PromptRegistryInvariantError,
    fragment_by_id,
    render_prompt_fragment,
    scene_contract_by_stage,
)
from market_support_crewai_agent.runtime.prompts.topology import (
    neutral_fragment_ids,
)

_SCENE_KEY_BY_PROMPT_SCENE: dict[PromptScene, SceneKeyV1] = {
    "direct": "wecom_direct.v1",
    "group": "wecom_group.v1",
}
_PROMPT_SCENE_BY_SCENE_KEY: dict[SceneKeyV1, PromptScene] = {
    "wecom_direct.v1": "direct",
    "wecom_group.v1": "group",
}
_USER_FACING_STAGE_BY_STAGE: dict[str, UserFacingPromptStage] = {
    "planner_intent": "planner_intent",
    "knowledge_composer": "knowledge_composer",
    "smalltalk_composer": "smalltalk_composer",
    "alignment_verifier": "alignment_verifier",
}


@dataclass(frozen=True, slots=True)
class PromptProgram:
    profile: PromptProfile
    program_id: str
    program_version: str
    agent_execution_spec: AgentExecutionSpecV1
    scene_key: SceneKeyV1
    scene_contract_id: str | None
    scene_contract_version: str | None
    fragment_ids: tuple[str, ...]
    prompt_text: str
    prompt_hash: str
    fragment_hashes: dict[str, str]
    layers: tuple[PromptLayer, ...]
    static_bytes: int
    baseline_bytes: int
    allowed_max_bytes: int
    static_prompt_text: str = ""
    mch1: str | None = None

    @property
    def hph1(self) -> str:
        return prompt_hph1(self.prompt_text, self.agent_execution_spec)


def assemble_prompt_program(
    ctx: StrictStageInputV1,
    profile: PromptProfile,
    fragment_ids: tuple[str, ...],
    program: PromptProgramV2,
) -> PromptProgram:
    return PromptAssembler().assemblePromptProgram(
        ctx,
        profile,
        fragment_ids,
        program,
    )


class PromptAssembler:
    def assembleCanonicalizationPrompt(
        self,
        prompt_id: str,
        *,
        stage: NeutralPromptStage,
        selector_input_json: str,
    ) -> str:
        fragment_ids = neutral_fragment_ids(stage)
        if stage == "llm_health_probe" or fragment_ids[-1] != prompt_id:
            raise PromptRegistryInvariantError("neutral_prompt_id_stage_mismatch")
        return "\n\n".join(
            render_prompt_fragment(
                fragment_id,
                stage,
                selector_input_json=selector_input_json,
            )
            for fragment_id in fragment_ids
        )

    def assemblePromptProgram(
        self,
        ctx: StrictStageInputV1,
        profile: PromptProfile,
        fragment_ids: tuple[str, ...],
        program: PromptProgramV2,
    ) -> PromptProgram:
        return _assemble_prompt_program(ctx, profile, fragment_ids, program)


def _assemble_prompt_program(
    ctx: StrictStageInputV1,
    profile: PromptProfile,
    fragment_ids: tuple[str, ...],
    program: PromptProgramV2,
) -> PromptProgram:
    if profile.stage != program.stage:
        raise PromptRegistryInvariantError("prompt_program_stage_mismatch")
    execution_spec = resolve_agent_execution_spec_v1(program.program_id)
    scene = ctx.scene
    expected_scene_key = _SCENE_KEY_BY_PROMPT_SCENE[scene]
    if program.scene_key != expected_scene_key:
        raise PromptRegistryInvariantError("prompt_program_scene_mismatch")
    ordered_fragment_ids = _dedupe(fragment_ids)
    if any(
        ordered_fragment_ids.count(source_id) != 1
        for source_id in active_prompt_source_ids_v2(program, profile.model_family)
    ):
        raise PromptRegistryInvariantError("prompt_program_v2_source_mismatch")
    template_context = _template_context()
    sections_by_layer: dict[PromptLayer, list[str]] = {
        layer: [] for layer in PROMPT_LAYER_ORDER
    }
    fragment_hashes: dict[str, str] = {}
    for fragment_id in ordered_fragment_ids:
        fragment = fragment_by_id(fragment_id, profile.stage)
        fragment_text = render_prompt_fragment(
            fragment_id,
            profile.stage,
            **template_context,
        )
        fragment_hashes[fragment_id] = _sha256(fragment_text)
        sections_by_layer[fragment.layer].append(
            f'<prompt_fragment id="{fragment_id}">\n{fragment_text}\n</prompt_fragment>'
        )

    static_prompt_text, _ = _render_sections(sections_by_layer)

    context_layers = render_prompt_context_layers(ctx)
    for layer, text in context_layers.items():
        if text.strip():
            sections_by_layer[layer].append(text.strip())

    prompt_text, layers = _render_sections(sections_by_layer)
    user_facing_stage = _USER_FACING_STAGE_BY_STAGE.get(profile.stage)
    scene = _PROMPT_SCENE_BY_SCENE_KEY.get(program.scene_key)
    if user_facing_stage is None or scene is None:
        raise PromptRegistryInvariantError("scene_prompt_stage_required")
    scene_contract = scene_contract_by_stage(user_facing_stage, scene)
    if (
        scene_contract.contract_id != program.scene_contract_id
        or scene_contract.version != program.scene_contract_version
    ):
        raise PromptRegistryInvariantError("prompt_program_scene_contract_mismatch")
    budget = load_prompt_static_budgets().row_for(
        program.program_id,
        profile.model_family,
        program.scene_key,
    )
    static_bytes = len(static_prompt_text.encode("utf-8"))
    budget.assert_within_budget(static_bytes)
    if static_bytes != budget.baseline_bytes:
        raise PromptRegistryInvariantError("prompt_static_budget_baseline_mismatch")
    return PromptProgram(
        profile=profile,
        program_id=program.program_id,
        program_version=program.program_version,
        agent_execution_spec=execution_spec,
        scene_key=program.scene_key,
        scene_contract_id=scene_contract.contract_id,
        scene_contract_version=scene_contract.version,
        fragment_ids=ordered_fragment_ids,
        prompt_text=prompt_text,
        prompt_hash=_sha256(prompt_text),
        fragment_hashes=fragment_hashes,
        layers=layers,
        static_bytes=static_bytes,
        baseline_bytes=budget.baseline_bytes,
        allowed_max_bytes=budget.allowed_max_bytes,
        static_prompt_text=static_prompt_text,
        mch1=hash_canonical_model(
            "model-visible-context.v1",
            ctx,
            prefix="mch1",
        ),
    )


def _render_sections(
    sections_by_layer: dict[PromptLayer, list[str]],
) -> tuple[str, tuple[PromptLayer, ...]]:
    active_layers: list[PromptLayer] = []
    for layer in PROMPT_LAYER_ORDER:
        if sections_by_layer[layer]:
            active_layers.append(layer)
    layers = tuple(active_layers)
    sections = tuple(
        '<prompt_layer id="{}">\n{}\n</prompt_layer>'.format(
            layer,
            "\n\n".join(sections_by_layer[layer]),
        )
        for layer in layers
    )
    return "\n\n".join(sections), layers


def _template_context() -> dict[str, str]:
    return {
        "compliance_policy_lines": "\n".join(compliance_policy_prompt_lines()),
    }


def _dedupe(fragment_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for fragment_id in fragment_ids:
        if fragment_id in seen:
            continue
        seen.add(fragment_id)
        output.append(fragment_id)
    return tuple(output)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def assembleCanonicalizationPrompt(
    prompt_id: str,
    *,
    stage: NeutralPromptStage,
    selector_input_json: str,
) -> str:
    return PromptAssembler().assembleCanonicalizationPrompt(
        prompt_id,
        stage=stage,
        selector_input_json=selector_input_json,
    )
