from __future__ import annotations

import hashlib
from dataclasses import dataclass

from market_support_crewai_agent.runtime.domain.compliance_policy import (
    compliance_policy_prompt_lines,
)
from market_support_crewai_agent.runtime.context.models import stable_json
from market_support_crewai_agent.runtime.llm.prompting.context import (
    PromptAssemblyContext,
    render_prompt_context_layers,
)
from market_support_crewai_agent.runtime.llm.prompting.registry import (
    PROMPT_LAYER_ORDER,
    PromptLayer,
    fragment_by_id,
    render_prompt_fragment,
)
from market_support_crewai_agent.runtime.llm.prompting.profiles import PromptProfile


@dataclass(frozen=True)
class PromptProgram:
    profile: PromptProfile
    fragment_ids: tuple[str, ...]
    prompt_text: str
    prompt_hash: str
    fragment_hashes: dict[str, str]
    layers: tuple[PromptLayer, ...]
    projection_id: str | None = None
    projection_pressure: dict | None = None
    projection_decision_count: int = 0
    model_visible_context_hash: str | None = None


def assemble_prompt_program(
    ctx: PromptAssemblyContext,
    profile: PromptProfile,
    fragment_ids: tuple[str, ...],
) -> PromptProgram:
    return PromptAssembler().assemblePromptProgram(ctx, profile, fragment_ids)


class PromptAssembler:
    def assemblePlannerPrompt(self, ctx: PromptAssemblyContext) -> PromptProgram:
        from market_support_crewai_agent.runtime.llm.prompting.router import (
            planner_fragment_ids,
        )
        from market_support_crewai_agent.runtime.llm.prompting.profiles import (
            prompt_profile_by_stage,
        )

        return self.assemblePromptProgram(
            ctx,
            prompt_profile_by_stage("planner_intent", ctx.model_family),
            tuple(planner_fragment_ids(ctx)),
        )

    def assembleAgentPrompt(self, ctx: PromptAssemblyContext) -> PromptProgram:
        from market_support_crewai_agent.runtime.llm.prompting.router import (
            agent_fragment_ids,
        )
        from market_support_crewai_agent.runtime.llm.prompting.profiles import (
            prompt_profile_by_stage,
        )

        return self.assemblePromptProgram(
            ctx,
            prompt_profile_by_stage(ctx.stage, ctx.model_family),
            tuple(agent_fragment_ids(ctx)),
        )

    def assembleVerifierPrompt(self, ctx: PromptAssemblyContext) -> PromptProgram:
        from market_support_crewai_agent.runtime.llm.prompting.router import (
            verifier_fragment_ids,
        )
        from market_support_crewai_agent.runtime.llm.prompting.profiles import (
            prompt_profile_by_stage,
        )

        return self.assemblePromptProgram(
            ctx,
            prompt_profile_by_stage("alignment_verifier", ctx.model_family),
            tuple(verifier_fragment_ids(ctx)),
        )

    def assembleCanonicalizationPrompt(
        self,
        prompt_id: str,
        *,
        stage: str,
        selector_input_json: str,
    ) -> str:
        return render_prompt_fragment(
            prompt_id,
            stage,  # type: ignore[arg-type]
            selector_input_json=selector_input_json,
        )

    def assembleGuardrailPrompt(
        self,
        prompt_id: str,
        *,
        stage: str,
        verifier_input_json: str,
    ) -> str:
        return render_prompt_fragment(
            prompt_id,
            stage,  # type: ignore[arg-type]
            verifier_input_json=verifier_input_json,
        )

    def assemblePromptProgram(
        self,
        ctx: PromptAssemblyContext,
        profile: PromptProfile,
        fragment_ids: tuple[str, ...],
    ) -> PromptProgram:
        return _assemble_prompt_program(ctx, profile, fragment_ids)


def _assemble_prompt_program(
    ctx: PromptAssemblyContext,
    profile: PromptProfile,
    fragment_ids: tuple[str, ...],
) -> PromptProgram:
    ordered_fragment_ids = _dedupe(fragment_ids)
    template_context = _template_context()
    sections_by_layer: dict[PromptLayer, list[str]] = {
        layer: [] for layer in PROMPT_LAYER_ORDER
    }
    fragment_hashes: dict[str, str] = {}
    for fragment_id in ordered_fragment_ids:
        fragment = fragment_by_id(fragment_id, ctx.stage)
        fragment_text = render_prompt_fragment(
            fragment_id,
            ctx.stage,
            **template_context,
        )
        fragment_hashes[fragment_id] = _sha256(fragment_text)
        sections_by_layer[fragment.layer].append(
            '<prompt_fragment id="{}">\n{}\n</prompt_fragment>'.format(
                fragment_id,
                fragment_text,
            )
        )

    context_layers = render_prompt_context_layers(ctx)
    for layer, text in context_layers.items():
        if text.strip():
            sections_by_layer[layer].append(text.strip())

    sections: list[str] = []
    layers: list[PromptLayer] = []
    for layer in PROMPT_LAYER_ORDER:
        layer_sections = sections_by_layer[layer]
        if not layer_sections:
            continue
        layers.append(layer)
        sections.append(
            '<prompt_layer id="{}">\n{}\n</prompt_layer>'.format(
                layer,
                "\n\n".join(layer_sections),
            )
        )
    prompt_text = "\n\n".join(sections)
    return PromptProgram(
        profile=profile,
        fragment_ids=ordered_fragment_ids,
        prompt_text=prompt_text,
        prompt_hash=_sha256(prompt_text),
        fragment_hashes=fragment_hashes,
        layers=tuple(layers),
        projection_id=(
            ctx.model_visible_context.projection_id
            if ctx.model_visible_context is not None
            else None
        ),
        projection_pressure=(
            ctx.model_visible_context.pressure.to_prompt_dict()
            if ctx.model_visible_context is not None
            and ctx.model_visible_context.pressure is not None
            else None
        ),
        projection_decision_count=(
            len(ctx.model_visible_context.decisions)
            if ctx.model_visible_context is not None
            else 0
        ),
        model_visible_context_hash=(
            _sha256(stable_json(ctx.model_visible_context.to_prompt_runtime_payload()))
            if ctx.model_visible_context is not None
            else None
        ),
    )


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


def assemblePlannerPrompt(ctx: PromptAssemblyContext) -> PromptProgram:
    return PromptAssembler().assemblePlannerPrompt(ctx)


def assembleAgentPrompt(ctx: PromptAssemblyContext) -> PromptProgram:
    return PromptAssembler().assembleAgentPrompt(ctx)


def assembleVerifierPrompt(ctx: PromptAssemblyContext) -> PromptProgram:
    return PromptAssembler().assembleVerifierPrompt(ctx)


def assembleCanonicalizationPrompt(
    prompt_id: str,
    *,
    stage: str,
    selector_input_json: str,
) -> str:
    return PromptAssembler().assembleCanonicalizationPrompt(
        prompt_id,
        stage=stage,
        selector_input_json=selector_input_json,
    )


def assembleGuardrailPrompt(
    prompt_id: str,
    *,
    stage: str,
    verifier_input_json: str,
) -> str:
    return PromptAssembler().assembleGuardrailPrompt(
        prompt_id,
        stage=stage,
        verifier_input_json=verifier_input_json,
    )


def assemble_planner_prompt(ctx: PromptAssemblyContext) -> PromptProgram:
    return assemblePlannerPrompt(ctx)


def assemble_agent_prompt(ctx: PromptAssemblyContext) -> PromptProgram:
    return assembleAgentPrompt(ctx)


def assemble_verifier_prompt(ctx: PromptAssemblyContext) -> PromptProgram:
    return assembleVerifierPrompt(ctx)


def assemble_canonicalization_prompt(
    prompt_id: str,
    *,
    stage: str,
    selector_input_json: str,
) -> str:
    return assembleCanonicalizationPrompt(
        prompt_id,
        stage=stage,
        selector_input_json=selector_input_json,
    )


def assemble_guardrail_prompt(
    prompt_id: str,
    *,
    stage: str,
    verifier_input_json: str,
) -> str:
    return assembleGuardrailPrompt(
        prompt_id,
        stage=stage,
        verifier_input_json=verifier_input_json,
    )
