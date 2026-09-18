from __future__ import annotations

from typing import Protocol, assert_never

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
    SmalltalkComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    approved_image_asset_by_marker,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.composer_projection import (
    ComposerInvocationV1,
    ComposerPromptInputV1,
    build_composer_prompt_input_v1,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    image_marker_filenames,
)


class V2ComposerUnavailable(RuntimeError):
    pass


class V2ComposerOutputRejected(RuntimeError):
    pass


class V2Composer(Protocol):
    async def compose(
        self, input_value: ComposerPromptInputV1
    ) -> ComposerReplyOutput: ...


async def compose_v2_reply(
    runtime: _V2ComposerRuntime,
    source: ComposerInvocationV1,
) -> ComposerReplyOutput:
    composer = runtime.v2_composer
    if composer is None:
        raise V2ComposerUnavailable("v2_composer_not_configured")
    input_value = build_composer_prompt_input_v1(source)
    output = await composer.compose(input_value)
    _validate_v2_composer_output(output, input_value, source)
    return output


class _V2ComposerRuntime(Protocol):
    @property
    def v2_composer(self) -> V2Composer | None: ...


def _validate_v2_composer_output(
    output: ComposerReplyOutput,
    input_value: ComposerPromptInputV1,
    source: ComposerInvocationV1 | None = None,
) -> None:
    if output.response_id != "":
        raise V2ComposerOutputRejected("v2_composer_response_id_forbidden")
    if output.actions:
        raise V2ComposerOutputRejected("v2_composer_actions_forbidden")
    if output.reply.mentions:
        raise V2ComposerOutputRejected("v2_composer_mentions_forbidden")
    if output.reply.kind not in input_value.output_ceilings.allowed_reply_kinds:
        raise V2ComposerOutputRejected("v2_composer_reply_kind_forbidden")
    if len(output.reply.text) > input_value.output_ceilings.max_reply_chars:
        raise V2ComposerOutputRejected("v2_composer_reply_text_too_long")
    cited_ids = set(output.evidence_ids)
    grounding_evidence = _grounding_evidence_sets(input_value)
    if not cited_ids <= set().union(*grounding_evidence):
        raise V2ComposerOutputRejected("v2_composer_unselected_evidence")
    markers = set(image_marker_filenames(output.reply.text))
    marker_bindings = _media_binding_evidence_ids(input_value, source)
    if not markers <= marker_bindings.keys():
        raise V2ComposerOutputRejected("v2_composer_media_forbidden")
    referenced_ids = cited_ids | {
        evidence_id for marker in markers for evidence_id in marker_bindings[marker]
    }
    if referenced_ids and not any(
        referenced_ids <= allowed_ids for allowed_ids in grounding_evidence
    ):
        raise V2ComposerOutputRejected("v2_composer_cross_unit_authority_forbidden")


def _grounding_evidence_sets(
    input_value: ComposerPromptInputV1,
) -> tuple[frozenset[str], ...]:
    match input_value:
        case KnowledgeComposerPromptInputV1(unit_groundings=groundings):
            return tuple(
                frozenset(grounding.allowed_evidence_ids) for grounding in groundings
            )
        case SmalltalkComposerPromptInputV1():
            return (frozenset(),)
        case unreachable:
            assert_never(unreachable)


def _media_binding_evidence_ids(
    input_value: ComposerPromptInputV1,
    source: ComposerInvocationV1 | None,
) -> dict[str, frozenset[str]]:
    if (
        source is None
        or source.request.identity.scene == "direct"
        or not source.policy.internal_company_knowledge_enabled
    ):
        return {}
    match input_value:
        case SmalltalkComposerPromptInputV1():
            return {}
        case KnowledgeComposerPromptInputV1():
            selected_ids = {
                ref.manifest_id
                for ref in input_value.selected_capabilities.manifest_refs
            }
            if "answer_internal_company_knowledge" not in selected_ids:
                return {}
        case unreachable:
            assert_never(unreachable)
    admitted_ids = set().union(*_grounding_evidence_sets(input_value))
    allowed: dict[str, set[str]] = {}
    for binding in source.media_bindings:
        filename = binding.marker.removeprefix("%%").removesuffix("%%")
        asset = approved_image_asset_by_marker(filename)
        if (
            binding.evidence_id in admitted_ids
            and asset is not None
            and asset.asset_id == binding.asset_id
        ):
            allowed.setdefault(filename, set()).add(binding.evidence_id)
    return {
        filename: frozenset(evidence_ids) for filename, evidence_ids in allowed.items()
    }


__all__ = [
    "ComposerInvocationV1",
    "ComposerPromptInputV1",
    "V2Composer",
    "V2ComposerOutputRejected",
    "V2ComposerUnavailable",
    "build_composer_prompt_input_v1",
    "compose_v2_reply",
]
