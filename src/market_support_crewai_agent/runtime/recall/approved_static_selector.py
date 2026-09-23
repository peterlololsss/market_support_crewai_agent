from __future__ import annotations

from typing import ClassVar, Literal, Protocol

from pydantic import ConfigDict, Field

from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.prompts.assembler import (
    assemble_canonicalization_prompt,
)
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    ApprovedImageAssetCandidateViewV1,
    ApprovedKnowledgeCandidateViewV1,
    ApprovedKnowledgeSelectorInputV1,
)
from market_support_crewai_agent.runtime.prompts.direct_provider_client import (
    run_direct_provider_text,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    SelectorCacheIdentityV1,
    TurnSelectorOutcomeCacheV1,
    resolve_turn_selector_outcome_cache,
    selector_input_hash,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderJsonSchemaFormatV1,
    require_active_prompt_program_v2,
    resolve_agent_execution_spec_v1,
    synthesize_direct_provider_messages,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    build_provider_target_from_settings,
)
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings


class _FrozenSelectorModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ApprovedImageAssetCandidate(_FrozenSelectorModel):
    asset_id: str
    title: str
    semantic_purpose: str
    usage_notes: str = ""


class ApprovedKnowledgeCandidate(_FrozenSelectorModel):
    entry_id: str
    manifest_ref: ManifestRefV1
    title: str
    semantic_purpose: str
    user_request_examples: tuple[str, ...] = ()
    image_assets: tuple[ApprovedImageAssetCandidate, ...] = ()


class ApprovedKnowledgeSelection(_FrozenSelectorModel):
    selected_entry_ids: tuple[str, ...] = ()
    selected_image_asset_ids: tuple[str, ...] = ()
    confidence: Literal["none", "low", "medium", "high"] = "none"
    rationale: str = Field(default="", max_length=600)


class ApprovedKnowledgeSelector(Protocol):
    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection: ...


class NoopApprovedKnowledgeSelector:
    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection:
        del user_message, evidence_query, catalog_manifest, max_entries, max_images
        return ApprovedKnowledgeSelection(confidence="none")


class DirectApprovedKnowledgeSelector:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        selector_cache: TurnSelectorOutcomeCacheV1 | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.selector_cache: TurnSelectorOutcomeCacheV1 | None = selector_cache

    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection:
        if not self.settings.llm_api_key:
            return ApprovedKnowledgeSelection(confidence="none")
        if not catalog_manifest or not (user_message or evidence_query):
            return ApprovedKnowledgeSelection(confidence="none")
        selector_input = _selector_input(
            user_message=user_message,
            evidence_query=evidence_query,
            catalog_manifest=catalog_manifest,
            max_entries=max_entries,
            max_images=max_images,
        )
        return await _run_direct_selector(
            selector_input,
            self.settings,
            selector_cache=self.selector_cache,
        )


def _selector_input(
    *,
    user_message: str,
    evidence_query: str,
    catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
    max_entries: int,
    max_images: int,
) -> ApprovedKnowledgeSelectorInputV1:
    return ApprovedKnowledgeSelectorInputV1(
        user_query=user_message,
        evidence_query=evidence_query[:200],
        candidates=tuple(
            _candidate_view(candidate) for candidate in catalog_manifest[:20]
        ),
        max_entries=max(1, min(max_entries, 5)),
        max_images=max(0, min(max_images, 8)),
        selected_manifest_ref=None,
    )


def _selector_prompt() -> str:
    return assemble_canonicalization_prompt(
        "canonicalization.approved_knowledge_selector",
        stage="approved_knowledge_selector",
        selector_input_json="",
    )


async def _run_direct_selector(
    selector_input: ApprovedKnowledgeSelectorInputV1,
    settings: Settings,
    *,
    selector_cache: TurnSelectorOutcomeCacheV1 | None = None,
) -> ApprovedKnowledgeSelection:
    selector_cache = resolve_turn_selector_outcome_cache(selector_cache)
    active_program = require_active_prompt_program_v2(
        program_id="approved_knowledge_selector.scene_neutral.v1@1",
        stage="approved_knowledge_selector",
        scene_key="scene_neutral.v1",
        scene_contract_id=None,
        scene_contract_version=None,
    )
    target = build_provider_target_from_settings(
        provider=settings.llm_provider,
        target_slot="selector",
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key_configured=settings.llm_api_key is not None,
        timeout_seconds=min(settings.llm_timeout_seconds, 30.0),
        temperature=0.0,
        max_tokens=min(settings.llm_max_tokens, 1200),
    )
    sih1 = selector_input_hash(
        SelectorCacheIdentityV1(
            selector_input_json=selector_input.model_dump_json(exclude_none=False),
            program_id=active_program.program_id,
            program_version=active_program.program_version,
            provider_target=target.identity(),
        )
    )
    if selector_cache is not None:
        cached = selector_cache.get(sih1)
        if cached is not None:
            return ApprovedKnowledgeSelection.model_validate(cached)
    synthesis = synthesize_direct_provider_messages(
        program=active_program,
        execution_spec=resolve_agent_execution_spec_v1(active_program.program_id),
        prompt_text=_selector_prompt(),
        stage_input=selector_input,
        output_schema=ProviderJsonSchemaFormatV1(
            name="ApprovedKnowledgeSelection",
            json_schema=ApprovedKnowledgeSelection.model_json_schema(),
        ),
    )
    if settings.llm_api_key is None:
        return ApprovedKnowledgeSelection(confidence="none")
    text = await run_direct_provider_text(
        synthesis=synthesis,
        target=target,
        api_key=settings.llm_api_key,
        response_model=ApprovedKnowledgeSelection,
    )
    selection = ApprovedKnowledgeSelection.model_validate_json(text)
    if selector_cache is not None:
        selector_cache.put(sih1, selection.model_dump(mode="json"))
    return selection


def _candidate_view(
    candidate: ApprovedKnowledgeCandidate,
) -> ApprovedKnowledgeCandidateViewV1:
    return ApprovedKnowledgeCandidateViewV1(
        entry_id=candidate.entry_id,
        question=_bounded_text(
            candidate.user_request_examples[0]
            if candidate.user_request_examples
            else "",
            fallback=candidate.title,
            max_chars=120,
        ),
        title=_bounded_text(
            candidate.title, fallback=candidate.entry_id, max_chars=120
        ),
        semantic_purpose=_bounded_text(
            candidate.semantic_purpose,
            fallback=candidate.title,
            max_chars=120,
        ),
        manifest_ref=candidate.manifest_ref,
        fact_type="document_context",
        image_assets=tuple(
            ApprovedImageAssetCandidateViewV1(
                asset_id=asset.asset_id,
                semantic_label=_bounded_text(
                    asset.semantic_purpose or asset.title,
                    fallback=asset.asset_id,
                    max_chars=120,
                ),
            )
            for asset in candidate.image_assets[:8]
        ),
    )


def _bounded_text(value: str, *, fallback: str, max_chars: int) -> str:
    text = (value or fallback).strip() or fallback
    return text[:max_chars]
