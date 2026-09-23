from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import Field

from market_support_crewai_agent.runtime.integrations.document_mcp.manifest import (
    DocumentProductCandidate,
    ProductSource,
    product_manifest,
)
from market_support_crewai_agent.runtime.prompts.assembler import (
    assemble_canonicalization_prompt,
)
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    DocumentProductSelectorInputV1,
    ProductCandidateViewV1,
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


class DocumentProductSelection(StrictModel):
    document_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=50)
    confidence: Literal["none", "low", "medium", "high"] = "none"
    rationale: str = Field(default="", max_length=600)


class DocumentProductRequest(Protocol):
    @property
    def message(self) -> str: ...


class DocumentProductSelector(Protocol):
    async def select(
        self,
        *,
        request: DocumentProductRequest,
        evidence_query: str,
        products: Sequence[ProductSource],
        max_documents: int,
    ) -> DocumentProductSelection: ...


class NoopDocumentProductSelector:
    async def select(
        self,
        *,
        request: DocumentProductRequest,
        evidence_query: str,
        products: Sequence[ProductSource],
        max_documents: int,
    ) -> DocumentProductSelection:
        del request, evidence_query, products, max_documents
        return DocumentProductSelection(confidence="none")


class DirectDocumentProductSelector:
    """Closed-set semantic selector for Document MCP product IDs."""

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
        request: DocumentProductRequest,
        evidence_query: str,
        products: Sequence[ProductSource],
        max_documents: int,
    ) -> DocumentProductSelection:
        if not self.settings.llm_api_key:
            return DocumentProductSelection(confidence="none")
        manifest = product_manifest(products)
        if not manifest:
            return DocumentProductSelection(confidence="none")
        selector_input = document_product_selector_input(
            request=request,
            evidence_query=evidence_query,
            candidates=manifest,
            max_documents=max_documents,
        )
        return await run_direct_document_product_selector(
            selector_input,
            self.settings,
            selector_cache=self.selector_cache,
        )


def document_product_selector_input(
    *,
    request: DocumentProductRequest,
    evidence_query: str,
    candidates: tuple[DocumentProductCandidate, ...],
    max_documents: int,
) -> DocumentProductSelectorInputV1:
    return DocumentProductSelectorInputV1(
        user_query=str(request.message or ""),
        evidence_query=evidence_query[:200],
        candidates=tuple(
            product_candidate_view(candidate) for candidate in candidates[:100]
        ),
        max_documents=max(1, min(max_documents, 50)),
    )


def document_product_selector_prompt() -> str:
    return assemble_canonicalization_prompt(
        "canonicalization.document_product_selector",
        stage="document_product_selector",
        selector_input_json="",
    )


async def run_direct_document_product_selector(
    selector_input: DocumentProductSelectorInputV1,
    settings: Settings,
    *,
    selector_cache: TurnSelectorOutcomeCacheV1 | None = None,
) -> DocumentProductSelection:
    selector_cache = resolve_turn_selector_outcome_cache(selector_cache)
    active_program = require_active_prompt_program_v2(
        program_id="document_product_selector.scene_neutral.v1@1",
        stage="document_product_selector",
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
            return DocumentProductSelection.model_validate(cached)
    synthesis = synthesize_direct_provider_messages(
        program=active_program,
        execution_spec=resolve_agent_execution_spec_v1(active_program.program_id),
        prompt_text=document_product_selector_prompt(),
        stage_input=selector_input,
        output_schema=ProviderJsonSchemaFormatV1(
            name="DocumentProductSelection",
            json_schema=DocumentProductSelection.model_json_schema(),
        ),
    )
    if settings.llm_api_key is None:
        return DocumentProductSelection(confidence="none")
    text = await run_direct_provider_text(
        synthesis=synthesis,
        target=target,
        api_key=settings.llm_api_key,
        response_model=DocumentProductSelection,
    )
    selection = DocumentProductSelection.model_validate_json(text)
    if selector_cache is not None:
        selector_cache.put(sih1, selection.model_dump(mode="json"))
    return selection


def product_candidate_view(
    candidate: DocumentProductCandidate,
) -> ProductCandidateViewV1:
    return ProductCandidateViewV1(
        id=candidate.id,
        name=candidate.name,
        title=candidate.title,
        category=candidate.category,
        keywords=candidate.keywords,
        summary=candidate.summary,
    )
