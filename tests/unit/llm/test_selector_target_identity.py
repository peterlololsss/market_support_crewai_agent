from __future__ import annotations

from dataclasses import dataclass

import anyio
import pytest
from pydantic import BaseModel

from market_support_crewai_agent.runtime.integrations.document_mcp import selection
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnSelectorOutcomeCacheV1,
    use_turn_selector_outcome_cache,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    DirectProviderSynthesisV1,
    ProviderTargetConfigV1,
)
from market_support_crewai_agent.runtime.prompts.provider_transport import (
    build_provider_transport_envelope,
)
from market_support_crewai_agent.settings_model import Settings


@dataclass(frozen=True, slots=True)
class SelectorRequest:
    message: str


def test_selector_slash_aliases_reuse_one_cache_entry_and_dispatch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one selector cache and a provider fake below the real selector path.
    dispatches: list[tuple[ProviderTargetConfigV1, str]] = []

    async def fake_provider_call(
        *,
        synthesis: DirectProviderSynthesisV1,
        target: ProviderTargetConfigV1,
        api_key: str,
        response_model: type[BaseModel],
    ) -> str:
        del api_key, response_model
        envelope = build_provider_transport_envelope(synthesis, target)
        dispatches.append((target, envelope.variant))
        return '{"document_ids":[],"confidence":"none","rationale":""}'

    monkeypatch.setattr(selection, "run_direct_provider_text", fake_provider_call)
    selector = selection.DirectDocumentProductSelector(
        Settings(
            llm_api_key="key",
            llm_base_url="https://EXAMPLE.com/v1",
        )
    )
    cache = TurnSelectorOutcomeCacheV1()
    request = SelectorRequest(message="Alpha")
    products = [{"id": "document:alpha", "title": "Alpha"}]

    async def run_aliases() -> None:
        with use_turn_selector_outcome_cache(cache):
            _ = await selector.select(
                request=request,
                evidence_query="Alpha",
                products=products,
                max_documents=1,
            )
            selector.settings = Settings(
                llm_api_key="key",
                llm_base_url="https://example.com/v1/",
            )
            _ = await selector.select(
                request=request,
                evidence_query="Alpha",
                products=products,
                max_documents=1,
            )

    # When: identical strict input is invoked through slash-equivalent targets.
    anyio.run(run_aliases)

    # Then: canonical target identity yields one provider call and one cache row.
    assert len(dispatches) == 1
    assert len(cache.keys) == 1
    dispatched_target, envelope_variant = dispatches[0]
    assert dispatched_target.normalized_endpoint == "https://example.com/v1"
    assert dispatched_target.transport_variant == "openai_chat_completions"
    assert envelope_variant == "openai_chat_completions"
