"""Answer-bearing evidence (selected knowledge documents) must reach the
composer whole, not as a 1200-char preview. These guard the per-fact inline
budget in ContextProjectionManager so the Document MCP per-document cap is not
silently re-truncated downstream."""

from __future__ import annotations

from market_support_crewai_agent.runtime.context.models import ContextProjectionPolicy
from market_support_crewai_agent.runtime.context.projection import ContextProjectionManager
from market_support_crewai_agent.runtime.evidence import EvidenceFact


def _fact(fact_type: str, value: str) -> EvidenceFact:
    return EvidenceFact(
        fact_type=fact_type,  # type: ignore[arg-type]
        value=value,
        source_type="document_mcp",
        source_id="faq",
    )


def _manager() -> ContextProjectionManager:
    return ContextProjectionManager(policy=ContextProjectionPolicy())


def test_document_context_value_is_inlined_in_full_above_compact_limit():
    manager = _manager()
    big = "净" * 14000  # > compact limit, < answer budget
    fact = _fact("document_context", big)

    output, previews = manager._replace_large_evidence(
        {"value": big, "metadata": {}}, fact, "eid"
    )

    assert output["value"] == big
    assert previews == []


def test_non_answer_evidence_value_is_previewed_above_compact_limit():
    manager = _manager()
    big = "净" * 14000
    fact = _fact("report_scope_products", big)

    output, previews = manager._replace_large_evidence(
        {"value": big, "metadata": {}}, fact, "eid"
    )

    assert output["value"] != big
    assert isinstance(output["value"], dict)  # replaced by a preview handle
    assert previews


def test_answer_evidence_is_still_bounded_above_the_answer_budget():
    manager = ContextProjectionManager(
        policy=ContextProjectionPolicy(max_answer_evidence_chars_inline=16000)
    )
    huge = "净" * 17000  # > custom answer budget -> still previewed
    fact = _fact("document_context", huge)

    output, previews = manager._replace_large_evidence(
        {"value": huge, "metadata": {}}, fact, "eid"
    )

    assert output["value"] != huge
    assert previews


def test_inline_value_limit_tracks_policy():
    manager = _manager()
    assert manager._inline_value_limit(_fact("document_context", "x")) == 1_000_000
    assert manager._inline_value_limit(_fact("report_scope_summary", "x")) == 6000
