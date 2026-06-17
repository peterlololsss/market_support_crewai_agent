"""Source metadata and precedence helpers for evidence-grounded claims."""

from market_support_crewai_agent.runtime.domain.sources.metadata import (
    SourceContextType,
    SourceMetadata,
    is_history_source,
    source_metadata_for_conversation_message,
    source_metadata_for_evidence,
    source_metadata_from_mapping,
    source_metadata_prompt_dict,
)
__all__ = [
    "SourceContextType",
    "SourceMetadata",
    "is_history_source",
    "source_metadata_for_conversation_message",
    "source_metadata_for_evidence",
    "source_metadata_from_mapping",
    "source_metadata_prompt_dict",
]
