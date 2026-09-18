from __future__ import annotations

from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.parsing import (
    DocumentEvidenceChunk,
)


def document_context_from_chunk(
    chunk: DocumentEvidenceChunk,
) -> GatewayDocumentContextV1:
    return GatewayDocumentContextV1(
        document_id=chunk.document_id,
        title=chunk.title,
        text=chunk.text,
    )
