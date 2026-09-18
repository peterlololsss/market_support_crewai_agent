from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import JsonValue, ValidationError

from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    kernel_channel_type,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.client import (
    DocumentMcpClient,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.parsing import (
    DocumentMcpError,
)
from market_support_crewai_agent.runtime.policy.capabilities.runtime_projection import (
    read_capabilities_for_artifact,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    MAX_QA_CORPUS_DOCUMENTS,
    DocumentQaEntry,
    KnowledgeQaMatch,
    QaCorpusShapeError,
    parse_qa_entries,
    search_qa_entries,
)
from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings

_DOC_CAPABILITY = next(iter(read_capabilities_for_artifact("knowledge_answer")), "")


class QaCorpusClient(Protocol):
    async def fetch_all_documents_async(
        self,
        *,
        max_documents: int = MAX_QA_CORPUS_DOCUMENTS,
        evidence_query: str,
    ) -> Sequence[JsonValue]: ...


class DocumentQaSearchService:
    def __init__(
        self,
        settings: Settings | None = None,
        client: QaCorpusClient | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.client: QaCorpusClient = client or DocumentMcpClient(self.settings)

    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> KnowledgeQaMatch:
        if not self.settings.doc_mcp_enabled or not self.settings.doc_mcp_base_url:
            return KnowledgeQaMatch(
                status="disabled", reason_code="document_mcp_disabled"
            )
        if (
            kernel_channel_type(request)
            not in self.settings.doc_mcp_allowed_channel_types
        ):
            return KnowledgeQaMatch(
                status="disabled", reason_code="document_mcp_channel_forbidden"
            )
        if _DOC_CAPABILITY not in policy.allowed_read_capabilities:
            return KnowledgeQaMatch(
                status="disabled", reason_code="document_mcp_policy_forbidden"
            )

        try:
            entries = await self._entries_for_scope(request, policy)
        except DocumentMcpError:
            return KnowledgeQaMatch(
                status="unavailable", reason_code="document_mcp_error"
            )
        except ValidationError:
            return KnowledgeQaMatch(
                status="unavailable", reason_code="qa_entry_parse_error"
            )
        except QaCorpusShapeError:
            return KnowledgeQaMatch(
                status="unavailable", reason_code="qa_entry_corpus_invalid"
            )
        if not entries:
            return KnowledgeQaMatch(
                status="no_match", reason_code="qa_entry_corpus_empty"
            )
        return search_qa_entries(request.message, entries)

    async def _entries_for_scope(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> tuple[DocumentQaEntry, ...]:
        del policy
        documents = await self.client.fetch_all_documents_async(
            max_documents=MAX_QA_CORPUS_DOCUMENTS,
            evidence_query=request.message,
        )
        entries = parse_qa_entries(
            documents,
            max_chars_per_document=self.settings.doc_mcp_max_chars_per_document,
        )
        return entries
