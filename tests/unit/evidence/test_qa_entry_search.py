from __future__ import annotations

from collections.abc import Sequence

import anyio
from pydantic import JsonValue

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.document_mcp.parsing import (
    DocumentMcpError,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_manifest_v2,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    MAX_QA_CORPUS_DOCUMENTS,
)
from market_support_crewai_agent.runtime.recall.document_qa_recall import (
    DocumentQaSearchService,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_envelope


def _policy(request: KernelReplyRequestV1) -> PolicyManifestV2:
    return compile_policy_manifest_v2(
        request,
        business_scope_authority_v1(request.business_scope),
        policy_ledger_summary_v1((), 0),
    )


class ErrorQaCorpusClient:
    async def fetch_all_documents_async(
        self,
        *,
        max_documents: int = MAX_QA_CORPUS_DOCUMENTS,
        evidence_query: str,
    ) -> Sequence[JsonValue]:
        del max_documents, evidence_query
        raise DocumentMcpError("boom")


class MalformedQaCorpusClient:
    async def fetch_all_documents_async(
        self,
        *,
        max_documents: int = MAX_QA_CORPUS_DOCUMENTS,
        evidence_query: str,
    ) -> Sequence[JsonValue]:
        del max_documents, evidence_query
        oversized_question = "坏" * 500
        document: JsonValue = {
            "id": "malformed",
            "title": "Broken",
            "content": f"Q：{oversized_question}\nA：不应该让 /reply 崩溃",
        }
        return (document,)


def test_document_qa_search_service_returns_unavailable_without_blocking_planner_fallback() -> (
    None
):
    request = make_v2_envelope("过拟合是什么意思").request
    service = DocumentQaSearchService(
        Settings(
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local",
        ),
        client=ErrorQaCorpusClient(),
    )

    match = anyio.run(service.collect, request, _policy(request))

    assert match.status == "unavailable"
    assert match.reason_code == "document_mcp_error"
    assert match.allow_planner_document_context_fallback is True
    assert match.candidates == []


def test_document_qa_search_service_handles_bounded_corpus_metadata() -> None:
    request = make_v2_envelope("坏数据").request
    service = DocumentQaSearchService(
        Settings(
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local",
        ),
        client=MalformedQaCorpusClient(),
    )

    match = anyio.run(service.collect, request, _policy(request))

    assert match.status == "matched"
    assert match.allow_planner_document_context_fallback is True
    assert match.candidates[0].doc_id == "malformed"
