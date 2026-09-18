from __future__ import annotations

import json

import anyio
from pydantic import JsonValue
from typing_extensions import override

from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DOCUMENT_CACHE,
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.client import (
    DocumentMcpClient,
)
from market_support_crewai_agent.settings_model import Settings

JsonMap = dict[str, JsonValue]


def _text_tool_result(payload: JsonMap) -> JsonMap:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    }


class CountingMcpClient(DocumentMcpClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.tool_calls: list[tuple[str, JsonMap]] = []
        self.products_payload: JsonMap = {"products": [{"id": "company"}]}
        self.documents_payload: JsonMap = {
            "documents": [{"id": "company", "title": "Company", "content": "公司规模"}]
        }

    @override
    def _call_tool(self, name: str, arguments: JsonMap) -> JsonMap:
        self.tool_calls.append((name, dict(arguments)))
        if name == "list_products":
            return _text_tool_result(self.products_payload)
        return _text_tool_result(self.documents_payload)


def _cache_config() -> DocumentMcpCacheConfigV1:
    return DocumentMcpCacheConfigV1(
        client_contract_version="document-mcp-client.v1",
        corpus_version="company-public.v1",
        request_schema_hash="osh1:" + "1" * 64,
        response_schema_hash="osh1:" + "2" * 64,
        timeout_milliseconds=1_000,
        max_candidates=8,
        ttl_seconds=300,
        capacity=8,
    )


def _cache_authority(state_digit: str) -> DocumentMcpCacheAuthorityV1:
    return DocumentMcpCacheAuthorityV1(
        state_key_ref="csk1:" + state_digit * 64,
        policy_id="pol1:" + "a" * 64,
        manifest_ref="answer_internal_company_knowledge@2026-07-18.1",
        business_scope_hash="bsh1:" + "b" * 64,
        source_cache_config=_cache_config(),
    )


def test_fetch_all_documents_cache_isolated_by_final_cache_authority() -> None:
    DOCUMENT_CACHE.clear()
    first_client = CountingMcpClient(
        Settings(
            doc_mcp_base_url="http://cache-docs-scoped:23000",
            doc_mcp_cache_ttl_seconds=300,
        )
    )
    first_client.products_payload = {"products": [{"id": "a"}]}
    first_client.documents_payload = {
        "documents": [{"id": "a", "title": "A", "content": "tenant-a"}]
    }
    second_client = CountingMcpClient(
        Settings(
            doc_mcp_base_url="http://cache-docs-scoped:23000",
            doc_mcp_cache_ttl_seconds=300,
        )
    )
    second_client.products_payload = {"products": [{"id": "a"}]}
    second_client.documents_payload = {
        "documents": [{"id": "a", "title": "A", "content": "tenant-b"}]
    }

    first = anyio.run(
        lambda: first_client.fetch_all_documents_async(
            max_documents=1,
            evidence_query="company",
            cache_authority=_cache_authority("1"),
        )
    )
    second = anyio.run(
        lambda: second_client.fetch_all_documents_async(
            max_documents=1,
            evidence_query="company",
            cache_authority=_cache_authority("2"),
        )
    )

    assert first[0]["content"] == "tenant-a"
    assert second[0]["content"] == "tenant-b"
    assert first_client.tool_calls == [
        ("list_products", {}),
        ("get_documents", {"documentIds": ["a"]}),
    ]
    assert second_client.tool_calls == [
        ("list_products", {}),
        ("get_documents", {"documentIds": ["a"]}),
    ]


def test_fetch_all_documents_inspects_every_manifest_document_for_qa_corpus() -> None:
    DOCUMENT_CACHE.clear()
    client = CountingMcpClient(
        Settings(
            doc_mcp_base_url="http://qa-corpus-docs:23000",
            doc_mcp_cache_ttl_seconds=300,
        )
    )
    client.products_payload = {
        "products": [
            {"id": "company"},
            {"id": "faq"},
            {"id": ""},
            {"id": "strategy"},
        ]
    }
    client.documents_payload = {
        "documents": [
            {"id": "company", "title": "Company", "content": "公司规模"},
            {"id": "faq", "title": "FAQ", "content": "Q：过拟合\nA：样本外失效"},
            {"id": "strategy", "title": "Strategy", "content": "策略说明"},
        ]
    }

    documents = anyio.run(
        lambda: client.fetch_all_documents_async(
            max_documents=10,
            evidence_query="qa corpus",
            cache_authority=_cache_authority("3"),
        )
    )

    assert [document["id"] for document in documents] == ["company", "faq", "strategy"]
    assert client.tool_calls == [
        ("list_products", {}),
        ("get_documents", {"documentIds": ["company", "faq", "strategy"]}),
    ]
