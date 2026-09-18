from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request

import anyio
from anyio.to_thread import run_sync
from pydantic import JsonValue, ValidationError

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.document_mcp import (
    manifest as document_manifest,
)
from market_support_crewai_agent.runtime.integrations.document_mcp import (
    parsing as document_parsing,
)
from market_support_crewai_agent.runtime.integrations.document_mcp import (
    selection as document_selection,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
    document_cache_get,
    document_cache_set,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    ProviderTargetError,
)
from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings

_DEFAULT_MAX_CHARS_PER_DOCUMENT = 1_000_000


class DocumentContextClient(Protocol):
    async def fetch_context_async(
        self,
        request: KernelReplyRequestV1,
        *,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> Sequence[document_parsing.DocumentEvidenceChunk]: ...


class DocumentMcpClient:
    def __init__(
        self,
        settings: Settings | None = None,
        product_selector: document_selection.DocumentProductSelector | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.base_url: str = (self.settings.doc_mcp_base_url or "").rstrip("/")
        self.timeout: float = self.settings.doc_mcp_timeout_seconds
        self.max_chars_per_document: int = self.settings.doc_mcp_max_chars_per_document
        self.cache_ttl_seconds: float = self.settings.doc_mcp_cache_ttl_seconds
        self.baseline_categories: tuple[str, ...] = (
            self.settings.doc_mcp_baseline_categories
        )
        self.product_selector: document_selection.DocumentProductSelector = (
            product_selector
            or document_selection.DirectDocumentProductSelector(self.settings)
        )

    def fetch_context(
        self,
        request: KernelReplyRequestV1,
        *,
        evidence_query: str | None = None,
        max_documents: int = 50,
        max_chars_per_document: int | None = None,
        cache_authority: DocumentMcpCacheAuthorityV1 | None = None,
    ) -> list[document_parsing.DocumentEvidenceChunk]:
        try:
            _ = anyio.get_current_task()
        except anyio.NoEventLoopError:
            return anyio.run(
                lambda: self.fetch_context_async(
                    request,
                    evidence_query=evidence_query,
                    max_documents=max_documents,
                    max_chars_per_document=max_chars_per_document,
                    cache_authority=cache_authority,
                )
            )
        raise document_parsing.DocumentMcpError(
            "synchronous document MCP fetch cannot run inside an active event loop"
        )

    async def fetch_context_async(
        self,
        request: KernelReplyRequestV1,
        *,
        evidence_query: str | None = None,
        max_documents: int = 50,
        max_chars_per_document: int | None = None,
        cache_authority: DocumentMcpCacheAuthorityV1 | None = None,
    ) -> list[document_parsing.DocumentEvidenceChunk]:
        if not self.base_url:
            return []

        effective_max_chars = (
            max_chars_per_document
            if max_chars_per_document is not None
            else self.max_chars_per_document or _DEFAULT_MAX_CHARS_PER_DOCUMENT
        )
        query_text = document_parsing.retrieval_query(request, evidence_query)
        products = await run_sync(self._list_products)
        try:
            selection = await self.product_selector.select(
                request=request,
                evidence_query=query_text,
                products=products,
                max_documents=max_documents,
            )
        except (
            ProviderInvocationError,
            ProviderTargetError,
            RuntimeError,
            ValidationError,
        ) as exc:
            raise document_parsing.DocumentMcpError(
                "document MCP product selector failed"
            ) from exc
        document_ids = document_manifest.validated_selected_document_ids(
            selection,
            products,
            max_documents=max_documents,
        )
        fallback_ids = document_manifest.fallback_document_ids(
            products,
            self.baseline_categories,
            max_documents=max_documents,
        )
        document_ids = document_manifest.selected_with_fallback_ids(
            document_ids,
            fallback_ids,
            max_documents=max_documents,
        )
        if not document_ids:
            return []

        documents = await run_sync(
            partial(
                self._get_documents,
                document_ids,
                query=query_text,
                max_results=max_documents,
                cache_authority=cache_authority,
            )
        )
        return document_parsing.document_chunks_from_payloads(
            documents,
            query_text=query_text,
            max_chars=effective_max_chars,
        )

    async def fetch_all_documents_async(
        self,
        *,
        max_documents: int = 50,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None = None,
    ) -> list[document_parsing.JsonMap]:
        if not self.base_url:
            return []
        products = await run_sync(self._list_products)
        document_ids = [
            str(product.get("id") or "").strip()
            for product in products
            if str(product.get("id") or "").strip()
        ][: max(1, max_documents)]
        if not document_ids:
            return []
        return await run_sync(
            partial(
                self._get_documents,
                document_ids,
                query=evidence_query,
                max_results=max_documents,
                cache_authority=cache_authority,
            )
        )

    def _list_products(self) -> list[document_parsing.JsonMap]:
        result = self._call_tool("list_products", {})
        payload = document_parsing.tool_text_json(result)
        return document_parsing.json_maps_from_value(payload.get("products"))

    def _get_documents(
        self,
        document_ids: list[str],
        *,
        query: str,
        max_results: int,
        cache_authority: DocumentMcpCacheAuthorityV1 | None = None,
    ) -> list[document_parsing.JsonMap]:
        ordered_ids = [document_id for document_id in document_ids if document_id]
        if not ordered_ids:
            return []

        cache_enabled = cache_authority is not None and self.cache_ttl_seconds > 0
        cache_key_ref = (
            cache_authority.key_for_document_query(
                query=query,
                document_ids=tuple(ordered_ids),
                max_results=max_results,
            )
            if cache_enabled and cache_authority is not None
            else None
        )
        if cache_key_ref is not None:
            cached_documents = document_parsing.json_maps_from_value(
                document_cache_get(cache_key_ref)
            )
            if cached_documents:
                return [dict(item) for item in cached_documents]

        document_id_values: list[JsonValue] = list(ordered_ids)
        result = self._call_tool("get_documents", {"documentIds": document_id_values})
        payload = document_parsing.tool_text_json(result)
        documents = document_parsing.json_maps_from_value(payload.get("documents"))
        resolved = {
            document_id: document
            for document in documents
            if (document_id := str(document.get("id") or ""))
        }
        output = [
            resolved[document_id]
            for document_id in ordered_ids
            if document_id in resolved
        ]
        if cache_key_ref is not None:
            document_cache_set(cache_key_ref, output, self.cache_ttl_seconds)
        return output

    def _call_tool(
        self,
        name: str,
        arguments: document_parsing.JsonMap,
    ) -> document_parsing.JsonMap:
        return self._post_json_rpc("tools/call", {"name": name, "arguments": arguments})

    def _post_json_rpc(
        self,
        method: str,
        params: document_parsing.JsonMap,
    ) -> document_parsing.JsonMap:
        http_request = Request(
            f"{self.base_url}/mcp",
            data=document_parsing.document_mcp_request_body(method, params),
            headers={
                "Content-Type": "application/json",
                "Accept": document_parsing.MCP_ACCEPT_HEADER,
            },
            method="POST",
        )
        try:
            raw = document_parsing.read_document_mcp_response(
                http_request,
                timeout=self.timeout,
            )
        except (HTTPError, URLError) as exc:
            raise document_parsing.document_mcp_transport_error(exc) from exc
        return document_parsing.parse_mcp_result(raw)
