from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import Field

from market_support_crewai_agent.runtime.domain.capabilities import read_capabilities_for_artifact
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.ontology import artifact_scope_for_evidence
from market_support_crewai_agent.runtime.domain.sources.metadata import (
    source_metadata_from_mapping,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.schemas import ReplyRequest, StrictModel
from market_support_crewai_agent.runtime.llm.prompting.assembler import (
    assembleCanonicalizationPrompt,
)
from market_support_crewai_agent.runtime.llm.prompting.registry import prompt_agent_spec_by_id
from market_support_crewai_agent.runtime.state.runtime_trace import trace_span
from market_support_crewai_agent.settings import Settings, get_settings

_DOC_CAPABILITY = next(iter(read_capabilities_for_artifact("knowledge_answer")), "")
_MCP_ACCEPT_HEADER = "application/json, text/event-stream"
# Fallback ceiling for helpers invoked without an explicit budget. Production
# callers thread Settings.doc_mcp_max_chars_per_document through instead.
_DEFAULT_MAX_CHARS_PER_DOCUMENT = 1_000_000
_LOCAL_LOCATOR_RE = re.compile(
    r"(?i)(file://\S+|/[Uu]sers/\S+|/[Hh]ome/\S+|[A-Za-z]:\\[^\s]+|"
    r"wecom-adapter:[^\s]+|mcp://\S+)"
)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|bearer|token|password|secret)\b\s*[:=]\s*\S+"
)
_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"(?i)\bignore (all )?(previous|above) instructions\b"),
    re.compile(r"(?i)\b(system|developer) (prompt|message|instruction)s?\b"),
    re.compile(r"(?i)\b(call|use|execute) (the )?(tool|function)\b"),
    re.compile(r"(?i)\byou are now\b"),
    re.compile(r"(?i)\bdisregard (all )?(previous|above) instructions\b"),
    re.compile(r"忽略(以上|之前|所有).*指令"),
    re.compile(r"不要遵守(以上|之前|所有).*指令"),
    re.compile(r"(调用|执行).*工具"),
    re.compile(r"你现在是"),
)
_REDACTED_LOCATOR = "[REDACTED_INTERNAL_LOCATOR]"
_REDACTED_SECRET = "[REDACTED_SECRET]"
_REMOVED_DOCUMENT_INSTRUCTION = "[REMOVED_DOCUMENT_INSTRUCTION]"


class _TtlCache:
    """Process-wide, thread-safe TTL cache for static Document MCP payloads.

    The client is rebuilt per request, so the cache lives at module scope. The
    MCP serves a small, static corpus; caching the manifest and document content
    avoids a network round-trip on every knowledge query without coupling the
    cache lifetime to any one client instance.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str) -> object | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: object, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._store[key] = (time.monotonic() + ttl_seconds, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_DOCUMENT_CACHE = _TtlCache()


@dataclass(frozen=True)
class DocumentEvidenceChunk:
    document_id: str
    title: str
    text: str
    metadata: dict[str, object] | None = None


class DocumentMcpError(RuntimeError):
    """Raised when the document MCP cannot provide bounded evidence."""


@dataclass(frozen=True)
class SanitizedDocumentText:
    text: str
    metadata: dict[str, object]


class DocumentProductCandidate(StrictModel):
    id: str = Field(min_length=1, max_length=160)
    name: str = Field(default="", max_length=160)
    title: str = Field(default="", max_length=240)
    category: str = Field(default="", max_length=80)
    keywords: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    summary: str = Field(default="", max_length=1200)


class DocumentProductSelection(StrictModel):
    document_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=50)
    confidence: Literal["none", "low", "medium", "high"] = "none"
    rationale: str = Field(default="", max_length=600)


class DocumentProductSelector(Protocol):
    async def select(
        self,
        *,
        request: ReplyRequest,
        evidence_query: str,
        canonical_context: CanonicalContext,
        products: list[dict],
        max_documents: int,
    ) -> DocumentProductSelection: ...


class NoopDocumentProductSelector:
    async def select(
        self,
        *,
        request: ReplyRequest,
        evidence_query: str,
        canonical_context: CanonicalContext,
        products: list[dict],
        max_documents: int,
    ) -> DocumentProductSelection:
        del request, evidence_query, canonical_context, products, max_documents
        return DocumentProductSelection(confidence="none")


class CrewAIDocumentProductSelector:
    """Closed-set semantic selector for Document MCP product IDs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def select(
        self,
        *,
        request: ReplyRequest,
        evidence_query: str,
        canonical_context: CanonicalContext,
        products: list[dict],
        max_documents: int,
    ) -> DocumentProductSelection:
        if not self.settings.llm_api_key:
            return DocumentProductSelection(confidence="none")
        manifest = _product_manifest(products)
        if not manifest:
            return DocumentProductSelection(confidence="none")
        prompt = _document_product_selector_prompt(
            request=request,
            evidence_query=evidence_query,
            canonical_context=canonical_context,
            candidates=manifest,
            max_documents=max_documents,
        )
        return await _run_crewai_document_product_selector(
            prompt,
            self.settings,
            timeout_seconds=self.settings.llm_timeout_seconds,
        )


class DocumentMcpClient:
    def __init__(
        self,
        settings: Settings | None = None,
        product_selector: DocumentProductSelector | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = (self.settings.doc_mcp_base_url or "").rstrip("/")
        self.timeout = self.settings.doc_mcp_timeout_seconds
        self.max_chars_per_document = self.settings.doc_mcp_max_chars_per_document
        self.cache_ttl_seconds = self.settings.doc_mcp_cache_ttl_seconds
        self.baseline_categories = self.settings.doc_mcp_baseline_categories
        self.product_selector = product_selector or CrewAIDocumentProductSelector(
            self.settings
        )

    def fetch_context(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        *,
        evidence_query: str | None = None,
        max_documents: int = 50,
        max_chars_per_document: int | None = None,
    ) -> list[DocumentEvidenceChunk]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.fetch_context_async(
                    request,
                    canonical_context,
                    evidence_query=evidence_query,
                    max_documents=max_documents,
                    max_chars_per_document=max_chars_per_document,
                )
            )
        raise DocumentMcpError(
            "synchronous document MCP fetch cannot run inside an active event loop"
        )

    async def fetch_context_async(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        *,
        evidence_query: str | None = None,
        max_documents: int = 50,
        max_chars_per_document: int | None = None,
    ) -> list[DocumentEvidenceChunk]:
        if not self.base_url:
            return []

        effective_max_chars = (
            max_chars_per_document
            if max_chars_per_document is not None
            else self.max_chars_per_document or _DEFAULT_MAX_CHARS_PER_DOCUMENT
        )
        query_text = _retrieval_query(request, evidence_query)
        products = await asyncio.to_thread(self._list_products)
        try:
            selection = await self.product_selector.select(
                request=request,
                evidence_query=query_text,
                canonical_context=canonical_context,
                products=products,
                max_documents=max_documents,
            )
        except Exception as exc:
            raise DocumentMcpError("document MCP product selector failed") from exc
        document_ids = _validated_selected_document_ids(
            selection,
            products,
            max_documents=max_documents,
        )
        if not document_ids:
            # When selection is unsure, fall back to bounded baseline/broad evidence.
            document_ids = _fallback_document_ids(
                products,
                self.baseline_categories,
                max_documents=max_documents,
            )
        if not document_ids:
            return []

        documents = await asyncio.to_thread(self._get_documents, document_ids)
        chunks: list[DocumentEvidenceChunk] = []
        for document in documents:
            document_id = str(document.get("id") or "")
            title = str(document.get("title") or document.get("name") or document_id)
            content = str(document.get("content") or "")
            if not document_id or not content:
                continue
            chunks.append(
                DocumentEvidenceChunk(
                    document_id=document_id,
                    title=title,
                    text=_select_document_text(
                        content,
                        query_text,
                        canonical_context,
                        max_chars=effective_max_chars,
                    ),
                    metadata=_document_artifact_metadata(document),
                )
            )
        return chunks

    def _list_products(self) -> list[dict]:
        cache_key = f"{self.base_url}|list_products"
        cached = _DOCUMENT_CACHE.get(cache_key)
        if isinstance(cached, list):
            return list(cached)
        result = self._call_tool("list_products", {})
        payload = _tool_text_json(result)
        products = payload.get("products")
        products = products if isinstance(products, list) else []
        _DOCUMENT_CACHE.set(cache_key, products, self.cache_ttl_seconds)
        return list(products)

    def _get_documents(self, document_ids: list[str]) -> list[dict]:
        ordered_ids = [document_id for document_id in document_ids if document_id]
        resolved: dict[str, dict] = {}
        missing: list[str] = []
        for document_id in ordered_ids:
            if document_id in resolved or document_id in missing:
                continue
            cached = _DOCUMENT_CACHE.get(f"{self.base_url}|document|{document_id}")
            if isinstance(cached, dict):
                resolved[document_id] = cached
            else:
                missing.append(document_id)
        if missing:
            result = self._call_tool("get_documents", {"documentIds": missing})
            payload = _tool_text_json(result)
            documents = payload.get("documents")
            for document in documents if isinstance(documents, list) else []:
                if not isinstance(document, dict):
                    continue
                document_id = str(document.get("id") or "")
                if not document_id:
                    continue
                _DOCUMENT_CACHE.set(
                    f"{self.base_url}|document|{document_id}",
                    document,
                    self.cache_ttl_seconds,
                )
                resolved.setdefault(document_id, document)
        return [resolved[document_id] for document_id in ordered_ids if document_id in resolved]

    def _call_tool(self, name: str, arguments: dict) -> dict:
        return self._post_json_rpc(
            "tools/call",
            {"name": name, "arguments": arguments},
        )

    def _post_json_rpc(self, method: str, params: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": method,
            "method": method,
            "params": params,
        }
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        http_request = Request(
            f"{self.base_url}/mcp",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": _MCP_ACCEPT_HEADER,
            },
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DocumentMcpError(
                f"document MCP returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise DocumentMcpError(f"document MCP request failed: {exc}") from exc

        message = _parse_mcp_message(raw)
        if "error" in message:
            raise DocumentMcpError(f"document MCP returned error: {message['error']}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise DocumentMcpError("document MCP returned an invalid result")
        return result


class DocumentMcpEvidenceService:
    def __init__(
        self,
        settings: Settings | None = None,
        client: DocumentMcpClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or DocumentMcpClient(self.settings)

    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        policy: PolicyManifest,
    ) -> list[EvidenceFact]:
        if not self.settings.doc_mcp_enabled or not self.settings.doc_mcp_base_url:
            return []
        if plan.compliance.is_compliant is not True:
            return []
        if "document_context" not in plan.capabilities:
            return []
        if request.channel_type not in self.settings.doc_mcp_allowed_channel_types:
            return [
                _document_context_unavailable_fact(
                    reason_code="document_mcp_channel_forbidden",
                )
            ]
        if _DOC_CAPABILITY not in policy.allowed_read_capabilities:
            return [
                _document_context_unavailable_fact(
                    reason_code="document_mcp_policy_forbidden",
                )
            ]

        try:
            chunks = await self.client.fetch_context_async(
                request,
                canonical_context,
                evidence_query=plan.evidence_query,
            )
        except DocumentMcpError as exc:
            return [
                _document_context_unavailable_fact(
                    reason_code="document_mcp_error",
                    error_type=type(exc).__name__,
                )
            ]
        if not chunks:
            return [
                _document_context_unavailable_fact(
                    reason_code="document_context_not_found",
                )
            ]

        facts: list[EvidenceFact] = []
        for chunk in chunks:
            sanitized = _sanitize_document_text(chunk.text)
            if not sanitized.text.strip():
                continue
            evidence_text, metadata = _bound_sanitized_document_text(
                sanitized,
                max_chars=self.settings.doc_mcp_max_chars_per_document,
            )
            facts.append(
                EvidenceFact(
                    fact_type="document_context",
                    value=evidence_text,
                    source_type="document_mcp",
                    source_id=chunk.document_id,
                    metadata={
                        "title": chunk.title,
                        "content_is_data_only": True,
                        "original_artifact_metadata": dict(chunk.metadata or {}),
                        **metadata,
                    },
                    artifact_type="document_context",
                    scope=artifact_scope_for_evidence(
                        channel_id="unknown",
                        artifact_type="document_context",
                        source_id=chunk.document_id,
                        source_type="document_mcp",
                        metadata={"title": chunk.title, **dict(chunk.metadata or {})},
                    ),
                    source_metadata=source_metadata_from_mapping(
                        {
                            **dict(chunk.metadata or {}),
                            "source_id": chunk.document_id,
                            "source_type": "retrieved_doc",
                            "artifact_type": "document_context",
                            "provenance": (
                                (chunk.metadata or {}).get("provenance")
                                or "document_mcp"
                            ),
                            "evidence_allowed_by_default": True,
                        }
                    ),
                )
            )
        if not facts:
            return [
                _document_context_unavailable_fact(
                    reason_code="document_context_empty_after_sanitization",
                )
            ]
        return facts


class NoopDocumentMcpEvidenceService:
    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        policy: PolicyManifest,
    ) -> list[EvidenceFact]:
        del request, canonical_context, plan, policy
        return []


def _parse_mcp_message(raw: str) -> dict:
    stripped = raw.strip()
    if not stripped:
        raise DocumentMcpError("document MCP returned an empty response")
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DocumentMcpError("document MCP returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise DocumentMcpError("document MCP returned a non-object JSON response")
        return payload

    data_lines = [
        line.removeprefix("data:").strip()
        for line in stripped.splitlines()
        if line.startswith("data:")
    ]
    if not data_lines:
        raise DocumentMcpError("document MCP SSE response has no data line")
    try:
        payload = json.loads(data_lines[-1])
    except json.JSONDecodeError as exc:
        raise DocumentMcpError("document MCP SSE data is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise DocumentMcpError("document MCP SSE data is not a JSON object")
    return payload


def _tool_text_json(result: dict) -> dict:
    content = result.get("content")
    if not isinstance(content, list):
        return {}
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            payload = json.loads(str(item.get("text") or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _document_artifact_metadata(document: dict) -> dict[str, object]:
    metadata = document.get("metadata")
    output = dict(metadata) if isinstance(metadata, dict) else {}
    for key in (
        "artifact_type",
        "channel_id",
        "strategy_id",
        "product_ids",
        "time_range",
        "created_at",
        "observed_at",
        "provenance",
    ):
        if key not in output and key in document:
            output[key] = document[key]
    return output




def _product_manifest(products: list[dict]) -> tuple[DocumentProductCandidate, ...]:
    candidates: list[DocumentProductCandidate] = []
    for product in products:
        if not isinstance(product, dict):
            continue
        product_id = str(product.get("id") or "").strip()
        if not product_id:
            continue
        keywords = tuple(
            keyword
            for keyword in (
                str(item or "").strip() for item in product.get("keywords") or ()
            )
            if keyword
        )
        candidates.append(
            DocumentProductCandidate(
                id=product_id,
                name=str(product.get("name") or "").strip(),
                title=str(product.get("title") or "").strip(),
                category=str(product.get("category") or "").strip(),
                keywords=keywords,
                summary=str(product.get("summary") or "").strip(),
            )
        )
    return tuple(candidates)


def _validated_selected_document_ids(
    selection: DocumentProductSelection,
    products: list[dict],
    *,
    max_documents: int,
) -> list[str]:
    if selection.confidence == "none":
        return []
    valid_ids = {
        str(product.get("id") or "").strip()
        for product in products
        if isinstance(product, dict)
    }
    output: list[str] = []
    seen: set[str] = set()
    for document_id in selection.document_ids:
        normalized_id = str(document_id or "").strip()
        if not normalized_id or normalized_id in seen or normalized_id not in valid_ids:
            continue
        seen.add(normalized_id)
        output.append(normalized_id)
        if len(output) >= max_documents:
            break
    return output


def _fallback_document_ids(
    products: list[dict],
    baseline_categories: tuple[str, ...],
    *,
    max_documents: int,
) -> list[str]:
    wanted = {category.strip() for category in baseline_categories if category.strip()}
    output: list[str] = []
    for product in products:
        if not isinstance(product, dict):
            continue
        product_id = str(product.get("id") or "").strip()
        category = str(product.get("category") or "").strip()
        if not product_id or category not in wanted or product_id in output:
            continue
        output.append(product_id)
        if len(output) >= max_documents:
            return output
    for product in products:
        if not isinstance(product, dict):
            continue
        product_id = str(product.get("id") or "").strip()
        if not product_id or product_id in output:
            continue
        output.append(product_id)
        if len(output) >= max_documents:
            return output
    return output


def _document_product_selector_prompt(
    *,
    request: ReplyRequest,
    evidence_query: str,
    canonical_context: CanonicalContext,
    candidates: tuple[DocumentProductCandidate, ...],
    max_documents: int,
) -> str:
    payload = {
        "user_message": str(request.message or "").strip(),
        "evidence_query": evidence_query,
        "canonical_context": canonical_context.to_prompt_dict(),
        "max_documents": max_documents,
        "candidate_documents": [
            candidate.model_dump(mode="json", exclude_none=True)
            for candidate in candidates
        ],
    }
    return assembleCanonicalizationPrompt(
        "canonicalization.document_product_selector",
        stage="document_product_selector",
        selector_input_json=json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
    )


async def _run_crewai_document_product_selector(
    prompt: str,
    settings: Settings,
    *,
    timeout_seconds: float,
) -> DocumentProductSelection:
    from crewai import Agent, LLM

    spec = prompt_agent_spec_by_id("agent.document_product_selector")
    agent = Agent(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory,
        llm=LLM(
            model=settings.llm_model,
            provider=settings.llm_provider,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0,
            max_tokens=min(settings.llm_max_tokens, 1200),
            timeout=settings.llm_timeout_seconds,
        ),
        allow_delegation=False,
        verbose=settings.crewai_verbose,
        max_iter=1,
        max_execution_time=settings.crewai_max_execution_time,
        max_retry_limit=settings.crewai_max_retry_limit,
        planning=False,
    )
    with trace_span(
        "llm.selector",
        stage="document_product_selector",
        prompt_chars=len(prompt),
        response_format="DocumentProductSelection",
    ):
        result = await asyncio.wait_for(
            agent.kickoff_async(prompt, response_format=DocumentProductSelection),
            timeout=timeout_seconds,
        )
    if result.pydantic is not None:
        return DocumentProductSelection.model_validate(result.pydantic)
    return DocumentProductSelection.model_validate_json(result.raw)


def _document_context_unavailable_fact(
    *,
    reason_code: str,
    error_type: str = "",
    document_id: str = "",
    title: str = "",
    char_count: int | None = None,
) -> EvidenceFact:
    metadata = {
        "status": "unavailable",
        "reason_code": reason_code,
        "content_is_data_only": True,
    }
    if error_type:
        metadata["error_type"] = error_type
    if document_id:
        metadata["document_id"] = document_id
    if title:
        metadata["title"] = title
    if char_count is not None:
        metadata["char_count"] = char_count
    return EvidenceFact(
        fact_type="document_context_unavailable",
        value=False,
        source_type="document_mcp",
        source_id="document_mcp",
        metadata=metadata,
        artifact_type="document_context",
        scope=artifact_scope_for_evidence(
            channel_id="unknown",
            artifact_type="document_context",
            source_id=document_id or "document_mcp",
            source_type="document_mcp",
            metadata=metadata,
        ),
    )


def _sanitize_document_text(text: str) -> SanitizedDocumentText:
    metadata = {
        "sanitized": False,
        "internal_locator_redacted": False,
        "secret_redacted": False,
        "document_instruction_removed": False,
        "char_count": 0,
    }
    sanitized = _LOCAL_LOCATOR_RE.sub(_REDACTED_LOCATOR, str(text or ""))
    if sanitized != text:
        metadata["sanitized"] = True
        metadata["internal_locator_redacted"] = True

    without_secrets = _SECRET_RE.sub(_REDACTED_SECRET, sanitized)
    if without_secrets != sanitized:
        metadata["sanitized"] = True
        metadata["secret_redacted"] = True
    sanitized = without_secrets

    cleaned_lines: list[str] = []
    for line in sanitized.splitlines():
        if _is_document_instruction_line(line):
            metadata["sanitized"] = True
            metadata["document_instruction_removed"] = True
            cleaned_lines.append(_REMOVED_DOCUMENT_INSTRUCTION)
            continue
        cleaned_lines.append(line)
    sanitized = "\n".join(cleaned_lines).strip()
    metadata["char_count"] = len(sanitized)
    return SanitizedDocumentText(text=sanitized, metadata=metadata)


def _bound_sanitized_document_text(
    sanitized: SanitizedDocumentText,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS_PER_DOCUMENT,
) -> tuple[str, dict[str, object]]:
    text = sanitized.text
    metadata = dict(sanitized.metadata)
    metadata["truncated"] = False
    if len(text) <= max_chars:
        return text, metadata

    bounded = text[:max_chars].rstrip()
    metadata["truncated"] = True
    metadata["original_char_count"] = len(text)
    metadata["char_count"] = len(bounded)
    return bounded, metadata


def _is_document_instruction_line(line: str) -> bool:
    normalized = line.strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _PROMPT_INJECTION_PATTERNS)


def _retrieval_query(request: ReplyRequest, evidence_query: str | None) -> str:
    semantic_query = str(evidence_query or "").strip()
    if semantic_query:
        return semantic_query
    return str(request.message or "").strip()


def _select_document_text(
    content: str,
    message: str,
    canonical_context: CanonicalContext,
    *,
    max_chars: int,
) -> str:
    del message, canonical_context
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip()
