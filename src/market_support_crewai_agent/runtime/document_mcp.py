from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from market_support_crewai_agent.runtime.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.planning import ReplyPlan
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings, get_settings

_DOC_CAPABILITY = "query_internal_company_info"
_MCP_ACCEPT_HEADER = "application/json, text/event-stream"
_MAX_CHARS_PER_DOCUMENT = 6000
_KNOWLEDGE_TERMS = (
    "公司",
    "团队",
    "股权",
    "规模",
    "地址",
    "官网",
    "公众号",
    "指数",
    "策略",
    "因子",
    "换手",
    "容量",
    "成分股",
    "持仓",
    "超额",
    "分红",
    "净值",
    "开放",
    "对冲",
    "中性",
)
_COMPANY_TERMS = ("公司", "团队", "股权", "地址", "官网", "公众号", "衍复")
_QUESTION_BLOCK_RE = re.compile(r"(?=Q[:：])")
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


@dataclass(frozen=True)
class DocumentEvidenceChunk:
    document_id: str
    title: str
    text: str


class DocumentMcpError(RuntimeError):
    """Raised when the document MCP cannot provide bounded evidence."""


@dataclass(frozen=True)
class SanitizedDocumentText:
    text: str
    metadata: dict[str, object]


class DocumentMcpClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = (self.settings.doc_mcp_base_url or "").rstrip("/")
        self.timeout = self.settings.doc_mcp_timeout_seconds

    def fetch_context(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        *,
        max_documents: int = 2,
        max_chars_per_document: int = _MAX_CHARS_PER_DOCUMENT,
    ) -> list[DocumentEvidenceChunk]:
        if not self.base_url:
            return []

        products = self._list_products()
        selected_products = _select_products(
            products,
            request.message,
            canonical_context,
            max_documents=max_documents,
        )
        if not selected_products:
            return []

        documents = self._get_documents(
            [str(product.get("id") or "") for product in selected_products]
        )
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
                        request.message,
                        canonical_context,
                        max_chars=max_chars_per_document,
                    ),
                )
            )
        return chunks

    async def fetch_context_async(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        *,
        max_documents: int = 2,
        max_chars_per_document: int = _MAX_CHARS_PER_DOCUMENT,
    ) -> list[DocumentEvidenceChunk]:
        return await asyncio.to_thread(
            self.fetch_context,
            request,
            canonical_context,
            max_documents=max_documents,
            max_chars_per_document=max_chars_per_document,
        )

    def _list_products(self) -> list[dict]:
        result = self._call_tool("list_products", {})
        payload = _tool_text_json(result)
        products = payload.get("products")
        return products if isinstance(products, list) else []

    def _get_documents(self, document_ids: list[str]) -> list[dict]:
        result = self._call_tool(
            "get_documents",
            {"documentIds": [document_id for document_id in document_ids if document_id]},
        )
        payload = _tool_text_json(result)
        documents = payload.get("documents")
        return documents if isinstance(documents, list) else []

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
        plan: ReplyPlan,
        policy: PolicyManifest,
    ) -> list[EvidenceFact]:
        if not self.settings.doc_mcp_enabled or not self.settings.doc_mcp_base_url:
            return []
        if plan.compliance.is_compliant is not True:
            return []
        if not any(
            evidence_request.capability == _DOC_CAPABILITY
            for evidence_request in plan.evidence_requests
        ):
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
            chunks = await self.client.fetch_context_async(request, canonical_context)
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
            evidence_text, metadata = _bound_sanitized_document_text(sanitized)
            facts.append(
                EvidenceFact(
                    fact_type="document_context",
                    value=evidence_text,
                    source_type="document_mcp",
                    source_id=chunk.document_id,
                    metadata={
                        "title": chunk.title,
                        "content_is_data_only": True,
                        **metadata,
                    },
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
        plan: ReplyPlan,
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
    max_chars: int = _MAX_CHARS_PER_DOCUMENT,
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


def _select_products(
    products: list[dict],
    message: str,
    canonical_context: CanonicalContext,
    *,
    max_documents: int,
) -> list[dict]:
    scored = [
        (_score_product(product, message, canonical_context), product)
        for product in products
    ]
    selected = [
        product
        for score, product in sorted(scored, key=lambda item: item[0], reverse=True)
        if score > 0
    ]
    if selected:
        return selected[:max_documents]

    faq_product = _first_category(products, "常见问答")
    return [faq_product] if faq_product else []


def _score_product(
    product: dict,
    message: str,
    canonical_context: CanonicalContext,
) -> int:
    normalized_message = message.lower()
    searchable = _product_searchable_text(product)
    score = 0

    strategy = (canonical_context.selected_strategy or "").lower()
    if strategy and strategy in searchable:
        score += 100

    if any(term in normalized_message for term in _COMPANY_TERMS):
        category = str(product.get("category") or "")
        if category == "公司介绍":
            score += 60
        if category == "常见问答":
            score += 20

    for keyword in product.get("keywords") or []:
        normalized_keyword = str(keyword).lower()
        if normalized_keyword and normalized_keyword in normalized_message:
            score += 10

    product_name = str(product.get("name") or "").lower()
    product_title = str(product.get("title") or "").lower()
    if product_name and product_name in normalized_message:
        score += 40
    if product_title and product_title in normalized_message:
        score += 40

    if score == 0 and str(product.get("category") or "") == "常见问答":
        if any(term in normalized_message for term in _KNOWLEDGE_TERMS):
            score += 5

    return score


def _product_searchable_text(product: dict) -> str:
    values = [
        product.get("id"),
        product.get("name"),
        product.get("title"),
        product.get("category"),
        product.get("summary"),
        " ".join(str(keyword) for keyword in product.get("keywords") or []),
    ]
    return " ".join(str(value or "") for value in values).lower()


def _first_category(products: list[dict], category: str) -> dict | None:
    for product in products:
        if product.get("category") == category:
            return product
    return None


def _select_document_text(
    content: str,
    message: str,
    canonical_context: CanonicalContext,
    *,
    max_chars: int,
) -> str:
    if len(content) <= max_chars:
        return content

    blocks = [block.strip() for block in _QUESTION_BLOCK_RE.split(content) if block.strip()]
    if not blocks:
        return content[:max_chars]

    scored_blocks = [
        (_score_text_block(block, message, canonical_context), block)
        for block in blocks
    ]
    selected = [
        block
        for score, block in sorted(scored_blocks, key=lambda item: item[0], reverse=True)
        if score > 0
    ]
    if not selected:
        selected = blocks[:1]

    output: list[str] = []
    current_len = 0
    for block in selected:
        if current_len >= max_chars:
            break
        remaining = max_chars - current_len
        piece = block[:remaining]
        output.append(piece)
        current_len += len(piece) + 2
    return "\n\n".join(output)


def _score_text_block(
    block: str,
    message: str,
    canonical_context: CanonicalContext,
) -> int:
    normalized_block = block.lower()
    normalized_message = message.lower()
    score = 0
    strategy = (canonical_context.selected_strategy or "").lower()
    if strategy and strategy in normalized_block:
        score += 20
    for term in _KNOWLEDGE_TERMS:
        if term in normalized_message and term in normalized_block:
            score += 5
    for token in _message_tokens(normalized_message):
        if token in normalized_block:
            score += 2
    return score


def _message_tokens(normalized_message: str) -> list[str]:
    raw_tokens = re.split(r"[\s,，。；;:：/？?！!（）()、]+", normalized_message)
    return [token for token in raw_tokens if len(token) >= 2]
