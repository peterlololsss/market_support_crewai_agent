from __future__ import annotations

import json
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Protocol, Self
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from pydantic import JsonValue, TypeAdapter, ValidationError

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1

MCP_ACCEPT_HEADER: Final = "application/json, text/event-stream"
JsonMap = dict[str, JsonValue]
_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class DocumentEvidenceChunk:
    document_id: str
    title: str
    text: str
    metadata: JsonMap | None = None


class DocumentMcpError(RuntimeError):
    """Raised when the document MCP cannot provide bounded evidence."""


class _DocumentMcpResponse(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def read(self) -> bytes: ...


class _DocumentMcpOpener(Protocol):
    def open(self, fullurl: Request, *, timeout: float) -> _DocumentMcpResponse: ...


_DOCUMENT_MCP_OPENER: _DocumentMcpOpener = build_opener()


def document_mcp_request_body(method: str, params: JsonMap) -> bytes:
    payload = {"jsonrpc": "2.0", "id": method, "method": method, "params": params}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def read_document_mcp_response(http_request: Request, *, timeout: float) -> str:
    response_context: _DocumentMcpResponse = _DOCUMENT_MCP_OPENER.open(
        http_request,
        timeout=timeout,
    )
    with response_context as response:
        return response.read().decode("utf-8", errors="replace")


def parse_mcp_result(raw: str) -> JsonMap:
    message = parse_mcp_message(raw)
    if "error" in message:
        raise DocumentMcpError(f"document MCP returned error: {message['error']}")
    result = message.get("result")
    if not isinstance(result, dict):
        raise DocumentMcpError("document MCP returned an invalid result")
    return result


def parse_mcp_message(raw: str) -> JsonMap:
    stripped = raw.strip()
    if not stripped:
        raise DocumentMcpError("document MCP returned an empty response")
    if stripped.startswith("{"):
        return parse_json_map(
            stripped,
            invalid_message="document MCP returned invalid JSON",
            non_object_message="document MCP returned a non-object JSON response",
        )

    data_lines = [
        line.removeprefix("data:").strip()
        for line in stripped.splitlines()
        if line.startswith("data:")
    ]
    if not data_lines:
        raise DocumentMcpError("document MCP SSE response has no data line")
    return parse_json_map(
        data_lines[-1],
        invalid_message="document MCP SSE data is invalid JSON",
        non_object_message="document MCP SSE data is not a JSON object",
    )


def tool_text_json(result: JsonMap) -> JsonMap:
    content = result.get("content")
    if not isinstance(content, list):
        return {}
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        payload = parse_json_map_or_none(str(item.get("text") or "{}"))
        if payload is not None:
            return payload
    return {}


def parse_json_map(
    raw_json: str,
    *,
    invalid_message: str,
    non_object_message: str,
) -> JsonMap:
    try:
        payload = _JSON_VALUE_ADAPTER.validate_json(raw_json)
    except ValidationError as exc:
        raise DocumentMcpError(invalid_message) from exc
    if not isinstance(payload, dict):
        raise DocumentMcpError(non_object_message)
    return payload


def parse_json_map_or_none(raw_json: str) -> JsonMap | None:
    try:
        payload = _JSON_VALUE_ADAPTER.validate_json(raw_json)
    except ValidationError:
        return None
    return payload if isinstance(payload, dict) else None


def json_maps_from_value(value: JsonValue | None) -> list[JsonMap]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def json_list_from_value(value: JsonValue | None) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def document_mcp_transport_error(exc: HTTPError | URLError) -> DocumentMcpError:
    if isinstance(exc, HTTPError):
        detail = exc.read().decode("utf-8", errors="replace")
        return DocumentMcpError(f"document MCP returned HTTP {exc.code}: {detail}")
    return DocumentMcpError(f"document MCP request failed: {exc}")


def document_chunks_from_payloads(
    documents: list[JsonMap],
    *,
    query_text: str,
    max_chars: int,
) -> list[DocumentEvidenceChunk]:
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
                text=select_document_text(content, query_text, max_chars=max_chars),
                metadata=document_artifact_metadata(document),
            )
        )
    return chunks


def document_artifact_metadata(document: JsonMap) -> JsonMap:
    metadata = document.get("metadata")
    output: JsonMap = dict(metadata) if isinstance(metadata, dict) else {}
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


def retrieval_query(request: KernelReplyRequestV1, evidence_query: str | None) -> str:
    semantic_query = str(evidence_query or "").strip()
    if semantic_query:
        return semantic_query
    return str(request.message or "").strip()


def select_document_text(content: str, message: str, *, max_chars: int) -> str:
    del message
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip()
