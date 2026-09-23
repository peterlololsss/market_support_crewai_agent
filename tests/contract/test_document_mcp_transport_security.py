from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import ClassVar

import anyio
import pytest
from typing_extensions import override

from market_support_crewai_agent.runtime.integrations import http_safety
from market_support_crewai_agent.runtime.integrations.document_mcp import (
    parsing as document_parsing,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.client import (
    DocumentMcpClient,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.parsing import (
    DocumentMcpError,
)
from market_support_crewai_agent.settings_model import Settings
from tests.contract.adapter_transport_security_fixtures import running_server

_EMPTY_PRODUCTS_BODY = json.dumps(
    {
        "jsonrpc": "2.0",
        "id": "tools/call",
        "result": {"content": [{"type": "text", "text": '{"products":[]}'}]},
    }
).encode("utf-8")


class ProductsHandler(BaseHTTPRequestHandler):
    paths: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        type(self).paths.append(self.path)
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_EMPTY_PRODUCTS_BODY)))
        self.end_headers()
        _ = self.wfile.write(_EMPTY_PRODUCTS_BODY)

    def do_GET(self) -> None:
        type(self).paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_EMPTY_PRODUCTS_BODY)))
        self.end_headers()
        _ = self.wfile.write(_EMPTY_PRODUCTS_BODY)

    @override
    def log_message(self, format: str, *args: str) -> None:
        del format, args


class RedirectHandler(BaseHTTPRequestHandler):
    location: ClassVar[str] = ""
    paths: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        type(self).paths.append(self.path)
        _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(302)
        self.send_header("Location", type(self).location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    @override
    def log_message(self, format: str, *args: str) -> None:
        del format, args


def _fetch_all(base_url: str) -> list[document_parsing.JsonMap]:
    client = DocumentMcpClient(
        Settings(doc_mcp_enabled=True, doc_mcp_base_url=base_url)
    )

    async def fetch() -> list[document_parsing.JsonMap]:
        return await client.fetch_all_documents_async(evidence_query="company")

    return anyio.run(fetch)


def test_document_mcp_rejects_redirect_before_second_request() -> None:
    ProductsHandler.paths = []
    RedirectHandler.paths = []

    with running_server(ProductsHandler) as destination:
        RedirectHandler.location = f"http://127.0.0.1:{destination.server_port}/mcp"
        with (
            running_server(RedirectHandler) as source,
            pytest.raises(DocumentMcpError, match="^document MCP redirect rejected$"),
        ):
            _ = _fetch_all(f"http://127.0.0.1:{source.server_port}")

    assert RedirectHandler.paths == ["/mcp"]
    assert ProductsHandler.paths == []


def test_document_mcp_rejects_response_larger_than_byte_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ProductsHandler.paths = []
    monkeypatch.setattr(document_parsing, "_MAX_RESPONSE_BYTES", 32)

    with (
        running_server(ProductsHandler) as server,
        pytest.raises(
            DocumentMcpError, match="^document MCP response exceeds size limit$"
        ),
    ):
        _ = _fetch_all(f"http://127.0.0.1:{server.server_port}")

    assert ProductsHandler.paths == ["/mcp"]


def test_document_mcp_accepts_bounded_same_origin_response() -> None:
    ProductsHandler.paths = []

    with running_server(ProductsHandler) as server:
        documents = _fetch_all(f"http://127.0.0.1:{server.server_port}")

    assert documents == []
    assert ProductsHandler.paths == ["/mcp"]


def test_redirect_rejecting_opener_ignores_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: proxy variables that a default urllib opener would route through.
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    ProductsHandler.paths = []

    # When: the shared opener is built after the environment changes.
    opener = http_safety.redirect_rejecting_opener()
    with (
        running_server(ProductsHandler) as server,
        opener.open(f"http://127.0.0.1:{server.server_port}/mcp", timeout=2) as reply,
    ):
        body = http_safety.read_bounded(reply, len(_EMPTY_PRODUCTS_BODY))

    # Then: the request reaches the server directly instead of the dead proxy port.
    assert body == _EMPTY_PRODUCTS_BODY
    assert ProductsHandler.paths == ["/mcp"]
