from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import ClassVar, Final
from urllib.request import OpenerDirector, Request

import pytest
from typing_extensions import override

from market_support_crewai_agent.runtime.integrations.adapter.client import (
    AdapterResolveClient,
)
from market_support_crewai_agent.settings_model import Settings

ADAPTER_TIMEOUT_SECONDS: Final = 5.0
SYNTHETIC_API_KEY: Final = "synthetic-security-test-key"
CAPABILITIES_BODY: Final = (
    b'{"service":"assistant-wecom-market-agent-adapter",'
    b'"contract_version":"adapter-resolve",'
    b'"batch_contract_version":"adapter-resolve-batch",'
    b'"action_contract_version":"adapter-action",'
    b'"endpoints":{"health":"/health",'
    b'"capabilities":"/adapter/capabilities",'
    b'"metrics":"/adapter/metrics",'
    b'"resolve":"/adapter/resolve",'
    b'"batch_resolve":"/adapter/resolve/batch"},'
    b'"resolve_types":["material_pack","weekly_report",'
    b'"monthly_report","sales_mention"],'
    b'"statuses":["resolved","missing","ambiguous",'
    b'"forbidden","temporarily_unavailable"],'
    b'"max_batch_requests":16,'
    b'"max_request_body_bytes":65536,'
    b'"cache_ttl_seconds":30,'
    b'"cache_max_entries":512,'
    b'"auth":{"header_schemes":["Authorization: Bearer <key>",'
    b'"X-API-Key: <key>"],'
    b'"protected_endpoints":["/adapter/resolve",'
    b'"/adapter/resolve/batch","/adapter/metrics"]}}'
)


def write_capabilities(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(CAPABILITIES_BODY)))
    handler.end_headers()
    _ = handler.wfile.write(CAPABILITIES_BODY)


class CapabilityHandler(BaseHTTPRequestHandler):
    paths: ClassVar[list[str]] = []
    authorizations: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        type(self).paths.append(self.path)
        type(self).authorizations.append(self.headers.get("Authorization", ""))
        write_capabilities(self)

    @override
    def log_message(self, format: str, *args: str) -> None:
        del format, args


class CrossOriginRedirectHandler(BaseHTTPRequestHandler):
    location: ClassVar[str] = ""
    paths: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        type(self).paths.append(self.path)
        self.send_response(302)
        self.send_header("Location", type(self).location)
        self.end_headers()

    @override
    def log_message(self, format: str, *args: str) -> None:
        del format, args


class SameOriginRedirectHandler(BaseHTTPRequestHandler):
    paths: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        type(self).paths.append(self.path)
        if self.path == "/adapter/capabilities":
            self.send_response(302)
            self.send_header("Location", "/redirected-capabilities")
            self.end_headers()
            return
        write_capabilities(self)

    @override
    def log_message(self, format: str, *args: str) -> None:
        del format, args


@contextmanager
def running_server(
    handler: type[BaseHTTPRequestHandler],
) -> Generator[ThreadingHTTPServer, None, None]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def adapter_client(
    base_url: str,
    api_key: str | None = SYNTHETIC_API_KEY,
) -> AdapterResolveClient:
    return AdapterResolveClient(
        Settings(
            adapter_base_url=base_url,
            adapter_api_key=api_key,
        ),
    )


@dataclass(frozen=True, slots=True)
class FakeResponse:
    response_url: str

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback

    def geturl(self) -> str:
        return self.response_url

    def read(self) -> bytes:
        return CAPABILITIES_BODY


def install_fake_response(
    monkeypatch: pytest.MonkeyPatch,
    response_url: str,
    captured_requests: list[Request],
) -> None:
    def open_response(
        _opener: OpenerDirector,
        fullurl: str | Request,
        data: bytes | None = None,
        timeout: float = ADAPTER_TIMEOUT_SECONDS,
    ) -> FakeResponse:
        assert data is None
        assert timeout == ADAPTER_TIMEOUT_SECONDS
        if isinstance(fullurl, Request):
            captured_requests.append(fullurl)
        else:
            pytest.fail("adapter transport submitted raw URL")
        return FakeResponse(response_url)

    monkeypatch.setattr(OpenerDirector, "open", open_response)
