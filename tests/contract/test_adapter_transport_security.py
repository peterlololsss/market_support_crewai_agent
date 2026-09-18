from __future__ import annotations

from urllib.request import Request

import pytest

from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
)
from tests.contract.adapter_transport_security_fixtures import (
    CapabilityHandler,
    CrossOriginRedirectHandler,
    SameOriginRedirectHandler,
    adapter_client,
    install_fake_response,
    running_server,
)


def test_adapter_client_rejects_cross_origin_redirect_before_second_request() -> None:
    CapabilityHandler.paths = []
    CapabilityHandler.authorizations = []
    CrossOriginRedirectHandler.paths = []

    with running_server(CapabilityHandler) as destination:
        CrossOriginRedirectHandler.location = (
            f"http://127.0.0.1:{destination.server_port}/adapter/capabilities"
        )
        with running_server(CrossOriginRedirectHandler) as source:
            with pytest.raises(
                AdapterClientError,
                match="^adapter request redirect rejected$",
            ):
                _ = adapter_client(
                    f"http://127.0.0.1:{source.server_port}",
                ).capabilities()

    assert CrossOriginRedirectHandler.paths == ["/adapter/capabilities"]
    assert CapabilityHandler.paths == []
    assert CapabilityHandler.authorizations == []


def test_adapter_client_rejects_same_origin_redirect_before_second_request() -> None:
    SameOriginRedirectHandler.paths = []

    with running_server(SameOriginRedirectHandler) as server:
        with pytest.raises(
            AdapterClientError,
            match="^adapter request redirect rejected$",
        ):
            _ = adapter_client(f"http://127.0.0.1:{server.server_port}").capabilities()

    assert SameOriginRedirectHandler.paths == ["/adapter/capabilities"]


@pytest.mark.parametrize(
    "api_key",
    [
        "synthetic-credential-marker\r\nX-Probe: synthetic-credential-marker",
        "synthetic-credential-marker-\u2603",
        "synthetic-credential-marker\x00suffix",
    ],
    ids=["crlf", "non-latin-1", "c0-control"],
)
def test_adapter_client_rejects_unsafe_bearer_before_network_io(
    api_key: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    CapabilityHandler.paths = []
    CapabilityHandler.authorizations = []

    with running_server(CapabilityHandler) as server:
        with pytest.raises(
            AdapterClientError,
            match="^adapter authentication is invalid$",
        ) as caught:
            _ = adapter_client(
                f"http://127.0.0.1:{server.server_port}",
                api_key,
            ).capabilities()

    assert "synthetic-credential-marker" not in str(caught.value)
    assert "synthetic-credential-marker" not in caplog.text
    assert CapabilityHandler.paths == []
    assert CapabilityHandler.authorizations == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://8.8.8.8:8011",
        "http://adapter.internal:8011",
        "http://localhost:8011",
        "ftp://127.0.0.1:8011",
        "file:///tmp/adapter.sock",
        "https://user:password@example.com",
        "https://example.com/adapter?tenant=primary",
        "https://example.com/adapter#fragment",
        "https:///adapter",
        "https://[::1",
        "https://example.com:0",
    ],
)
def test_adapter_client_rejects_unsafe_configured_base_url(base_url: str) -> None:
    with pytest.raises(
        AdapterClientError, match="^adapter base URL is invalid or unsafe$"
    ):
        _ = adapter_client(base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://adapter.example.com",
        "http://127.0.0.1:8011",
        "http://[::1]:8011",
        "http://10.1.2.3:8011",
        "http://172.16.0.1:8011",
        "http://172.31.255.255:8011",
        "http://10.0.0.11:8011",
        "http://169.254.1.2:8011",
        "http://[fe80::1]:8011",
    ],
)
def test_adapter_client_accepts_https_or_literal_private_http(base_url: str) -> None:
    client = adapter_client(base_url)

    assert client.base_url == base_url


def test_adapter_client_rejects_response_from_origin_other_than_configured_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[Request] = []
    install_fake_response(
        monkeypatch,
        "https://attacker.example/adapter/capabilities",
        captured_requests,
    )

    with pytest.raises(AdapterClientError, match="^adapter response origin mismatch$"):
        _ = adapter_client("https://adapter.example.com").capabilities()

    assert [request.full_url for request in captured_requests] == [
        "https://adapter.example.com/adapter/capabilities",
    ]


def test_adapter_client_https_request_authenticates_only_configured_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[Request] = []
    install_fake_response(
        monkeypatch,
        "https://adapter.example.com:443/adapter/capabilities",
        captured_requests,
    )

    result = adapter_client("https://adapter.example.com").capabilities()

    assert result.service == "assistant-wecom-market-agent-adapter"
    assert len(captured_requests) == 1
    assert captured_requests[0].full_url == (
        "https://adapter.example.com/adapter/capabilities"
    )
    assert captured_requests[0].get_header("Authorization") == (
        "Bearer synthetic-security-test-key"
    )
