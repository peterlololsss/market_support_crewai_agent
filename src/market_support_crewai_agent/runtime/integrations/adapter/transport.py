from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Final, Literal, NewType, Protocol, Self
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from market_support_crewai_agent.runtime.integrations.http_safety import (
    ResponseTooLargeError,
    read_bounded,
    redirect_rejecting_opener,
)

_RFC1918_NETWORKS: Final = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)
type _AdapterScheme = Literal["http", "https"]
_AdapterBearerToken = NewType("_AdapterBearerToken", str)
_SCHEME_DEFAULT_PORTS: Final[Mapping[str, tuple[_AdapterScheme, int]]] = (
    MappingProxyType({"http": ("http", 80), "https": ("https", 443)})
)
_HEADER_VALUE_MIN_CODEPOINT: Final = 32
_HEADER_VALUE_MAX_CODEPOINT: Final = 255
_HEADER_VALUE_DELETE_CODEPOINT: Final = 127
_MAX_RESPONSE_BYTES: Final = 4 * 1024 * 1024


class AdapterClientError(RuntimeError):
    """Raised when the WeCom adapter API boundary cannot return a valid result."""


@dataclass(frozen=True, slots=True)
class AdapterOrigin:
    scheme: _AdapterScheme
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class AdapterEndpoint:
    base_url: str
    origin: AdapterOrigin

    @classmethod
    def parse(cls, raw_url: str) -> AdapterEndpoint:
        if (
            not raw_url
            or raw_url != raw_url.strip()
            or any(
                ord(character) < 32 or ord(character) == 127 for character in raw_url
            )
        ):
            raise _invalid_url("adapter base URL")
        try:
            parsed = urlsplit(raw_url)
        except ValueError:
            raise _invalid_url("adapter base URL") from None
        if (
            parsed.username is not None
            or parsed.password is not None
            or "\\" in parsed.netloc
            or "?" in raw_url
            or "#" in raw_url
        ):
            raise _invalid_url("adapter base URL")
        origin = _parse_origin(raw_url, label="adapter base URL")
        match origin.scheme:
            case "http":
                if not _allows_cleartext(origin.host):
                    raise _invalid_url("adapter base URL")
            case "https":
                pass
        return cls(base_url=raw_url.rstrip("/"), origin=origin)

    def request_url(self, path: str) -> str:
        if not path.startswith("/"):
            raise AdapterClientError("adapter request path is invalid")
        url = f"{self.base_url}{path}"
        if _parse_origin(url, label="adapter request URL") != self.origin:
            raise AdapterClientError("adapter request origin mismatch")
        return url


class _AdapterResponse(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def geturl(self) -> str: ...

    def read(self, amt: int, /) -> bytes: ...


class _AdapterOpener(Protocol):
    def open(self, fullurl: Request, *, timeout: float) -> _AdapterResponse: ...


_REDIRECT_REJECTING_OPENER: Final[_AdapterOpener] = redirect_rejecting_opener()


@dataclass(frozen=True, slots=True)
class AdapterTransport:
    endpoint: AdapterEndpoint
    timeout_seconds: float
    api_key: _AdapterBearerToken | None

    @classmethod
    def create(
        cls,
        base_url: str,
        timeout_seconds: float,
        api_key: str | None,
    ) -> AdapterTransport:
        return cls(
            endpoint=AdapterEndpoint.parse(base_url),
            timeout_seconds=timeout_seconds,
            api_key=_parse_bearer_token(api_key),
        )

    def request_json(
        self,
        method: Literal["GET", "POST"],
        path: str,
        body: bytes | None = None,
    ) -> str:
        url = self.endpoint.request_url(path)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with _REDIRECT_REJECTING_OPENER.open(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_origin = _parse_origin(
                    response.geturl(),
                    label="adapter response URL",
                )
                if response_origin != self.endpoint.origin:
                    raise AdapterClientError("adapter response origin mismatch")
                return read_bounded(response, _MAX_RESPONSE_BYTES).decode("utf-8")
        except ResponseTooLargeError:
            raise AdapterClientError("adapter response exceeds size limit") from None
        except HTTPError as exc:
            status_code = exc.code
            exc.close()
            if 300 <= status_code < 400:
                raise AdapterClientError("adapter request redirect rejected") from None
            raise AdapterClientError(
                f"adapter request returned HTTP {status_code}"
            ) from None
        except URLError:
            raise AdapterClientError("adapter request failed") from None


def _parse_origin(url: str, *, label: str) -> AdapterOrigin:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
        scheme, default_port = _SCHEME_DEFAULT_PORTS[parsed.scheme.lower()]
    except ValueError:
        raise _invalid_url(label) from None
    except KeyError:
        raise _invalid_url(label) from None
    if not host or port == 0:
        raise _invalid_url(label)
    canonical_host = _canonical_host(host, label=label)
    return AdapterOrigin(
        scheme,
        canonical_host,
        default_port if port is None else port,
    )


def _canonical_host(host: str, *, label: str) -> str:
    normalized = host.rstrip(".").lower()
    if (
        not normalized
        or "%" in normalized
        or any(character.isspace() for character in normalized)
    ):
        raise _invalid_url(label)
    try:
        return ipaddress.ip_address(normalized).compressed
    except ValueError:
        try:
            return normalized.encode("idna").decode("ascii")
        except UnicodeError:
            raise _invalid_url(label) from None


def _allows_cleartext(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if address.is_loopback or address.is_link_local:
        return True
    match address:
        case ipaddress.IPv4Address():
            return any(address in network for network in _RFC1918_NETWORKS)
        case ipaddress.IPv6Address():
            return False


def _invalid_url(label: str) -> AdapterClientError:
    return AdapterClientError(f"{label} is invalid or unsafe")


def _parse_bearer_token(raw_token: str | None) -> _AdapterBearerToken | None:
    if raw_token is None:
        return None
    if any(
        ord(character) < _HEADER_VALUE_MIN_CODEPOINT
        or ord(character) > _HEADER_VALUE_MAX_CODEPOINT
        or ord(character) == _HEADER_VALUE_DELETE_CODEPOINT
        for character in raw_token
    ):
        raise AdapterClientError("adapter authentication is invalid")
    return _AdapterBearerToken(raw_token)
