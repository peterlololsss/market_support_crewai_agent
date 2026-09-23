from __future__ import annotations

from email.message import Message
from typing import IO, Protocol, override
from urllib.error import HTTPError
from urllib.request import (
    HTTPRedirectHandler,
    OpenerDirector,
    ProxyHandler,
    Request,
    build_opener,
)


class ResponseTooLargeError(RuntimeError):
    """Raised when a response body exceeds its transport's byte ceiling."""


class BoundedReadable(Protocol):
    def read(self, amt: int, /) -> bytes: ...


class _RejectRedirectHandler(HTTPRedirectHandler):
    @override
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> Request | None:
        raise HTTPError(req.full_url, code, msg, headers, fp)


def redirect_rejecting_opener() -> OpenerDirector:
    # Ignores proxy environment variables and surfaces every 3xx as an HTTPError,
    # so a response can only come from the configured origin.
    return build_opener(ProxyHandler({}), _RejectRedirectHandler())


def read_bounded(response: BoundedReadable, max_bytes: int) -> bytes:
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ResponseTooLargeError(f"response body exceeds {max_bytes} bytes")
    return body
