from __future__ import annotations

from typing import Final, Literal


ProviderFailureCodeV1 = Literal[
    "provider_output_missing",
    "provider_output_type",
    "provider_output_encoding",
    "provider_output_too_large",
    "provider_output_contract",
    "provider_timeout",
    "provider_auth_failed",
    "provider_rate_limited",
    "provider_http_error",
    "provider_transport_unavailable",
    "provider_internal_error",
    "direct_provider_stdio_violation",
]

_PROVIDER_FAILURE_CODES: Final[dict[str, ProviderFailureCodeV1]] = {
    "provider_output_missing": "provider_output_missing",
    "provider_output_type": "provider_output_type",
    "provider_output_encoding": "provider_output_encoding",
    "provider_output_too_large": "provider_output_too_large",
    "provider_output_contract": "provider_output_contract",
    "provider_timeout": "provider_timeout",
    "provider_auth_failed": "provider_auth_failed",
    "provider_rate_limited": "provider_rate_limited",
    "provider_http_error": "provider_http_error",
    "provider_transport_unavailable": "provider_transport_unavailable",
    "provider_internal_error": "provider_internal_error",
    "direct_provider_stdio_violation": "direct_provider_stdio_violation",
}
_OUTPUT_FAILURE_CODES: Final[frozenset[ProviderFailureCodeV1]] = frozenset(
    {
        "provider_output_missing",
        "provider_output_type",
        "provider_output_encoding",
        "provider_output_too_large",
        "provider_output_contract",
    }
)
_HTTP_FAILURE_CODES: Final[dict[int, ProviderFailureCodeV1]] = {
    401: "provider_auth_failed",
    403: "provider_auth_failed",
    429: "provider_rate_limited",
}
_EXCEPTION_FAILURE_CODES: Final[dict[str, ProviderFailureCodeV1]] = {
    "AuthenticationError": "provider_auth_failed",
    "PermissionDeniedError": "provider_auth_failed",
    "RateLimitError": "provider_rate_limited",
    "APITimeoutError": "provider_timeout",
    "ConnectTimeout": "provider_timeout",
    "PoolTimeout": "provider_timeout",
    "ReadTimeout": "provider_timeout",
    "TimeoutException": "provider_timeout",
    "WriteTimeout": "provider_timeout",
    "APIConnectionError": "provider_transport_unavailable",
    "ConnectError": "provider_transport_unavailable",
    "ConnectionError": "provider_transport_unavailable",
    "RequestError": "provider_transport_unavailable",
    "ServiceUnavailableError": "provider_transport_unavailable",
    "APIStatusError": "provider_http_error",
    "HTTPStatusError": "provider_http_error",
    "InternalServerError": "provider_http_error",
}


class ProviderInvocationError(RuntimeError):
    __slots__ = ("_code",)

    def __init__(self, code: ProviderFailureCodeV1) -> None:
        self._code = code
        super().__init__(code)

    @property
    def code(self) -> ProviderFailureCodeV1:
        return self._code

    def __str__(self) -> str:
        return self.code


def normalize_provider_failure_code(code: str) -> ProviderFailureCodeV1:
    return _PROVIDER_FAILURE_CODES.get(code, "provider_transport_unavailable")


def provider_failure_code_from_exception(
    exc: Exception,
) -> ProviderFailureCodeV1:
    if isinstance(exc, ProviderInvocationError):
        return exc.code
    if isinstance(exc, TimeoutError):
        return "provider_timeout"
    status_code = _provider_status_code(exc)
    if status_code is not None:
        return provider_http_failure_code(status_code)
    return _EXCEPTION_FAILURE_CODES.get(
        type(exc).__name__,
        "provider_transport_unavailable",
    )


def provider_http_failure_code(status_code: int) -> ProviderFailureCodeV1:
    return _HTTP_FAILURE_CODES.get(status_code, "provider_http_error")


def is_provider_output_failure(code: ProviderFailureCodeV1) -> bool:
    return code in _OUTPUT_FAILURE_CODES


def _provider_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response_status_code = getattr(
        getattr(exc, "response", None),
        "status_code",
        None,
    )
    if isinstance(response_status_code, int):
        return response_status_code
    return None
