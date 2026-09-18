from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
    normalize_provider_failure_code,
    provider_failure_code_from_exception,
    provider_http_failure_code,
)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    (
        (401, "provider_auth_failed"),
        (403, "provider_auth_failed"),
        (429, "provider_rate_limited"),
        (500, "provider_http_error"),
    ),
)
def test_provider_http_status_maps_to_closed_code(
    status_code: int,
    expected: str,
) -> None:
    assert provider_http_failure_code(status_code) == expected


def test_arbitrary_provider_prose_normalizes_without_echoing_input() -> None:
    raw = "provider-secret-canary https://private.invalid model=secret-model"

    code = normalize_provider_failure_code(raw)
    raised = ProviderInvocationError(code)

    assert code == "provider_transport_unavailable"
    assert str(raised) == "provider_transport_unavailable"
    assert raw not in repr(raised)


def test_exception_status_is_classified_without_reading_exception_text() -> None:
    class ProviderStatusError(RuntimeError):
        status_code = 429

    code = provider_failure_code_from_exception(
        ProviderStatusError("rate-limit-secret-canary")
    )

    assert code == "provider_rate_limited"
