from __future__ import annotations

import asyncio

from market_support_crewai_agent.runtime.llm.retry import RetryPolicy, run_with_retry


def test_retry_policy_uses_exponential_delay_with_cap():
    policy = RetryPolicy(
        retry_attempts=3,
        base_delay_seconds=0.25,
        multiplier=2,
        max_delay_seconds=0.75,
    )

    assert policy.max_attempts == 4
    assert policy.delay_for_retry(0) == 0.25
    assert policy.delay_for_retry(1) == 0.5
    assert policy.delay_for_retry(2) == 0.75


def test_run_with_retry_retries_retryable_results(monkeypatch):
    sleeps: list[float] = []
    attempts = {"count": 0}

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def call():
        attempts["count"] += 1
        return "" if attempts["count"] < 3 else "ok"

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        run_with_retry(
            call,
            policy=RetryPolicy(retry_attempts=3, base_delay_seconds=0.1),
            should_retry_result=lambda value: "empty" if not value else None,
        )
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_run_with_retry_does_not_retry_timeout(monkeypatch):
    async def fake_sleep(_delay):
        raise AssertionError("timeout should not sleep before retry")

    async def call():
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    try:
        asyncio.run(
            run_with_retry(
                call,
                policy=RetryPolicy(retry_attempts=3, base_delay_seconds=0.1),
                should_retry_exception=lambda _exc: "retryable",
            )
        )
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("expected TimeoutError")
