from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    retry_attempts: int = 0
    base_delay_seconds: float = 0.0
    multiplier: float = 2.0
    max_delay_seconds: float | None = None

    @property
    def max_attempts(self) -> int:
        return max(1, int(self.retry_attempts) + 1)

    def delay_for_retry(self, retry_index: int) -> float:
        delay = max(0.0, self.base_delay_seconds) * (self.multiplier ** retry_index)
        if self.max_delay_seconds is not None:
            delay = min(delay, max(0.0, self.max_delay_seconds))
        return delay


RetryCallback = Callable[[int, float, str], None]
ResultRetryPredicate = Callable[[T], str | None]
ExceptionRetryPredicate = Callable[[Exception], str | None]


async def run_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    should_retry_result: ResultRetryPredicate[T] | None = None,
    should_retry_exception: ExceptionRetryPredicate | None = None,
    on_retry: RetryCallback | None = None,
) -> T:
    for attempt in range(policy.max_attempts):
        try:
            result = await call()
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            reason = should_retry_exception(exc) if should_retry_exception else None
            if not reason or attempt >= policy.max_attempts - 1:
                raise
            await _sleep_before_retry(policy, attempt, reason, on_retry)
            continue

        reason = should_retry_result(result) if should_retry_result else None
        if not reason or attempt >= policy.max_attempts - 1:
            return result
        await _sleep_before_retry(policy, attempt, reason, on_retry)

    raise RuntimeError("retry loop exhausted")


async def _sleep_before_retry(
    policy: RetryPolicy,
    attempt: int,
    reason: str,
    on_retry: RetryCallback | None,
) -> None:
    delay_seconds = policy.delay_for_retry(attempt)
    if on_retry is not None:
        on_retry(attempt + 1, delay_seconds, reason)
    if delay_seconds:
        await asyncio.sleep(delay_seconds)
