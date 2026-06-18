from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable

from market_support_crewai_agent.schemas import (
    ActionExecutionFeedback,
    ActionFeedbackRequest,
)


DEFAULT_ACTION_LEDGER_TTL_SECONDS = 86400


@dataclass(frozen=True)
class ActionLedgerRecord:
    conversation_key: str
    group_id: str
    sender_id: str
    context_id: str | None
    response_id: str | None
    execution: ActionExecutionFeedback
    received_at: datetime
    dedupe_key: tuple


class ActionLedger:
    """Thread-safe in-memory adapter execution ledger.

    This is intentionally small and bounded. It gives the runtime a stable
    integration target for adapter-confirmed execution status before a durable
    store is introduced.
    """

    def __init__(
        self,
        max_records: int = 5000,
        ttl_seconds: int | None = DEFAULT_ACTION_LEDGER_TTL_SECONDS,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be greater than zero")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        self._max_records = max_records
        self._ttl = None if ttl_seconds is None else timedelta(seconds=ttl_seconds)
        self._now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self._records: list[ActionLedgerRecord] = []
        self._keys: set[tuple] = set()
        self._lock = RLock()

    def record_feedback(self, feedback: ActionFeedbackRequest) -> int:
        now = self._now()
        candidates = []
        for index, execution in enumerate(feedback.executions):
            key = _feedback_record_key(feedback, execution, index)
            record = ActionLedgerRecord(
                conversation_key=feedback.conversation_key,
                group_id=feedback.group_id,
                sender_id=feedback.sender_id,
                context_id=feedback.context_id,
                response_id=feedback.response_id,
                execution=execution,
                received_at=now,
                dedupe_key=key,
            )
            candidates.append((key, record))
        if not candidates:
            return 0

        with self._lock:
            self._cleanup_expired_locked(now)
            records = []
            for key, record in candidates:
                if key in self._keys:
                    continue
                self._keys.add(key)
                records.append(record)
            if not records:
                return 0
            self._records.extend(records)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records :]
                self._keys = {record.dedupe_key for record in self._records}
            return len(records)

    def recent_for_conversation(
        self,
        conversation_key: str,
        limit: int = 20,
    ) -> list[ActionLedgerRecord]:
        if limit <= 0:
            return []
        with self._lock:
            self._cleanup_expired_locked(self._now())
            matches = [
                record
                for record in self._records
                if record.conversation_key == conversation_key
            ]
            return list(matches[-limit:])

    def recent_executed_for_conversation(
        self,
        conversation_key: str,
        limit: int = 20,
    ) -> list[ActionLedgerRecord]:
        if limit <= 0:
            return []
        with self._lock:
            self._cleanup_expired_locked(self._now())
            matches = [
                record
                for record in self._records
                if record.conversation_key == conversation_key
                and record.execution.status == "executed"
            ]
            return list(matches[-limit:])

    def by_context_id(self, context_id: str) -> list[ActionLedgerRecord]:
        with self._lock:
            self._cleanup_expired_locked(self._now())
            return [
                record
                for record in self._records
                if record.context_id == context_id
            ]

    def count(self) -> int:
        with self._lock:
            self._cleanup_expired_locked(self._now())
            return len(self._records)

    def cleanup_expired(self) -> int:
        """Remove expired records immediately and return the deletion count."""
        with self._lock:
            return self._cleanup_expired_locked(self._now())

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._keys.clear()

    def _cleanup_expired_locked(self, now: datetime) -> int:
        if self._ttl is None:
            return 0
        retained = [
            record for record in self._records if record.received_at + self._ttl > now
        ]
        removed = len(self._records) - len(retained)
        if removed:
            self._records = retained
            self._keys = {record.dedupe_key for record in self._records}
        return removed

    def _now(self) -> datetime:
        value = self._now_factory()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


_DEFAULT_ACTION_LEDGER = ActionLedger()


def get_action_ledger() -> ActionLedger:
    return _DEFAULT_ACTION_LEDGER


def _feedback_record_key(
    feedback: ActionFeedbackRequest,
    execution: ActionExecutionFeedback,
    index: int,
) -> tuple:
    return (
        feedback.conversation_key,
        feedback.context_id or "",
        feedback.response_id or "",
        execution.action_id or "index:{}".format(index),
        execution.action_type,
        execution.status,
        execution.material_type or "",
        execution.material_pack_option or "",
        execution.material_id or "",
        execution.version or "",
    )
