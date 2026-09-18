from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock

from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    state_key_ref,
)
from market_support_crewai_agent.schemas.feedback import (
    ActionExecutionFeedbackV2,
    ActionFeedbackRequestV2,
)

DEFAULT_ACTION_LEDGER_TTL_SECONDS = 86400


@dataclass(frozen=True)
class ActionLedgerRecord:
    state_key: ConversationStateKey
    state_key_ref: str
    group_id: str
    sender_id: str
    context_id: str | None
    response_id: str | None
    execution: ActionExecutionFeedbackV2
    received_at: datetime
    dedupe_key: tuple[str, ...]


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
        self._max_records: int = max_records
        self._ttl: timedelta | None = (
            None if ttl_seconds is None else timedelta(seconds=ttl_seconds)
        )
        self._now_factory: Callable[[], datetime] = now_factory or (
            lambda: datetime.now(UTC)
        )
        self._records: list[ActionLedgerRecord] = []
        self._keys: set[tuple[str, ...]] = set()
        self._lock: RLock = RLock()

    def record_feedback(
        self,
        feedback: ActionFeedbackRequestV2,
        state_key: ConversationStateKey,
    ) -> int:
        now = self._now()
        candidates: list[tuple[tuple[str, ...], ActionLedgerRecord]] = []
        for index, execution in enumerate(feedback.executions):
            key = _feedback_record_key(feedback, execution, index, state_key)
            record = ActionLedgerRecord(
                state_key=state_key,
                state_key_ref=state_key_ref(state_key),
                group_id=state_key.subject_ref,
                sender_id=state_key.principal_ref,
                context_id=feedback.request_id,
                response_id=feedback.response_id,
                execution=execution,
                received_at=now,
                dedupe_key=key,
            )
            candidates.append((key, record))
        if not candidates:
            return 0

        with self._lock:
            _ = self._cleanup_expired_locked(now)
            records: list[ActionLedgerRecord] = []
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
        state_key: ConversationStateKey,
        limit: int = 20,
    ) -> list[ActionLedgerRecord]:
        if limit <= 0:
            return []
        with self._lock:
            _ = self._cleanup_expired_locked(self._now())
            matches = [
                record for record in self._records if record.state_key == state_key
            ]
            return list(matches[-limit:])

    def recent_executed_for_conversation(
        self,
        state_key: ConversationStateKey,
        limit: int = 20,
    ) -> list[ActionLedgerRecord]:
        if limit <= 0:
            return []
        with self._lock:
            _ = self._cleanup_expired_locked(self._now())
            matches = [
                record
                for record in self._records
                if record.state_key == state_key
                and record.execution.status == "executed"
            ]
            return list(matches[-limit:])

    def by_context_id(self, context_id: str) -> list[ActionLedgerRecord]:
        with self._lock:
            _ = self._cleanup_expired_locked(self._now())
            return [
                record for record in self._records if record.context_id == context_id
            ]

    def count(self) -> int:
        with self._lock:
            _ = self._cleanup_expired_locked(self._now())
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
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


_DEFAULT_ACTION_LEDGER = ActionLedger()


def get_action_ledger() -> ActionLedger:
    return _DEFAULT_ACTION_LEDGER


def _feedback_record_key(
    feedback: ActionFeedbackRequestV2,
    execution: ActionExecutionFeedbackV2,
    index: int,
    state_key: ConversationStateKey,
) -> tuple[str, ...]:
    artifact = execution.artifact
    return (
        state_key_ref(state_key),
        feedback.feedback_id,
        feedback.request_id,
        feedback.response_id,
        execution.action_id or f"index:{index}",
        execution.action_type,
        execution.status,
        artifact.type if artifact is not None else "",
        getattr(artifact, "option", "") or "",
        getattr(artifact, "period", "") or "",
        getattr(artifact, "report_date", "") or "",
        artifact.artifact_ref
        if artifact is not None and artifact.artifact_ref is not None
        else "",
    )
