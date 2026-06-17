from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Literal

from market_support_crewai_agent.runtime.domain.sources.metadata import (
    SourceMetadata,
    source_metadata_for_conversation_message,
)
from market_support_crewai_agent.settings import Settings


ConversationRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class ConversationMessage:
    role: ConversationRole
    content: str
    created_at: datetime
    source_metadata: SourceMetadata | None = None


@dataclass
class _ConversationSession:
    last_active_at: datetime
    expires_at: datetime
    messages: list[ConversationMessage]


class ConversationStore:
    """Thread-safe, in-memory conversation history with TTL and size bounds."""

    def __init__(
        self,
        ttl_seconds: int = 86400,
        max_messages: int = 12,
        max_sessions: int = 5000,
        cleanup_interval_seconds: int = 300,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be greater than zero")
        if cleanup_interval_seconds < 0:
            raise ValueError("cleanup_interval_seconds cannot be negative")

        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_messages = max_messages
        self._max_sessions = max_sessions
        self._cleanup_interval = timedelta(seconds=cleanup_interval_seconds)
        self._now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self._sessions: dict[str, _ConversationSession] = {}
        self._lock = RLock()
        self._next_cleanup_at = self._now() + self._cleanup_interval

    @classmethod
    def from_settings(cls, settings: Settings) -> ConversationStore:
        return cls(
            ttl_seconds=settings.agent_conversation_ttl_seconds,
            max_messages=settings.agent_conversation_max_messages,
            max_sessions=settings.agent_conversation_max_sessions,
            cleanup_interval_seconds=(
                settings.agent_conversation_cleanup_interval_seconds
            ),
        )

    def get_recent(self, conversation_key: str) -> list[ConversationMessage]:
        """Return a copy of recent messages for an active conversation."""
        with self._lock:
            now = self._now()
            self._cleanup_if_due_locked(now)
            session = self._sessions.get(conversation_key)
            if session is None:
                return []
            if session.expires_at <= now:
                del self._sessions[conversation_key]
                return []
            return list(session.messages)

    def save_turn(
        self,
        conversation_key: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Persist a completed user/assistant exchange."""
        with self._lock:
            now = self._now()
            self._cleanup_if_due_locked(now)
            session = self._sessions.get(conversation_key)
            if session is None or session.expires_at <= now:
                session = _ConversationSession(
                    last_active_at=now,
                    expires_at=now + self._ttl,
                    messages=[],
                )
                self._sessions[conversation_key] = session

            session.messages.extend(
                [
                    ConversationMessage(
                        "user",
                        user_content,
                        now,
                        source_metadata_for_conversation_message(
                            conversation_key=conversation_key,
                            role="user",
                            created_at=now,
                        ),
                    ),
                    ConversationMessage(
                        "assistant",
                        assistant_content,
                        now,
                        source_metadata_for_conversation_message(
                            conversation_key=conversation_key,
                            role="assistant",
                            created_at=now,
                        ),
                    ),
                ]
            )
            if len(session.messages) > self._max_messages:
                session.messages = session.messages[-self._max_messages :]
            session.last_active_at = now
            session.expires_at = now + self._ttl
            self._enforce_session_cap_locked(preserve_key=conversation_key)

    def cleanup_expired(self) -> int:
        """Remove expired sessions immediately and return the deletion count."""
        with self._lock:
            now = self._now()
            removed = self._cleanup_expired_locked(now)
            self._next_cleanup_at = now + self._cleanup_interval
            return removed

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _cleanup_if_due_locked(self, now: datetime) -> None:
        if now >= self._next_cleanup_at:
            self._cleanup_expired_locked(now)
            self._next_cleanup_at = now + self._cleanup_interval

    def _cleanup_expired_locked(self, now: datetime) -> int:
        expired_keys = [
            key for key, session in self._sessions.items() if session.expires_at <= now
        ]
        for key in expired_keys:
            del self._sessions[key]
        return len(expired_keys)

    def _enforce_session_cap_locked(self, preserve_key: str) -> None:
        while len(self._sessions) > self._max_sessions:
            candidates = (
                (key, session)
                for key, session in self._sessions.items()
                if key != preserve_key
            )
            oldest = min(
                candidates,
                key=lambda item: (item[1].last_active_at, item[0]),
                default=None,
            )
            if oldest is None:
                break
            del self._sessions[oldest[0]]

    def _now(self) -> datetime:
        value = self._now_factory()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
