"""Session conversation memory with active turn interruption tokens and optional Firestore sync."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.infrastructure.firestore import FirestoreRepository


class TurnMessage(BaseModel):
    """Single conversation turn entry."""

    role: Literal["user", "assistant", "tool"]
    content: str
    normalized_content: str | None = None
    language: str = "de-DE"
    dialect: str | None = None
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ConversationSession(BaseModel):
    """Stateful session tracking history, language, and Swiss German dialect preference."""

    session_id: str
    preferred_language: str = "auto"
    active_language: str = "de-DE"
    active_dialect: str = "vs"
    messages: list[TurnMessage] = Field(default_factory=list)
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at_utc: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class SessionMemoryStore:
    """Manages active sessions in memory (with optional Firestore backing) and turn cancellation."""

    def __init__(
        self,
        firestore_repo: FirestoreRepository,
        default_dialect: str = "vs",
        max_turns_per_session: int = 40,
    ) -> None:
        self._firestore = firestore_repo
        self._default_dialect = default_dialect
        self._max_turns = max_turns_per_session
        self._sessions: dict[str, ConversationSession] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}

    async def get_or_create(self, session_id: str) -> ConversationSession:
        if session_id in self._sessions:
            return self._sessions[session_id]

        loaded = await self._firestore.load_session(session_id)
        if loaded is not None:
            session = ConversationSession.model_validate(loaded)
        else:
            session = ConversationSession(
                session_id=session_id,
                active_dialect=self._default_dialect,
            )
        self._sessions[session_id] = session
        return session

    async def append_message(self, session_id: str, message: TurnMessage) -> ConversationSession:
        session = await self.get_or_create(session_id)
        session.messages.append(message)
        if len(session.messages) > self._max_turns:
            session.messages = session.messages[-self._max_turns :]
        session.updated_at_utc = datetime.now(tz=UTC)
        await self._firestore.save_session(session_id, session.model_dump(mode="json"))
        return session

    async def update_language_state(
        self,
        session_id: str,
        active_language: str,
        active_dialect: str | None = None,
        preferred_language: str | None = None,
    ) -> ConversationSession:
        session = await self.get_or_create(session_id)
        session.active_language = active_language
        if active_dialect:
            session.active_dialect = active_dialect
        if preferred_language is not None:
            session.preferred_language = preferred_language
        session.updated_at_utc = datetime.now(tz=UTC)
        await self._firestore.save_session(session_id, session.model_dump(mode="json"))
        return session

    async def delete_session(self, session_id: str) -> None:
        self.interrupt_turn(session_id)
        self._sessions.pop(session_id, None)
        self._cancel_events.pop(session_id, None)
        await self._firestore.delete_session(session_id)

    def start_new_turn_cancel_token(self, session_id: str) -> asyncio.Event:
        """Cancel any in-flight generation for this session and create a fresh cancel Event."""
        previous = self._cancel_events.get(session_id)
        if previous is not None:
            previous.set()
        fresh = asyncio.Event()
        self._cancel_events[session_id] = fresh
        return fresh

    def interrupt_turn(self, session_id: str) -> bool:
        """Signal immediate cancellation of any pending LLM or TTS work on this session."""
        ev = self._cancel_events.get(session_id)
        if ev is not None and not ev.is_set():
            ev.set()
            return True
        return False
