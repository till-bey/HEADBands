"""Optional Firestore persistence for session conversations and configuration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class FirestoreRepository:
    """Async Firestore adapter for conversation state in europe-west6."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    @property
    def enabled(self) -> bool:
        return self._settings.firestore_enabled and bool(self._settings.gcp_project_id)

    def _get_client(self):
        if self._client is None:
            from google.cloud import firestore

            self._client = firestore.AsyncClient(
                project=self._settings.gcp_project_id,
                database=self._settings.firestore_database,
            )
        return self._client

    async def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            client = self._get_client()
            doc_ref = client.collection(self._settings.firestore_collection).document(session_id)
            payload = {
                **data,
                "updated_at": datetime.now(tz=UTC).isoformat(),
            }
            await doc_ref.set(payload, merge=True)
        except Exception as exc:
            logger.warning(
                "Firestore save_session failed; keeping in-memory session",
                extra={"session_id": session_id, "error": str(exc)},
            )

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            client = self._get_client()
            doc = await client.collection(self._settings.firestore_collection).document(session_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as exc:
            logger.warning(
                "Firestore load_session failed",
                extra={"session_id": session_id, "error": str(exc)},
            )
            return None

    async def delete_session(self, session_id: str) -> None:
        if not self.enabled:
            return
        try:
            client = self._get_client()
            await client.collection(self._settings.firestore_collection).document(session_id).delete()
        except Exception as exc:
            logger.warning(
                "Firestore delete_session failed",
                extra={"session_id": session_id, "error": str(exc)},
            )
