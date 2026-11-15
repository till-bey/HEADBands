"""Optional Cloud Storage audio adapter with privacy-by-default and retention metadata."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioStorageService:
    """Stores audio blobs ONLY when AUDIO_STORAGE_ENABLED=true, with explicit retention expiry."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    @property
    def enabled(self) -> bool:
        return self._settings.audio_storage_enabled and bool(self._settings.gcs_audio_bucket)

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self._settings.gcp_project_id or None)
        return self._client

    async def save_audio_if_enabled(
        self,
        audio_bytes: bytes,
        session_id: str,
        language: str,
        content_type: str = "audio/wav",
    ) -> str | None:
        """Persist audio only when explicitly enabled; otherwise discard immediately for privacy."""
        if not self.enabled:
            return None

        now = datetime.now(tz=UTC)
        expires_at = now + timedelta(days=self._settings.audio_retention_days)
        object_name = f"sessions/{now:%Y/%m/%d}/{session_id}/{uuid4().hex}.wav"

        def _upload() -> str:
            client = self._get_client()
            bucket = client.bucket(self._settings.gcs_audio_bucket)
            blob = bucket.blob(object_name)
            blob.metadata = {
                "session_id": session_id,
                "language": language,
                "retention_days": str(self._settings.audio_retention_days),
                "delete_after_utc": expires_at.isoformat(),
            }
            blob.custom_time = now
            blob.upload_from_string(audio_bytes, content_type=content_type)
            return f"gs://{self._settings.gcs_audio_bucket}/{object_name}"

        uri = await asyncio.to_thread(_upload)
        logger.info(
            "Stored optional session audio with retention policy",
            extra={
                "session_id": session_id,
                "gcs_uri": uri,
                "retention_days": self._settings.audio_retention_days,
                "audio_num_bytes": len(audio_bytes),
            },
        )
        return uri

    async def delete_session_recordings(self, session_id: str) -> int:
        """Delete any stored audio objects matching a session_id."""
        if not self.enabled:
            return 0

        def _delete() -> int:
            client = self._get_client()
            bucket = client.bucket(self._settings.gcs_audio_bucket)
            deleted = 0
            for blob in bucket.list_blobs(prefix="sessions/"):
                if f"/{session_id}/" in blob.name:
                    blob.delete()
                    deleted += 1
            return deleted

        return await asyncio.to_thread(_delete)
