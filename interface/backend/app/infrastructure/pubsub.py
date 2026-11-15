"""Optional Google Cloud Pub/Sub adapter for asynchronous robot telemetry and events."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TelemetryPubSubPublisher:
    """Publishes robot telemetry and status events to Cloud Pub/Sub when enabled."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._publisher = None
        self._topic_path: str | None = None

    @property
    def enabled(self) -> bool:
        return (
            self._settings.pubsub_enabled
            and bool(self._settings.gcp_project_id)
            and bool(self._settings.pubsub_telemetry_topic)
        )

    def _ensure_publisher(self) -> tuple[Any, str]:
        if self._publisher is None:
            from google.cloud import pubsub_v1

            self._publisher = pubsub_v1.PublisherClient()
            self._topic_path = self._publisher.topic_path(
                self._settings.gcp_project_id,
                self._settings.pubsub_telemetry_topic,
            )
        assert self._topic_path is not None
        return self._publisher, self._topic_path

    async def publish_event(self, event_type: str, payload: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None

        envelope = {
            "event_type": event_type,
            "published_at": datetime.now(tz=UTC).isoformat(),
            "payload": payload,
        }
        encoded = json.dumps(envelope, default=str).encode("utf-8")

        def _publish() -> str:
            publisher, topic_path = self._ensure_publisher()
            future = publisher.publish(topic_path, encoded, event_type=event_type)
            return str(future.result(timeout=5.0))

        try:
            return await asyncio.to_thread(_publish)
        except Exception as exc:
            logger.warning(
                "Pub/Sub publish failed; continuing locally",
                extra={"event_type": event_type, "error": str(exc)},
            )
            return None
