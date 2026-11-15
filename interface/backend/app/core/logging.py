"""Structured Cloud Logging setup with strict audio & secret redaction."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

REDACTED_VALUE = "[REDACTED]"
FORBIDDEN_LOG_SUBSTRINGS = (
    "audio",
    "pcm",
    "wav_bytes",
    "raw_bytes",
    "secret",
    "token",
    "api_key",
    "apikey",
    "password",
    "authorization",
    "private_key",
)


def sanitize_log_payload(value: Any, key_name: str = "") -> Any:
    """Recursively sanitize structured log fields so raw audio and secrets never hit logs."""
    lowered_key = key_name.lower()
    if lowered_key and any(marker in lowered_key for marker in FORBIDDEN_LOG_SUBSTRINGS):
        if isinstance(value, (bytes, bytearray)):
            return f"{REDACTED_VALUE} ({len(value)} bytes)"
        if lowered_key in {"audio_storage_enabled", "audio_duration_ms", "audio_num_bytes", "audio_sample_rate"}:
            return value
        return REDACTED_VALUE

    if isinstance(value, (bytes, bytearray)):
        return f"[BINARY_REDACTED:{len(value)}_bytes]"

    if isinstance(value, dict):
        return {str(k): sanitize_log_payload(v, str(k)) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [sanitize_log_payload(item, key_name) for item in value]

    return value


class CloudLoggingJSONFormatter(logging.Formatter):
    """Formats Python LogRecords into Google Cloud Logging structured JSON."""

    _STANDARD_ATTRS = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras: dict[str, Any] = {}
        for key, val in record.__dict__.items():
            if key not in self._STANDARD_ATTRS and not key.startswith("_"):
                extras[key] = sanitize_log_payload(val, key)

        if extras:
            payload["context"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with structured Cloud Logging JSON output."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingJSONFormatter())

    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
