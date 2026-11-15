"""Unit tests for Phase 1: Config validation, structured logging redaction, security, and health."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.core.logging import CloudLoggingJSONFormatter, sanitize_log_payload
from app.core.security import verify_constant_time_token
from app.main import create_app


def test_default_cpu_sandbox_settings_valid() -> None:
    settings = Settings(
        app_env="development",
        whisper_device="cpu",
        whisper_compute_type="int8",
        stt_default_provider="mock",
        stt_swiss_german_provider="mock",
        tts_default_provider="mock",
        tts_swiss_german_provider="mock",
        llm_provider="mock",
        robotics_provider="mock",
    )
    assert settings.gcp_region == "europe-west6"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"
    assert settings.audio_storage_enabled is False


def test_cpu_float16_is_rejected_at_startup() -> None:
    with pytest.raises(ValidationError, match="float16"):
        Settings(whisper_device="cpu", whisper_compute_type="float16")


def test_non_commercial_swissdial_blocked_in_production() -> None:
    with pytest.raises(ValidationError, match="CC BY-NC 4.0"):
        Settings(
            app_env="production",
            tts_swiss_german_provider="piper",
            piper_model_license="cc-by-nc-4.0",
            allow_non_commercial_models=True,
        )


def test_audio_storage_requires_bucket() -> None:
    with pytest.raises(ValidationError, match="GCS_AUDIO_BUCKET"):
        Settings(audio_storage_enabled=True, gcs_audio_bucket="")


def test_log_redaction_strips_raw_audio_and_secrets() -> None:
    sanitized = sanitize_log_payload(
        {
            "session_id": "s-123",
            "raw_audio_pcm": b"\x00\x01\x02\x03" * 100,
            "api_key": "AIzaSySecretKey",
            "robot_token": "super-secret",
            "audio_duration_ms": 1450,
            "nested": {"payload": b"binary-blob"},
        }
    )
    assert sanitized["session_id"] == "s-123"
    assert "REDACTED" in sanitized["raw_audio_pcm"]
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["robot_token"] == "[REDACTED]"
    assert sanitized["audio_duration_ms"] == 1450
    assert "BINARY_REDACTED" in sanitized["nested"]["payload"]


def test_cloud_logging_formatter_emits_valid_json() -> None:
    formatter = CloudLoggingJSONFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="Voice session opened",
        args=(),
        exc_info=None,
    )
    record.audio_bytes = b"\x01\x02\x03"
    record.language = "gsw"
    output = json.loads(formatter.format(record))
    assert output["severity"] == "INFO"
    assert output["message"] == "Voice session opened"
    assert output["context"]["language"] == "gsw"
    assert "REDACTED" in output["context"]["audio_bytes"]


def test_constant_time_token_verification() -> None:
    assert verify_constant_time_token("Bearer secret-123", "secret-123") is True
    assert verify_constant_time_token("wrong", "secret-123") is False
    assert verify_constant_time_token(None, "secret-123") is False
    assert verify_constant_time_token(None, "") is True


def test_health_and_public_config_endpoints() -> None:
    settings = Settings(
        app_env="test",
        stt_default_provider="mock",
        stt_swiss_german_provider="mock",
        tts_default_provider="mock",
        tts_swiss_german_provider="mock",
        llm_provider="mock",
        robotics_provider="mock",
        serve_frontend=False,
    )
    app = create_app(settings_override=settings)
    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"
        assert r_health.json()["region"] == "europe-west6"

        r_ready = client.get("/health/ready")
        assert r_ready.status_code == 200
        assert r_ready.json()["status"] == "ready"
        assert r_ready.json()["providers"]["whisper_device"] == "cpu"

        r_pub = client.get("/api/v1/config/public")
        assert r_pub.status_code == 200
        data = r_pub.json()
        assert data["default_dialect"] == "vs"
        assert data["residency_disclosures"]["swiss_only_guaranteed_for_all_apis"] is False
        assert any(d["code"] == "vs" for d in data["swiss_dialects"])
