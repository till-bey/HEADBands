"""Health-check, readiness, and public non-secret configuration endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from app import __version__
from app.core.config import SWISS_DIALECT_LABELS, Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """Liveness probe for Cloud Run and Docker."""
    return {
        "status": "ok",
        "service": "swiss-voice-robotics-backend",
        "version": __version__,
        "environment": settings.app_env,
        "region": settings.gcp_region,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


@router.get("/health/ready")
async def readiness_check(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Readiness probe reporting initialized subsystem states without exposing secrets."""
    services_ready = getattr(request.app.state, "ready", False)
    return {
        "status": "ready" if services_ready else "initializing",
        "providers": {
            "stt_default": settings.stt_default_provider,
            "stt_swiss_german": settings.stt_swiss_german_provider,
            "whisper_device": settings.whisper_device,
            "whisper_compute_type": settings.whisper_compute_type,
            "asr_remote_configured": bool(settings.asr_endpoint),
            "tts_default": settings.tts_default_provider,
            "tts_swiss_german": settings.tts_swiss_german_provider,
            "llm": settings.llm_provider,
            "robotics_enabled": settings.robotics_enabled,
            "robotics_provider": settings.robotics_provider,
        },
        "privacy": {
            "audio_storage_enabled": settings.audio_storage_enabled,
            "firestore_enabled": settings.firestore_enabled,
        },
    }


@router.get("/api/v1/config/public")
async def public_capabilities(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Expose safe, non-secret runtime metadata & residency disclosures to the Web UI."""
    return {
        "app_name": settings.app_name,
        "version": __version__,
        "supported_languages": [
            {"code": "auto", "label": "Auto-Detect (CH-DE / DE / FR / EN / ES / CS / 中文)"},
            {"code": "gsw", "label": "Schwiizerdütsch (Swiss German)"},
            {"code": "de-DE", "label": "Hochdeutsch (Standard German)"},
            {"code": "de-CH", "label": "Schweizer Hochdeutsch (de-CH)"},
            {"code": "fr-FR", "label": "Français (French)"},
            {"code": "fr-CH", "label": "Français Suisse (fr-CH)"},
            {"code": "en-US", "label": "English (en-US)"},
            {"code": "en-GB", "label": "English UK (en-GB)"},
            {"code": "es-ES", "label": "Español — España (es-ES)"},
            {"code": "es-US", "label": "Español — Latinoamérica (es-US)"},
            {"code": "cs-CZ", "label": "Čeština — Czech (cs-CZ)"},
            {"code": "zh-CN", "label": "中文 — Mandarin Simplified (zh-CN)"},
            {"code": "zh-TW", "label": "繁體中文 — Mandarin Traditional (zh-TW)"},
        ],
        "swiss_dialects": [
            {"code": code, "label": label} for code, label in SWISS_DIALECT_LABELS.items()
        ],
        "default_dialect": settings.piper_default_dialect,
        "audio_format": {
            "encoding": "PCM_S16LE",
            "sample_rate_hz": 16000,
            "channels": 1,
            "bits_per_sample": 16,
        },
        "robotics": {
            "enabled": settings.robotics_enabled,
            "provider": settings.robotics_provider,
            "model": settings.gemini_robotics_model,
            "requires_confirmation": settings.robot_require_confirmation,
            "max_linear_speed_mps": settings.robot_max_linear_speed_mps,
            "max_workspace_radius_m": settings.robot_max_workspace_radius_m,
        },
        "residency_disclosures": {
            "hosting_region": settings.gcp_region,
            "stt_endpoint_region": settings.gcp_stt_location,
            "tts_endpoint_region": settings.gcp_tts_location,
            "llm_endpoint_region": settings.gcp_llm_location,
            "robotics_er2_endpoint": settings.gcp_robotics_location,
            "swiss_only_guaranteed_for_all_apis": False,
            "piper_swissdial_license": settings.piper_model_license,
        },
    }
