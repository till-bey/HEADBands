"""Application configuration with strict startup validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SupportedLanguageSetting = Literal[
    "auto",
    "gsw",
    "de-CH",
    "de-DE",
    "fr-CH",
    "fr-FR",
    "en-US",
    "en-GB",
    "es-ES",
    "es-US",
    "cs-CZ",
    "zh-CN",
    "zh-TW",
]
SwissDialectCode = Literal["ag", "be", "bs", "gr", "lu", "sg", "vs", "zh"]

SWISS_DIALECT_LABELS: dict[str, str] = {
    "ag": "Aargau (Aargauerdütsch)",
    "be": "Bern (Bärndütsch)",
    "bs": "Basel (Baseldytsch)",
    "gr": "Graubünden (Bündnerdütsch)",
    "lu": "Luzern (Lozärnerdütsch)",
    "sg": "St. Gallen (Ostschwiizerdütsch)",
    "vs": "Valais / Leukerbad (Wallisertiitsch)",
    "zh": "Zürich (Züridütsch)",
}


class Settings(BaseSettings):
    """Central validated settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General
    app_name: str = "Swiss Multilingual Voice & Robotics Assistant"
    app_env: Literal["development", "staging", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    port: int = Field(default=8080, ge=1, le=65535)
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]
    )
    serve_frontend: bool = True

    # GCP Regions & Endpoints
    gcp_project_id: str = ""
    gcp_region: str = "europe-west6"
    gcp_location: str = "europe-west6"
    gcp_llm_location: str = "global"
    gcp_robotics_location: str = "global"
    gcp_stt_location: str = "eu"
    gcp_tts_location: str = "eu"

    # STT Routing & Models
    stt_default_provider: Literal["google", "faster_whisper", "mock"] = "google"
    stt_swiss_german_provider: Literal["faster_whisper", "google", "mock"] = "faster_whisper"
    stt_standard_de_provider: Literal["google", "faster_whisper", "mock"] = "google"
    stt_standard_de_language: str = "de-DE"
    stt_standard_de_model: str = "chirp_3"
    stt_fr_provider: Literal["google", "faster_whisper", "mock"] = "google"
    stt_fr_language: str = "fr-FR"
    stt_french_model: str = "chirp_3"
    stt_en_provider: Literal["google", "faster_whisper", "mock"] = "google"
    stt_en_language: str = "en-US"
    stt_english_model: str = "chirp_3"
    stt_es_provider: Literal["google", "faster_whisper", "mock"] = "google"
    stt_es_language: str = "es-ES"
    stt_spanish_model: str = "chirp_3"
    stt_cs_provider: Literal["google", "faster_whisper", "mock"] = "google"
    stt_cs_language: str = "cs-CZ"
    stt_czech_model: str = "chirp_3"
    stt_zh_provider: Literal["google", "faster_whisper", "mock"] = "google"
    stt_zh_language: str = "cmn-Hans-CN"
    stt_chinese_model: str = "chirp_3"
    stt_de_ch_model: str = "long"

    # Swiss German Self-Hosted ASR (CPU-only int8 default for GPU-less GCP sandbox)
    whisper_model: str = "small"
    whisper_device: Literal["cpu", "cuda", "auto"] = "cpu"
    whisper_compute_type: Literal["int8", "int8_float32", "float32", "float16"] = "int8"
    whisper_cpu_threads: int = Field(default=4, ge=1, le=64)
    asr_endpoint: str = ""
    asr_auth_enabled: bool = False
    asr_timeout_seconds: float = Field(default=30.0, gt=0.5, le=120.0)

    # TTS Routing & Models
    tts_default_provider: Literal["google", "piper", "mock"] = "google"
    tts_swiss_german_provider: Literal["piper", "google", "mock"] = "piper"
    tts_speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0)

    piper_models_dir: Path = Path("./models/piper")
    piper_model_path: str = ""
    piper_config_path: str = ""
    piper_default_dialect: SwissDialectCode = "vs"
    piper_model_license: Literal["cc-by-nc-4.0", "commercial", "cc0"] = "cc-by-nc-4.0"
    allow_non_commercial_models: bool = True

    google_tts_language_de: str = "de-DE"
    google_tts_language_fr: str = "fr-FR"
    google_tts_language_en: str = "en-US"
    google_tts_language_es: str = "es-ES"
    google_tts_language_cs: str = "cs-CZ"
    google_tts_language_zh: str = "cmn-CN"
    google_tts_voice_de: str = "de-DE-Neural2-F"
    google_tts_voice_fr: str = "fr-FR-Chirp3-HD-Zephyr"
    google_tts_voice_en: str = "en-US-Chirp3-HD-Aoede"
    google_tts_voice_es: str = "es-ES-Neural2-A"
    google_tts_voice_cs: str = "cs-CZ-Wavenet-A"
    google_tts_voice_zh: str = "cmn-CN-Chirp3-HD-Aoede"

    # Main LLM
    llm_provider: Literal["google", "mock"] = "google"
    gemini_model: str = "gemini-3-flash-preview"

    # Robotics (Gemini Robotics ER 2 + ROS 2 Safety Gateway)
    robotics_enabled: bool = True
    robotics_provider: Literal["gemini_robotics_er", "mock"] = "gemini_robotics_er"
    gemini_robotics_model: str = "gemini-robotics-er-2-preview"
    robot_gateway_url: str = ""
    robot_bridge_shared_token: str = ""
    robot_require_confirmation: bool = True
    robot_command_ttl_seconds: int = Field(default=30, ge=2, le=300)
    robot_watchdog_timeout_seconds: float = Field(default=10.0, gt=0.5, le=120.0)
    robot_max_linear_speed_mps: float = Field(default=0.5, gt=0.0, le=2.0)
    robot_max_angular_speed_rps: float = Field(default=1.0, gt=0.0, le=3.14)
    robot_max_workspace_radius_m: float = Field(default=10.0, gt=0.1, le=20.0)

    # Privacy-first Storage, Database & Pub/Sub
    audio_storage_enabled: bool = False
    audio_retention_days: int = Field(default=1, ge=1, le=365)
    gcs_audio_bucket: str = ""

    firestore_enabled: bool = False
    firestore_database: str = "(default)"
    firestore_collection: str = "conversations"

    pubsub_enabled: bool = False
    pubsub_telemetry_topic: str = "robot-telemetry"

    secret_manager_enabled: bool = False
    api_auth_token: str = ""

    @field_validator("asr_endpoint", "robot_gateway_url")
    @classmethod
    def validate_optional_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if cleaned and not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            raise ValueError(f"Endpoint URL must start with http:// or https://, got: {cleaned}")
        return cleaned

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> Settings:
        # 1. CPU compute-type safety: float16 on CPU fails in CTranslate2 / faster-whisper
        if self.whisper_device == "cpu" and self.whisper_compute_type == "float16":
            raise ValueError(
                "WHISPER_COMPUTE_TYPE='float16' is invalid when WHISPER_DEVICE='cpu'. "
                "Use 'int8' or 'float32' for CPU-only inference."
            )

        # 2. Non-commercial license enforcement (CC BY-NC 4.0 SwissDial Piper models)
        uses_non_commercial_piper = (
            self.tts_swiss_german_provider == "piper"
            and self.piper_model_license == "cc-by-nc-4.0"
        )
        if uses_non_commercial_piper:
            if self.app_env == "production":
                raise ValueError(
                    "d41k4n/piper-swissdial models use CC BY-NC 4.0 (non-commercial) and cannot be "
                    "enabled when APP_ENV='production'. Configure a commercial/CC0 checkpoint and "
                    "set PIPER_MODEL_LICENSE='commercial' or 'cc0', or switch TTS_SWISS_GERMAN_PROVIDER."
                )
            if not self.allow_non_commercial_models:
                raise ValueError(
                    "ALLOW_NON_COMMERCIAL_MODELS=false while PIPER_MODEL_LICENSE='cc-by-nc-4.0'. "
                    "Enable ALLOW_NON_COMMERCIAL_MODELS=true for non-commercial research/sandbox use "
                    "or configure a CC0/commercial voice."
                )

        # 3. Audio storage bucket check
        if self.audio_storage_enabled and not self.gcs_audio_bucket.strip():
            raise ValueError("GCS_AUDIO_BUCKET must be set when AUDIO_STORAGE_ENABLED=true.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached validated Settings singleton."""
    return Settings()
