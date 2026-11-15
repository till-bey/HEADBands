"""Base ASRProvider interface, Transcription model, and MockASRProvider."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Transcription(BaseModel):
    """Structured speech recognition output preserving original Swiss German dialect text."""

    text: str = Field(description="Original verbatim transcription (preserves dialect for gsw).")
    normalized_text: str | None = Field(
        default=None,
        description="Optional Standard German gloss stored separately; never replaces original text.",
    )
    language: str = Field(description="BCP-47 or application language code (gsw, de-DE, fr-FR, etc.).")
    detected_dialect: str | None = Field(
        default=None,
        description="Optional detected Swiss German dialect code (vs, zh, be, bs, sg, etc.).",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provider: str = Field(default="unknown")
    model: str = Field(default="unknown")
    latency_ms: float = Field(default=0.0, ge=0.0)
    is_partial: bool = False


class ASRProvider(ABC):
    """Replaceable interface for Speech-to-Text providers."""

    provider_name: str = "base"

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
        is_partial: bool = False,
    ) -> Transcription:
        """Transcribe 16kHz 16-bit mono PCM or WAV bytes."""
        raise NotImplementedError


class MockASRProvider(ASRProvider):
    """Deterministic offline ASR provider for local/CI tests and credential-free sandboxes."""

    provider_name = "mock"

    CANNED_TRANSCRIPTS: dict[str, tuple[str, str, str | None]] = {
        "gsw": ("Grüezi mitenand, wie isch s Wätter z Leukerbad hüt?", "gsw", "vs"),
        "de-CH": ("Grüezi mitenand, wie ist das Wetter in Leukerbad heute?", "de-CH", None),
        "de-DE": ("Guten Tag, wie kann mir der Assistent heute helfen?", "de-DE", None),
        "fr-FR": ("Bonjour, pouvez-vous vérifier l'état du robot s'il vous plaît ?", "fr-FR", None),
        "fr-CH": ("Bonjour depuis le Valais, comment allez-vous aujourd'hui ?", "fr-CH", None),
        "en-US": ("Hello, can you check the robot status please?", "en-US", None),
        "en-GB": ("Hello, could you check the robot status please?", "en-GB", None),
        "es-ES": ("Hola, ¿puedes comprobar el estado del robot por favor?", "es-ES", None),
        "es-US": ("Hola, ¿puedes revisar el estado del robot por favor?", "es-US", None),
        "cs-CZ": ("Dobrý den, můžete prosím zkontrolovat stav robota?", "cs-CZ", None),
        "zh-CN": ("你好，请帮我检查一下机器人的状态。", "zh-CN", None),
        "zh-TW": ("你好，請幫我檢查一下機器人的狀態。", "zh-TW", None),
    }

    def __init__(self, override_text: str | None = None) -> None:
        self._override_text = override_text

    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
        is_partial: bool = False,
    ) -> Transcription:
        if not audio:
            return Transcription(
                text="",
                language=language if language != "auto" else "de-DE",
                confidence=0.0,
                provider=self.provider_name,
                model="mock-asr-v1",
                latency_ms=1.0,
                is_partial=is_partial,
            )

        selected_key = language if language in self.CANNED_TRANSCRIPTS else "de-DE"
        text, resolved_lang, dialect = self.CANNED_TRANSCRIPTS[selected_key]
        if self._override_text is not None:
            text = self._override_text

        return Transcription(
            text=text,
            language=resolved_lang,
            detected_dialect=dialect,
            confidence=0.98,
            provider=self.provider_name,
            model="mock-asr-v1",
            latency_ms=2.5,
            is_partial=is_partial,
        )
