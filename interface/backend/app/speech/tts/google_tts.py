"""Google Cloud Text-to-Speech provider (EU endpoint, Chirp 3 HD & Neural2 voices)."""

from __future__ import annotations

import asyncio

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError, InvalidArgument
from google.cloud import texttospeech

from app.core.config import Settings
from app.core.logging import get_logger
from app.speech.tts.base import TTSProvider

logger = get_logger(__name__)

FALLBACK_NEURAL2_VOICES: dict[str, str] = {
    "de-DE": "de-DE-Neural2-B",
    "fr-FR": "fr-FR-Neural2-A",
    "en-US": "en-US-Neural2-F",
    "en-GB": "en-GB-Neural2-A",
    "es-ES": "es-ES-Neural2-A",
    "es-US": "es-US-Neural2-A",
    "cs-CZ": "cs-CZ-Wavenet-A",
    "cmn-CN": "cmn-CN-Wavenet-A",
    "cmn-TW": "cmn-TW-Wavenet-A",
}


class GoogleTTSProvider(TTSProvider):
    """Google Cloud Text-to-Speech adapter with EU regional endpoint and Neural2 fallback."""

    provider_name = "google"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: texttospeech.TextToSpeechAsyncClient | None = None
        if settings.app_env != "test" and settings.gcp_project_id:
            try:
                self._get_client()
            except Exception:
                self._client = None

    def _get_client(self) -> texttospeech.TextToSpeechAsyncClient:
        if self._client is None:
            loc = self._settings.gcp_tts_location.strip()
            options = (
                ClientOptions(api_endpoint=f"{loc}-texttospeech.googleapis.com")
                if loc and loc != "global"
                else None
            )
            self._client = texttospeech.TextToSpeechAsyncClient(client_options=options)
        return self._client

    def _resolve_locale_and_voice(
        self,
        language: str,
        requested_voice: str | None,
    ) -> tuple[str, str]:
        if language.startswith(("zh", "cmn")):
            if any(tag in language.lower() for tag in ("tw", "hk", "hant")):
                return "cmn-TW", (requested_voice or FALLBACK_NEURAL2_VOICES["cmn-TW"])
            locale = self._settings.google_tts_language_zh
            default_voice = self._settings.google_tts_voice_zh or FALLBACK_NEURAL2_VOICES["cmn-CN"]
            return locale, (requested_voice or default_voice)

        if language.startswith("en"):
            if language.lower() == "en-gb":
                return "en-GB", (requested_voice or "en-GB-Chirp3-HD-Aoede")
            locale = self._settings.google_tts_language_en
            default_voice = self._settings.google_tts_voice_en or FALLBACK_NEURAL2_VOICES["en-US"]
            return locale, (requested_voice or default_voice)

        if language.startswith("es"):
            if any(tag in language.lower() for tag in ("us", "mx", "419")):
                return "es-US", (requested_voice or FALLBACK_NEURAL2_VOICES["es-US"])
            locale = self._settings.google_tts_language_es
            default_voice = self._settings.google_tts_voice_es or FALLBACK_NEURAL2_VOICES["es-ES"]
            return locale, (requested_voice or default_voice)

        if language.startswith(("cs", "cz")):
            locale = self._settings.google_tts_language_cs
            default_voice = self._settings.google_tts_voice_cs or FALLBACK_NEURAL2_VOICES["cs-CZ"]
            return locale, (requested_voice or default_voice)

        if language.startswith("fr"):
            locale = self._settings.google_tts_language_fr
            default_voice = self._settings.google_tts_voice_fr or FALLBACK_NEURAL2_VOICES["fr-FR"]
            return locale, (requested_voice or default_voice)

        # Standard German (and Google fallback when Swiss German Piper is disabled)
        locale = self._settings.google_tts_language_de
        default_voice = self._settings.google_tts_voice_de or FALLBACK_NEURAL2_VOICES["de-DE"]
        return locale, (requested_voice or default_voice)

    async def _call_tts(
        self,
        text: str,
        locale: str,
        voice_name: str,
        speaking_rate: float,
    ) -> bytes:
        client = self._get_client()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=locale,
            name=voice_name,
        )

        # Chirp 3 HD voices may reject non-1.0 speaking_rate on certain regional releases
        is_chirp_hd = "Chirp3-HD" in voice_name or "Chirp-HD" in voice_name
        audio_kwargs: dict[str, object] = {
            "audio_encoding": texttospeech.AudioEncoding.LINEAR16,
            "sample_rate_hertz": 24000 if is_chirp_hd else 16000,
        }
        if not is_chirp_hd and abs(speaking_rate - 1.0) > 1e-3:
            audio_kwargs["speaking_rate"] = speaking_rate

        audio_config = texttospeech.AudioConfig(**audio_kwargs)
        response = await client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
        return bytes(response.audio_content)

    async def synthesize(
        self,
        text: str,
        language: str,
        voice: str | None = None,
        dialect: str | None = None,
        speaking_rate: float = 1.0,
        cancel_event: asyncio.Event | None = None,
    ) -> bytes:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("Google TTS cancelled before request")

        cleaned = text.strip()
        if not cleaned:
            return b""

        rate = speaking_rate or self._settings.tts_speaking_rate
        locale, voice_name = self._resolve_locale_and_voice(language, voice)

        try:
            return await self._call_tts(cleaned, locale, voice_name, rate)
        except (InvalidArgument, GoogleAPICallError) as exc:
            fallback_voice = FALLBACK_NEURAL2_VOICES.get(locale)
            if fallback_voice and fallback_voice != voice_name:
                logger.warning(
                    "Primary Google TTS voice unavailable; retrying with Neural2 fallback",
                    extra={"primary_voice": voice_name, "fallback_voice": fallback_voice, "reason": str(exc)},
                )
                return await self._call_tts(cleaned, locale, fallback_voice, rate)
            raise
