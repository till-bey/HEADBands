"""Language-aware ASR and TTS provider router with acoustic Swiss German disambiguation in auto mode."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import Settings
from app.core.logging import get_logger
from app.speech.asr.base import ASRProvider, Transcription
from app.speech.audio_utils import DEFAULT_SAMPLE_RATE, is_wav_bytes, pcm16_to_wav
from app.speech.language.detection import (
    detect_language_and_dialect_from_text,
    detect_swiss_dialect,
    tokenize_words,
)
from app.speech.language.normalization import build_swiss_german_gloss
from app.speech.tts.base import TTSProvider
from app.speech.tts.piper import resolve_swissdial_voice_key

logger = get_logger(__name__)


class _GermanicAudioCheck(BaseModel):
    language: str
    verbatim_transcript: str


_GERMANIC_DISAMBIGUATION_PROMPT = (
    "Listen carefully to the spoken audio. Classify whether the speaker is speaking genuine "
    "Swiss German dialect ('gsw' — Schwiizerdütsch, e.g. Wallisertiitsch, Bärndütsch, Züridütsch, "
    "with dialect grammar/words such as 'isch', 'nöd/nid', 'hüt', 'gang', 'füre', 'dräi', 'wier/mier') "
    "OR Standard High German ('de-DE' — Hochdeutsch, even if spoken with a Swiss accent, e.g. 'ich', "
    "'nicht', 'geh/gehen', 'fahr/fahren', 'vorwärts', 'dreh dich um', 'zurück'). "
    "If the grammar and words are Standard High German, you MUST return language='de-DE'. "
    "Return JSON with language ('gsw' or 'de-DE') and verbatim_transcript."
)


@dataclass
class TTSRoute:
    """Selected TTS provider, target language, voice, and optional dialect."""

    provider: TTSProvider
    language: str
    voice: str | None
    dialect: str | None


class SpeechRouter:
    """Routes speech recognition and synthesis dynamically across Swiss German, German, French, English, and Chinese."""

    def __init__(
        self,
        settings: Settings,
        asr_providers: dict[str, ASRProvider],
        tts_providers: dict[str, TTSProvider],
    ) -> None:
        self._settings = settings
        self._asr = asr_providers
        self._tts = tts_providers
        self._genai_client: genai.Client | None = None
        if settings.app_env != "test" and settings.llm_provider == "google" and settings.gcp_project_id:
            try:
                self._get_genai_client()
            except Exception:
                self._genai_client = None

    def _get_genai_client(self) -> genai.Client | None:
        if self._settings.llm_provider != "google" or not self._settings.gcp_project_id:
            return None
        if self._genai_client is None:
            self._genai_client = genai.Client(
                vertexai=True,
                project=self._settings.gcp_project_id,
                location=self._settings.gcp_llm_location,
            )
        return self._genai_client

    def select_asr_provider(self, language: str, session_hint_language: str | None = None) -> ASRProvider:
        # When language=='auto', always route to the fast multilingual auto-detecting default ASR
        # (Chirp 3 'auto' across EN / DE / FR / GSW, with acoustic GSW disambiguation & Whisper fallback)
        # so the user can switch freely turn-by-turn without ever touching a language selector.
        if language in {"auto", ""}:
            provider_key = self._settings.stt_default_provider
        elif language == "gsw":
            provider_key = self._settings.stt_swiss_german_provider
        elif language in {"fr-FR", "fr-CH"} or language.startswith("fr"):
            provider_key = self._settings.stt_fr_provider
        elif language in {"de-DE", "de-CH"} or language.startswith("de"):
            provider_key = self._settings.stt_standard_de_provider
        elif language in {"en-US", "en-GB"} or language.startswith("en"):
            provider_key = self._settings.stt_en_provider
        elif language in {"es-ES", "es-US"} or language.startswith("es"):
            provider_key = self._settings.stt_es_provider
        elif language == "cs-CZ" or language.startswith(("cs", "cz")):
            provider_key = self._settings.stt_cs_provider
        elif language in {"zh-CN", "zh-TW"} or language.startswith(("zh", "cmn")):
            provider_key = self._settings.stt_zh_provider
        else:
            provider_key = self._settings.stt_default_provider

        provider = self._asr.get(provider_key)
        if provider is None:
            raise KeyError(f"Configured ASR provider '{provider_key}' is not registered.")
        return provider

    async def _disambiguate_germanic_audio(
        self,
        audio: bytes,
        primary_result: Transcription,
    ) -> Transcription:
        """Distinguish spoken Swiss German (gsw) from Standard German (de-DE) acoustically in auto mode."""
        client = self._get_genai_client()
        if client is None:
            return primary_result

        wav_bytes = audio if is_wav_bytes(audio) else pcm16_to_wav(audio, DEFAULT_SAMPLE_RATE)
        thinking_cfg: types.ThinkingConfig | None = None
        if "gemini-3" in self._settings.gemini_model:
            thinking_cfg = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)

        try:
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self._settings.gemini_model,
                    contents=[
                        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                        _GERMANIC_DISAMBIGUATION_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=_GermanicAudioCheck,
                        temperature=0.0,
                        max_output_tokens=256,
                        thinking_config=thinking_cfg,
                    ),
                ),
                timeout=1.35,
            )
            parsed = (
                resp.parsed
                if isinstance(resp.parsed, _GermanicAudioCheck)
                else _GermanicAudioCheck.model_validate_json(resp.text or "{}")
            )
            if parsed.language.strip().lower() == "gsw" and parsed.verbatim_transcript.strip():
                gsw_text = parsed.verbatim_transcript.strip()
                det_lang, det_dialect = detect_language_and_dialect_from_text(
                    gsw_text,
                    hint_language="auto",
                    default_dialect=self._settings.piper_default_dialect,
                )
                # Guard against accented Hochdeutsch being mislabeled as 'gsw'
                if det_lang == "gsw":
                    dialect = det_dialect or detect_swiss_dialect(
                        tokenize_words(gsw_text),
                        fallback_dialect=self._settings.piper_default_dialect,
                    )
                    return Transcription(
                        text=gsw_text,
                        normalized_text=build_swiss_german_gloss(gsw_text),
                        language="gsw",
                        detected_dialect=dialect or self._settings.piper_default_dialect,
                        confidence=primary_result.confidence,
                        provider=primary_result.provider,
                        model=f"{primary_result.model}+gsw-acoustic",
                        latency_ms=primary_result.latency_ms,
                        is_partial=primary_result.is_partial,
                    )
        except Exception as exc:
            logger.warning(
                "Acoustic Swiss German disambiguation skipped; keeping primary STT result",
                extra={"error": str(exc)},
            )
        return primary_result

    async def transcribe_routed(
        self,
        audio: bytes,
        requested_language: str = "auto",
        session_language_hint: str | None = None,
        is_partial: bool = False,
    ) -> Transcription:
        """Transcribe audio and support seamless EN / FR / DE / Swiss German auto-detection."""
        primary_asr = self.select_asr_provider(requested_language, session_language_hint)
        result = await primary_asr.transcribe(audio, language=requested_language, is_partial=is_partial)

        # Fallback to self-hosted Whisper if primary Google STT returned empty/low confidence in auto mode
        if (
            not is_partial
            and requested_language in {"auto", ""}
            and primary_asr.provider_name == "google"
            and self._settings.stt_swiss_german_provider in self._asr
            and self._settings.stt_swiss_german_provider != "google"
            and (not result.text or result.confidence < 0.55)
        ):
            gsw_asr = self._asr[self._settings.stt_swiss_german_provider]
            try:
                alt = await gsw_asr.transcribe(audio, language="auto", is_partial=False)
                if alt.text and alt.confidence >= result.confidence:
                    result = alt
            except Exception as exc:
                logger.warning("Secondary Swiss German ASR probe failed; keeping primary result", extra={"error": str(exc)})

        # In auto mode, Chirp 3 normalizes Swiss German speech into Hochdeutsch text ('de-DE').
        # Skip acoustic disambiguation when session is already in Standard German or explicit 'ß'/'Hochdeutsch' is present.
        if (
            not is_partial
            and requested_language in {"auto", ""}
            and session_language_hint not in {"de-DE", "de-CH"}
            and result.language == "de-DE"
            and result.text
            and "ß" not in result.text
            and "hochdeutsch" not in result.text.lower()
        ):
            result = await self._disambiguate_germanic_audio(audio, result)

        # Enrich dialect detection if Swiss German
        if result.language == "gsw" and not result.detected_dialect:
            _, dialect = detect_language_and_dialect_from_text(
                result.text,
                hint_language="gsw",
                default_dialect=self._settings.piper_default_dialect,
            )
            result.detected_dialect = dialect or self._settings.piper_default_dialect

        return result

    def select_tts_route(
        self,
        language: str,
        dialect: str | None = None,
        requested_voice: str | None = None,
    ) -> TTSRoute:
        """Choose the appropriate TTS provider & dialect voice so Swiss German never silently becomes Hochdeutsch."""
        if language == "gsw":
            provider_key = self._settings.tts_swiss_german_provider
            provider = self._tts.get(provider_key)
            if provider is None:
                raise KeyError(f"Configured Swiss German TTS provider '{provider_key}' is not registered.")
            resolved_dialect, voice_key = resolve_swissdial_voice_key(
                requested_voice or dialect,
                default_dialect=self._settings.piper_default_dialect,
            )
            return TTSRoute(
                provider=provider,
                language="gsw",
                voice=requested_voice or voice_key,
                dialect=resolved_dialect,
            )

        provider_key = self._settings.tts_default_provider
        provider = self._tts.get(provider_key)
        if provider is None:
            raise KeyError(f"Configured default TTS provider '{provider_key}' is not registered.")

        if language.startswith(("zh", "cmn")):
            is_trad = any(tag in language.lower() for tag in ("tw", "hk", "hant"))
            default_zh_voice = "cmn-TW-Wavenet-A" if is_trad else self._settings.google_tts_voice_zh
            voice = requested_voice or default_zh_voice or None
            return TTSRoute(provider=provider, language=language, voice=voice, dialect=None)

        if language.startswith("en"):
            default_en_voice = "en-GB-Chirp3-HD-Aoede" if language == "en-GB" else self._settings.google_tts_voice_en
            voice = requested_voice or default_en_voice or None
            return TTSRoute(provider=provider, language=language, voice=voice, dialect=None)

        if language.startswith("es"):
            is_latam = any(tag in language.lower() for tag in ("us", "mx", "419"))
            default_es_voice = "es-US-Neural2-A" if is_latam else self._settings.google_tts_voice_es
            voice = requested_voice or default_es_voice or None
            return TTSRoute(provider=provider, language=language, voice=voice, dialect=None)

        if language.startswith(("cs", "cz")):
            voice = requested_voice or self._settings.google_tts_voice_cs or None
            return TTSRoute(provider=provider, language="cs-CZ", voice=voice, dialect=None)

        if language.startswith("fr"):
            voice = requested_voice or self._settings.google_tts_voice_fr or None
            return TTSRoute(provider=provider, language=language, voice=voice, dialect=None)

        voice = requested_voice or self._settings.google_tts_voice_de or None
        return TTSRoute(provider=provider, language=language, voice=voice, dialect=None)
