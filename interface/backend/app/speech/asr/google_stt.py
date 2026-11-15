"""Google Cloud Speech-to-Text V2 provider (EU endpoint, Chirp 3 & Swiss locales)."""

from __future__ import annotations

import time

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InvalidArgument, NotFound
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

from app.core.config import Settings
from app.core.logging import get_logger
from app.speech.asr.base import ASRProvider, Transcription
from app.speech.audio_utils import DEFAULT_SAMPLE_RATE, is_wav_bytes, pcm16_to_wav
from app.speech.language.detection import (
    clean_cjk_spacing,
    detect_language_and_dialect_from_text,
    normalize_language_code,
)
from app.speech.language.normalization import build_swiss_german_gloss

logger = get_logger(__name__)


class GoogleASRProvider(ASRProvider):
    """Speech-to-Text V2 adapter targeting the configured regional endpoint (default: eu)."""

    provider_name = "google"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: speech_v2.SpeechAsyncClient | None = None
        if settings.app_env != "test" and settings.gcp_project_id:
            try:
                self._get_client()
            except Exception:
                self._client = None

    def _get_client(self) -> speech_v2.SpeechAsyncClient:
        if self._client is None:
            loc = self._settings.gcp_stt_location.strip()
            options = (
                ClientOptions(api_endpoint=f"{loc}-speech.googleapis.com")
                if loc and loc != "global"
                else None
            )
            self._client = speech_v2.SpeechAsyncClient(client_options=options)
        return self._client

    def _resolve_language_and_model(self, requested_language: str) -> tuple[list[str], str]:
        if requested_language in {"auto", "", "gsw"}:
            # Verified on eu-speech.googleapis.com: Chirp 3 supports native language_codes=['auto']
            # across English, German, French, Chinese, and Swiss German speech in a single fast request.
            return (["auto"], self._settings.stt_standard_de_model)
        if requested_language in {"de-DE", "de-CH"}:
            # On eu-speech.googleapis.com, de-DE with chirp_3 delivers wideband ~350ms recognition
            # for both Standard German and Swiss Standard German without HTTP 400 fallback to 8kHz telephony.
            return ([self._settings.stt_standard_de_language], self._settings.stt_standard_de_model)
        if requested_language == "fr-CH":
            return (["fr-CH"], self._settings.stt_french_model)
        if requested_language == "fr-FR":
            return ([self._settings.stt_fr_language], self._settings.stt_french_model)
        if requested_language == "en-GB":
            return (["en-GB"], self._settings.stt_english_model)
        if requested_language in {"en", "en-US"}:
            return ([self._settings.stt_en_language], self._settings.stt_english_model)
        if requested_language in {"es-US", "es-MX", "es-419"}:
            return (["es-US"], self._settings.stt_spanish_model)
        if requested_language in {"es", "es-ES"} or requested_language.startswith("es"):
            return ([self._settings.stt_es_language], self._settings.stt_spanish_model)
        if requested_language in {"cs", "cz", "cs-CZ"} or requested_language.startswith(("cs", "cz")):
            return ([self._settings.stt_cs_language], self._settings.stt_czech_model)
        if requested_language in {"zh-TW", "zh-HK", "cmn-TW", "cmn-Hant-TW"}:
            return (["cmn-Hant-TW"], self._settings.stt_chinese_model)
        if requested_language in {"zh", "zh-CN", "cmn-CN", "cmn-Hans-CN"}:
            return ([self._settings.stt_zh_language], self._settings.stt_chinese_model)
        return ([requested_language], self._settings.stt_standard_de_model)

    async def _recognize_once(
        self,
        wav_audio: bytes,
        language_codes: list[str],
        model: str,
    ) -> cloud_speech.RecognizeResponse:
        if not self._settings.gcp_project_id:
            raise RuntimeError("GCP_PROJECT_ID must be configured to use GoogleASRProvider.")

        loc = self._settings.gcp_stt_location.strip() or "eu"
        recognizer = f"projects/{self._settings.gcp_project_id}/locations/{loc}/recognizers/_"

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=language_codes,
            model=model,
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer,
            config=config,
            content=wav_audio,
        )
        client = self._get_client()
        return await client.recognize(request=request)

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
                model=self._settings.stt_standard_de_model,
                latency_ms=0.0,
                is_partial=is_partial,
            )

        wav_bytes = audio if is_wav_bytes(audio) else pcm16_to_wav(audio, DEFAULT_SAMPLE_RATE)
        language_codes, primary_model = self._resolve_language_and_model(language)
        started = time.perf_counter()
        used_model = primary_model

        try:
            response = await self._recognize_once(wav_bytes, language_codes, primary_model)
        except (InvalidArgument, NotFound) as exc:
            # Handle locale/model availability mismatch cleanly (e.g., fr-CH, de-CH, or cmn-Hant-TW)
            if language_codes == ["cmn-Hant-TW"]:
                fallback_codes = ["auto"]
                fallback_model = "chirp_3"
            elif language_codes == ["fr-CH"]:
                fallback_codes = ["fr-FR"]
                fallback_model = "long" if primary_model != "long" else "telephony"
            elif language_codes == ["auto"]:
                fallback_codes = ["de-DE", "en-US", "fr-FR"]
                fallback_model = "long" if primary_model != "long" else "telephony"
            else:
                fallback_codes = language_codes
                fallback_model = "long" if primary_model != "long" else "telephony"
            logger.warning(
                "Primary STT V2 model/locale unavailable; retrying with fallback",
                extra={
                    "requested_codes": language_codes,
                    "primary_model": primary_model,
                    "fallback_codes": fallback_codes,
                    "fallback_model": fallback_model,
                    "reason": str(exc),
                },
            )
            used_model = fallback_model
            response = await self._recognize_once(wav_bytes, fallback_codes, fallback_model)

        latency_ms = (time.perf_counter() - started) * 1000.0
        segments: list[str] = []
        confidences: list[float] = []
        detected_lang = language_codes[0]

        for res in response.results:
            if res.language_code:
                detected_lang = res.language_code
            if res.alternatives:
                top = res.alternatives[0]
                if top.transcript:
                    segments.append(top.transcript.strip())
                if top.confidence:
                    confidences.append(float(top.confidence))

        transcript_text = clean_cjk_spacing(" ".join(segments).strip())
        avg_conf = sum(confidences) / len(confidences) if confidences else (0.9 if transcript_text else 0.0)

        norm_detected = normalize_language_code(detected_lang)
        is_auto = language in {"auto", ""}
        hint_for_text = norm_detected if is_auto else normalize_language_code(language)
        inferred_lang, dialect = detect_language_and_dialect_from_text(
            transcript_text,
            hint_language=hint_for_text,
            default_dialect=self._settings.piper_default_dialect,
        )
        if not is_auto:
            final_lang = normalize_language_code(language)
            if final_lang != "gsw":
                dialect = None
        else:
            final_lang = (
                "gsw"
                if inferred_lang == "gsw"
                else (inferred_lang if inferred_lang != "auto" else norm_detected)
            )
            if final_lang == "auto":
                final_lang = "de-DE"

        if final_lang == "de-CH" and "ß" in transcript_text:
            transcript_text = transcript_text.replace("ß", "ss")
        gloss = build_swiss_german_gloss(transcript_text) if final_lang == "gsw" else None

        return Transcription(
            text=transcript_text,
            normalized_text=gloss,
            language=final_lang,
            detected_dialect=dialect,
            confidence=min(max(avg_conf, 0.0), 1.0),
            provider=self.provider_name,
            model=used_model,
            latency_ms=round(latency_ms, 2),
            is_partial=is_partial,
        )
