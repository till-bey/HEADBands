"""Self-hosted CPU-ready Faster-Whisper ASR provider (remote Cloud Run service or local CPU int8)."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import Any

import httpx
import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.security import mint_cloud_run_id_token
from app.speech.asr.base import ASRProvider, Transcription
from app.speech.audio_utils import DEFAULT_SAMPLE_RATE, is_wav_bytes, pcm16_to_wav
from app.speech.language.detection import (
    detect_language_and_dialect_from_text,
    normalize_language_code,
)
from app.speech.language.normalization import build_swiss_german_gloss

logger = get_logger(__name__)

# Prompt nudging Whisper to preserve Swiss German dialect orthography instead of normalizing to Hochdeutsch
SWISS_GERMAN_INITIAL_PROMPT = (
    "Schweizerdeutsch, Schwiizerdütsch, Wallisertiitsch, Bärndütsch, Züridütsch, Mundart."
)


class FasterWhisperProvider(ASRProvider):
    """Swiss German & multilingual ASR via remote CPU ASR microservice or local faster-whisper (int8 CPU)."""

    provider_name = "faster_whisper"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._local_model: Any = None
        self._model_lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._settings.asr_timeout_seconds)
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _ensure_local_model(self) -> Any:
        if self._local_model is not None:
            return self._local_model

        async with self._model_lock:
            if self._local_model is not None:
                return self._local_model

            def _load() -> Any:
                import os

                os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from faster_whisper import WhisperModel

                logger.info(
                    "Loading local faster-whisper model in memory",
                    extra={
                        "model": self._settings.whisper_model,
                        "device": self._settings.whisper_device,
                        "compute_type": self._settings.whisper_compute_type,
                        "cpu_threads": self._settings.whisper_cpu_threads,
                    },
                )
                return WhisperModel(
                    self._settings.whisper_model,
                    device=self._settings.whisper_device,
                    compute_type=self._settings.whisper_compute_type,
                    cpu_threads=self._settings.whisper_cpu_threads,
                )

            self._local_model = await asyncio.to_thread(_load)
            return self._local_model

    async def _transcribe_remote(
        self,
        wav_bytes: bytes,
        language: str,
        is_partial: bool,
    ) -> Transcription:
        endpoint = f"{self._settings.asr_endpoint}/v1/transcribe"
        headers: dict[str, str] = {}
        if self._settings.asr_auth_enabled:
            oidc_token = await asyncio.to_thread(mint_cloud_run_id_token, self._settings.asr_endpoint)
            if oidc_token:
                headers["Authorization"] = f"Bearer {oidc_token}"

        started = time.perf_counter()
        client = self._get_http_client()
        response = await client.post(
            endpoint,
            data={"language": language, "preserve_dialect": "true"},
            files={"audio": ("utterance.wav", wav_bytes, "audio/wav")},
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = (time.perf_counter() - started) * 1000.0

        text = str(payload.get("text", "")).strip()
        raw_lang = normalize_language_code(str(payload.get("language", language)))
        inferred_lang, dialect = detect_language_and_dialect_from_text(
            text,
            hint_language=language if language != "auto" else raw_lang,
            default_dialect=self._settings.piper_default_dialect,
        )
        final_lang = "gsw" if language == "gsw" else inferred_lang
        gloss = build_swiss_german_gloss(text) if final_lang == "gsw" else None

        return Transcription(
            text=text,
            normalized_text=gloss,
            language=final_lang,
            detected_dialect=payload.get("detected_dialect") or dialect,
            confidence=float(payload.get("confidence", 0.9)),
            provider=self.provider_name,
            model=str(payload.get("model", self._settings.whisper_model)),
            latency_ms=round(latency_ms, 2),
            is_partial=is_partial,
        )

    async def _transcribe_local(
        self,
        wav_bytes: bytes,
        language: str,
        is_partial: bool,
    ) -> Transcription:
        model = await self._ensure_local_model()
        whisper_lang: str | None
        initial_prompt: str | None = None

        if language in {"gsw", "de-CH"}:
            whisper_lang = "de"
            initial_prompt = SWISS_GERMAN_INITIAL_PROMPT
        elif language == "de-DE":
            whisper_lang = "de"
        elif language in {"fr-FR", "fr-CH"}:
            whisper_lang = "fr"
        elif language.startswith("en"):
            whisper_lang = "en"
        elif language.startswith("es"):
            whisper_lang = "es"
        elif language.startswith(("cs", "cz")):
            whisper_lang = "cs"
        elif language.startswith(("zh", "cmn")):
            whisper_lang = "zh"
        else:
            whisper_lang = None  # Automatic language detection

        started = time.perf_counter()

        def _run_inference() -> tuple[str, str, float]:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                src_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

            if sampwidth == 1:
                audio_arr = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:
                audio_arr = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                audio_arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

            if channels > 1 and len(audio_arr) >= channels:
                usable = (len(audio_arr) // channels) * channels
                audio_arr = audio_arr[:usable].reshape(-1, channels).mean(axis=1)

            if src_rate != 16000 and len(audio_arr) > 1:
                dst_len = max(1, int(round(len(audio_arr) * 16000.0 / float(src_rate))))
                audio_arr = np.interp(
                    np.linspace(0.0, 1.0, num=dst_len, endpoint=False),
                    np.linspace(0.0, 1.0, num=len(audio_arr), endpoint=False),
                    audio_arr,
                ).astype(np.float32)
            else:
                audio_arr = audio_arr.astype(np.float32)

            segments, info = model.transcribe(
                audio_arr,
                language=whisper_lang,
                beam_size=1 if is_partial else 2,
                vad_filter=True,
                initial_prompt=initial_prompt,
                condition_on_previous_text=False,
            )
            parts: list[str] = []
            probs: list[float] = []
            for seg in segments:
                if seg.text:
                    parts.append(seg.text.strip())
                if hasattr(seg, "avg_logprob"):
                    probs.append(float(min(1.0, max(0.05, 1.0 + seg.avg_logprob / 2.0))))
            joined = " ".join(parts).strip()
            conf = sum(probs) / len(probs) if probs else float(getattr(info, "language_probability", 0.85))
            return joined, getattr(info, "language", "de"), conf

        text, whisper_detected, conf = await asyncio.to_thread(_run_inference)
        latency_ms = (time.perf_counter() - started) * 1000.0

        mapped_hint = language
        if mapped_hint == "auto":
            mapped_hint = normalize_language_code(whisper_detected)

        inferred_lang, dialect = detect_language_and_dialect_from_text(
            text,
            hint_language=mapped_hint,
            default_dialect=self._settings.piper_default_dialect,
        )
        final_lang = "gsw" if language == "gsw" else inferred_lang
        gloss = build_swiss_german_gloss(text) if final_lang == "gsw" else None

        return Transcription(
            text=text,
            normalized_text=gloss,
            language=final_lang,
            detected_dialect=dialect,
            confidence=round(min(max(conf, 0.0), 1.0), 4),
            provider=self.provider_name,
            model=f"{self._settings.whisper_model}-{self._settings.whisper_compute_type}-{self._settings.whisper_device}",
            latency_ms=round(latency_ms, 2),
            is_partial=is_partial,
        )

    async def transcribe(
        self,
        audio: bytes,
        language: str = "auto",
        is_partial: bool = False,
    ) -> Transcription:
        if not audio:
            return Transcription(
                text="",
                language=language if language != "auto" else "gsw",
                confidence=0.0,
                provider=self.provider_name,
                model=self._settings.whisper_model,
                latency_ms=0.0,
                is_partial=is_partial,
            )

        wav_bytes = audio if is_wav_bytes(audio) else pcm16_to_wav(audio, DEFAULT_SAMPLE_RATE)
        if self._settings.asr_endpoint:
            return await self._transcribe_remote(wav_bytes, language, is_partial)
        return await self._transcribe_local(wav_bytes, language, is_partial)
