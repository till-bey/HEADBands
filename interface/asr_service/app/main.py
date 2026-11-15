"""Self-hosted CPU int8 Faster-Whisper inference service for Cloud Run (europe-west6) or GCE."""

from __future__ import annotations

import asyncio
import io
import os
import time
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status

SWISS_GERMAN_PROMPT = (
    "Schweizerdeutsch, Schwiizerdütsch, Wallisertiitsch, Bärndütsch, Züridütsch, Mundart."
)


def _ensure_wav(raw: bytes) -> bytes:
    if len(raw) >= 44 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return raw
    if len(raw) % 2 == 1:
        raw = raw[:-1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(raw)
    return buf.getvalue()


def _wav_to_float32_16k(wav_bytes: bytes) -> np.ndarray:
    """Decode PCM WAV directly to 16 kHz mono float32 [-1.0, 1.0], bypassing PyAV version quirks."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if not frames or src_rate <= 0:
        return np.zeros(0, dtype=np.float32)

    if sampwidth == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)

    if src_rate != 16000 and len(samples) > 1:
        dst_len = max(1, int(round(len(samples) * 16000.0 / float(src_rate))))
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)

    return np.clip(samples, -1.0, 1.0).astype(np.float32)


class WhisperModelHolder:
    """Loads the faster-whisper CPU int8 model once and keeps it warm across requests."""

    def __init__(self) -> None:
        self.model_name = os.getenv("WHISPER_MODEL", "small")
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "4"))
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "int8"
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model

            def _load() -> Any:
                os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from faster_whisper import WhisperModel

                return WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                )

            self._model = await asyncio.to_thread(_load)
            return self._model


holder = WhisperModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    preload = os.getenv("ASR_PRELOAD_ON_STARTUP", "false").lower() == "true"
    if preload:
        await holder.get_model()
    yield


app = FastAPI(
    title="Swiss German CPU ASR Service (faster-whisper int8)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "swiss-asr-service",
        "model": holder.model_name,
        "device": holder.device,
        "compute_type": holder.compute_type,
        "loaded": holder._model is not None,
    }


@app.post("/v1/transcribe")
async def transcribe_endpoint(
    audio: Annotated[UploadFile, File(...)],
    language: Annotated[str, Form()] = "gsw",
    preserve_dialect: Annotated[bool, Form()] = True,
) -> dict[str, Any]:
    raw_bytes = await audio.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload.")

    wav_bytes = _ensure_wav(raw_bytes)
    model = await holder.get_model()

    whisper_lang: str | None
    prompt: str | None = None
    if language in {"gsw", "de-CH"}:
        whisper_lang = "de"
        if preserve_dialect:
            prompt = SWISS_GERMAN_PROMPT
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
        whisper_lang = None

    started = time.perf_counter()

    def _infer() -> tuple[str, str, float]:
        audio_16k = _wav_to_float32_16k(wav_bytes)
        segments, info = model.transcribe(
            audio_16k,
            language=whisper_lang,
            beam_size=2,
            vad_filter=True,
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        texts: list[str] = []
        confs: list[float] = []
        for seg in segments:
            if seg.text:
                texts.append(seg.text.strip())
            if hasattr(seg, "avg_logprob"):
                confs.append(float(min(1.0, max(0.05, 1.0 + seg.avg_logprob / 2.0))))
        avg_conf = sum(confs) / len(confs) if confs else float(getattr(info, "language_probability", 0.85))
        return " ".join(texts).strip(), getattr(info, "language", "de"), avg_conf

    transcript, detected_lang, confidence = await asyncio.to_thread(_infer)
    latency_ms = (time.perf_counter() - started) * 1000.0

    return {
        "text": transcript,
        "language": "gsw" if language == "gsw" else detected_lang,
        "confidence": round(confidence, 4),
        "model": f"{holder.model_name}-{holder.compute_type}-{holder.device}",
        "latency_ms": round(latency_ms, 2),
    }
