"""Base TTSProvider interface with sentence-chunked streaming, interruption, and MockTTSProvider."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.speech.audio_utils import split_into_sentences, synthesize_sine_wav


class TTSProvider(ABC):
    """Replaceable interface for Text-to-Speech synthesis."""

    provider_name: str = "base"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: str,
        voice: str | None = None,
        dialect: str | None = None,
        speaking_rate: float = 1.0,
        cancel_event: asyncio.Event | None = None,
    ) -> bytes:
        """Synthesize text into playable WAV audio bytes."""
        raise NotImplementedError

    async def synthesize_stream(
        self,
        text: str,
        language: str,
        voice: str | None = None,
        dialect: str | None = None,
        speaking_rate: float = 1.0,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[tuple[str, bytes]]:
        """Synthesize sentence chunks concurrently while yielding in strict sentence order with instant barge-in."""
        sentences = split_into_sentences(text)
        if not sentences:
            return
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("TTS stream interrupted by user")

        tasks: list[asyncio.Task[bytes]] = [
            asyncio.create_task(
                self.synthesize(
                    text=sentence,
                    language=language,
                    voice=voice,
                    dialect=dialect,
                    speaking_rate=speaking_rate,
                    cancel_event=cancel_event,
                )
            )
            for sentence in sentences
        ]
        try:
            for sentence, task in zip(sentences, tasks, strict=True):
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError("TTS stream interrupted by user")
                wav_chunk = await task
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError("TTS stream interrupted by user")
                if wav_chunk:
                    yield sentence, wav_chunk
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()


class MockTTSProvider(TTSProvider):
    """Deterministic playable WAV synthesizer for unit tests and offline sandboxes."""

    provider_name = "mock"

    LANGUAGE_PITCH_HZ: dict[str, float] = {
        "gsw": 523.25,    # C5
        "de-CH": 493.88,  # B4
        "de-DE": 440.00,  # A4
        "fr-FR": 392.00,  # G4
        "fr-CH": 415.30,  # G#4
        "en-US": 349.23,  # F4
        "en-GB": 369.99,  # F#4
        "es-ES": 293.66,  # D4
        "es-US": 277.18,  # C#4
        "cs-CZ": 261.63,  # C4
        "zh-CN": 329.63,  # E4
        "zh-TW": 311.13,  # D#4
    }

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
            raise asyncio.CancelledError("TTS interrupted")
        cleaned = text.strip()
        if not cleaned:
            return b""
        pitch = self.LANGUAGE_PITCH_HZ.get(language, 440.0)
        duration_ms = max(120, min(600, int(len(cleaned) * 12 / max(0.5, speaking_rate))))
        return synthesize_sine_wav(duration_ms=duration_ms, freq_hz=pitch, sample_rate=16000)
