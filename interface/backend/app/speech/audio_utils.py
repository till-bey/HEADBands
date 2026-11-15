"""In-memory PCM 16kHz 16-bit mono processing, VAD, utterance buffering, and sentence splitting."""

from __future__ import annotations

import io
import math
import re
import wave
from dataclasses import dataclass, field

import numpy as np

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit signed PCM
DEFAULT_CHANNELS = 1

_ABBREV_PLACEHOLDER = "\u0000"
_PROTECTED_ABBREVS = (
    "St.",
    "z.B.",
    "u.a.",
    "bzw.",
    "Dr.",
    "Prof.",
    "Nr.",
    "ca.",
    "M.",
    "Mme.",
    "Sr.",
    "Sra.",
    "Ud.",
    "např.",
    "atd.",
    "tzv.",
)


def is_wav_bytes(data: bytes) -> bool:
    """Return True if data begins with a valid RIFF/WAVE header."""
    return len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def pcm16_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> bytes:
    """Wrap raw 16-bit signed little-endian PCM bytes into an in-memory WAV file."""
    if is_wav_bytes(pcm_bytes):
        return pcm_bytes
    if len(pcm_bytes) % 2 == 1:
        pcm_bytes = pcm_bytes[:-1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(DEFAULT_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def extract_pcm16_from_wav(data: bytes) -> tuple[bytes, int, int]:
    """Return (raw_pcm_bytes, sample_rate, channels) from WAV or raw 16kHz mono PCM."""
    if not is_wav_bytes(data):
        if len(data) % 2 == 1:
            data = data[:-1]
        return data, DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS

    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth * 8}-bit (expected 16-bit PCM)")

    if channels > 1:
        samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
        mono = samples.mean(axis=1).astype(np.int16)
        frames = mono.tobytes()
        channels = 1

    return frames, sample_rate, channels


def pcm16_duration_ms(pcm_bytes: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Compute duration in milliseconds of mono 16-bit PCM."""
    num_samples = len(pcm_bytes) // DEFAULT_SAMPLE_WIDTH
    if sample_rate <= 0:
        return 0.0
    return (num_samples / sample_rate) * 1000.0


def compute_rms_energy(pcm_bytes: bytes) -> float:
    """Compute normalized RMS amplitude in [0.0, 1.0] for 16-bit signed PCM."""
    if len(pcm_bytes) < 2:
        return 0.0
    if len(pcm_bytes) % 2 == 1:
        pcm_bytes = pcm_bytes[:-1]
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


@dataclass
class VADEvent:
    """Outcome of processing an incoming PCM audio frame through VAD + utterance buffer."""

    is_speech: bool
    rms: float
    speech_started: bool = False
    barge_in_confirmed: bool = False
    should_emit_partial: bool = False
    end_of_utterance: bool = False
    utterance_pcm: bytes = b""


@dataclass
class StreamingAudioBuffer:
    """In-memory streaming audio buffer with energy VAD, partial windows, and EoU detection."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    energy_threshold: float = 0.015
    min_speech_ms: float = 200.0
    barge_in_min_speech_ms: float = 280.0
    end_silence_ms: float = 650.0
    partial_interval_ms: float = 900.0
    max_utterance_ms: float = 25000.0

    _buffer: bytearray = field(default_factory=bytearray, init=False)
    _pre_roll: bytearray = field(default_factory=bytearray, init=False)
    _in_speech: bool = field(default=False, init=False)
    _barge_in_emitted: bool = field(default=False, init=False)
    _speech_ms: float = field(default=0.0, init=False)
    _silence_ms: float = field(default=0.0, init=False)
    _since_partial_ms: float = field(default=0.0, init=False)

    @property
    def speech_duration_ms(self) -> float:
        """Total voiced audio duration accumulated in the active utterance buffer."""
        return self._speech_ms

    def push_pcm(self, chunk: bytes) -> VADEvent:
        """Ingest a 16-bit mono PCM chunk and evaluate VAD & utterance state."""
        if len(chunk) % 2 == 1:
            chunk = chunk[:-1]
        if not chunk:
            return VADEvent(is_speech=False, rms=0.0)

        chunk_ms = pcm16_duration_ms(chunk, self.sample_rate)
        rms = compute_rms_energy(chunk)
        is_speech = rms >= self.energy_threshold

        if not self._in_speech:
            # Keep ~240ms of pre-roll so initial consonants ("Grüezi", "Chasch", "Geh") are never clipped
            max_preroll_bytes = int(self.sample_rate * 0.24) * 2
            self._pre_roll.extend(chunk)
            if len(self._pre_roll) > max_preroll_bytes:
                del self._pre_roll[:-max_preroll_bytes]

            if is_speech:
                self._in_speech = True
                self._speech_ms = chunk_ms
                self._silence_ms = 0.0
                self._since_partial_ms = chunk_ms
                self._barge_in_emitted = chunk_ms >= self.barge_in_min_speech_ms
                self._buffer.clear()
                self._buffer.extend(self._pre_roll)
                self._pre_roll.clear()
                return VADEvent(
                    is_speech=True,
                    rms=rms,
                    speech_started=True,
                    barge_in_confirmed=self._barge_in_emitted,
                )

            return VADEvent(is_speech=False, rms=rms)

        # Currently inside an active speech segment
        self._buffer.extend(chunk)
        total_ms = pcm16_duration_ms(self._buffer, self.sample_rate)

        barge_in_now = False
        if is_speech:
            self._speech_ms += chunk_ms
            self._silence_ms = 0.0
            self._since_partial_ms += chunk_ms
            if not self._barge_in_emitted and self._speech_ms >= self.barge_in_min_speech_ms:
                self._barge_in_emitted = True
                barge_in_now = True
        else:
            self._silence_ms += chunk_ms

        # Force flush if utterance reaches max duration
        if total_ms >= self.max_utterance_ms:
            completed = self.flush()
            return VADEvent(
                is_speech=is_speech,
                rms=rms,
                barge_in_confirmed=barge_in_now,
                end_of_utterance=bool(completed),
                utterance_pcm=completed,
            )

        # End of utterance reached after trailing silence
        if self._silence_ms >= self.end_silence_ms:
            if self._speech_ms >= self.min_speech_ms:
                completed = self.flush()
                return VADEvent(
                    is_speech=False,
                    rms=rms,
                    end_of_utterance=True,
                    utterance_pcm=completed,
                )
            self.reset()
            return VADEvent(is_speech=False, rms=rms)

        emit_partial = False
        if self._since_partial_ms >= self.partial_interval_ms and self._speech_ms >= self.min_speech_ms:
            self._since_partial_ms = 0.0
            emit_partial = True

        return VADEvent(
            is_speech=is_speech,
            rms=rms,
            barge_in_confirmed=barge_in_now,
            should_emit_partial=emit_partial,
            utterance_pcm=bytes(self._buffer) if emit_partial else b"",
        )

    def snapshot_pcm(self) -> bytes:
        """Return current buffered utterance bytes without clearing."""
        return bytes(self._buffer)

    def flush(self) -> bytes:
        """Flush and return buffered utterance bytes, resetting state."""
        data = bytes(self._buffer)
        self.reset()
        return data

    def reset(self) -> None:
        """Clear all buffered audio from memory."""
        self._buffer.clear()
        self._pre_roll.clear()
        self._in_speech = False
        self._barge_in_emitted = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._since_partial_ms = 0.0


def split_into_sentences(text: str) -> list[str]:
    """Split text into natural sentences for low-latency streaming TTS while preserving Swiss abbreviations."""
    cleaned = text.strip()
    if not cleaned:
        return []

    masked = cleaned
    for abbrev in _PROTECTED_ABBREVS:
        masked = masked.replace(abbrev, abbrev.replace(".", _ABBREV_PLACEHOLDER))

    raw_parts = re.split(r"(?<=[.!?])\s+|(?<=[。！？；])\s*|\n+", masked)
    sentences: list[str] = []
    for part in raw_parts:
        restored = part.replace(_ABBREV_PLACEHOLDER, ".").strip()
        if restored:
            sentences.append(restored)
    return sentences


def synthesize_sine_wav(
    duration_ms: int = 160,
    freq_hz: float = 440.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.25,
) -> bytes:
    """Generate a valid playable 16-bit mono WAV tone for mock providers & audio tests."""
    num_samples = max(1, int(sample_rate * (duration_ms / 1000.0)))
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    wave_float = amplitude * np.sin(2.0 * math.pi * freq_hz * t)
    pcm = (wave_float * 32767.0).astype(np.int16).tobytes()
    return pcm16_to_wav(pcm, sample_rate=sample_rate, channels=1)
