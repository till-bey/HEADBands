"""Swiss German Piper ONNX CPU TTS provider supporting all 8 SwissDial dialects."""

from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path
from typing import Any

import httpx

from app.core.config import SWISS_DIALECT_LABELS, Settings
from app.core.logging import get_logger
from app.speech.audio_utils import pcm16_to_wav
from app.speech.tts.base import TTSProvider

logger = get_logger(__name__)

SWISSDIAL_HF_BASE = "https://huggingface.co/d41k4n/piper-swissdial/resolve/main"
THORSTEN_CC0_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium"


def resolve_swissdial_voice_key(dialect: str | None, default_dialect: str = "vs") -> tuple[str, str]:
    """Map dialect code or voice name to (dialect_code, voice_key)."""
    if dialect:
        cleaned = dialect.strip().lower()
        if cleaned in SWISS_DIALECT_LABELS:
            return cleaned, f"de_CH-swissdial_{cleaned}-medium"
        for code in SWISS_DIALECT_LABELS:
            if f"swissdial_{code}" in cleaned:
                return code, f"de_CH-swissdial_{code}-medium"
    return default_dialect, f"de_CH-swissdial_{default_dialect}-medium"


class PiperTTSProvider(TTSProvider):
    """CPU-optimized ONNX Piper TTS provider for Swiss German (d41k4n/piper-swissdial)."""

    provider_name = "piper"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._voices: dict[str, Any] = {}
        self._download_lock = asyncio.Lock()
        self._synth_lock = asyncio.Lock()

    def _assert_license_compliance(self) -> None:
        if self._settings.piper_model_license == "cc-by-nc-4.0":
            if self._settings.app_env == "production":
                raise PermissionError(
                    "SwissDial Piper voices (CC BY-NC 4.0) are non-commercial and blocked in production."
                )
            if not self._settings.allow_non_commercial_models:
                raise PermissionError(
                    "SwissDial Piper voices require ALLOW_NON_COMMERCIAL_MODELS=true in non-commercial mode."
                )

    async def _ensure_model_files(self, dialect_code: str, voice_key: str) -> tuple[Path, Path]:
        # Explicit single-model override via env
        if self._settings.piper_model_path:
            model_path = Path(self._settings.piper_model_path)
            cfg_path = (
                Path(self._settings.piper_config_path)
                if self._settings.piper_config_path
                else Path(f"{self._settings.piper_model_path}.json")
            )
            if model_path.is_file() and cfg_path.is_file():
                return model_path, cfg_path

        self._assert_license_compliance()

        models_dir = Path(self._settings.piper_models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        if self._settings.piper_model_license == "cc0":
            onnx_path = models_dir / "de_DE-thorsten-medium.onnx"
            json_path = models_dir / "de_DE-thorsten-medium.onnx.json"
            onnx_url = f"{THORSTEN_CC0_HF_BASE}/de_DE-thorsten-medium.onnx"
            json_url = f"{THORSTEN_CC0_HF_BASE}/de_DE-thorsten-medium.onnx.json"
        else:
            onnx_path = models_dir / f"{voice_key}.onnx"
            json_path = models_dir / f"{voice_key}.onnx.json"
            rel_dir = f"de/de_CH/swissdial_{dialect_code}/medium"
            onnx_url = f"{SWISSDIAL_HF_BASE}/{rel_dir}/{voice_key}.onnx"
            json_url = f"{SWISSDIAL_HF_BASE}/{rel_dir}/{voice_key}.onnx.json"

        if onnx_path.is_file() and json_path.is_file():
            return onnx_path, json_path

        async with self._download_lock:
            if onnx_path.is_file() and json_path.is_file():
                return onnx_path, json_path

            logger.info(
                "Downloading Piper ONNX voice checkpoint",
                extra={
                    "voice_key": voice_key,
                    "dialect": dialect_code,
                    "license": self._settings.piper_model_license,
                },
            )
            async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
                r_json = await client.get(json_url)
                r_json.raise_for_status()
                tmp_json = json_path.with_suffix(".json.part")
                tmp_json.write_bytes(r_json.content)
                tmp_json.replace(json_path)

                tmp_onnx = onnx_path.with_suffix(".onnx.part")
                async with client.stream("GET", onnx_url) as stream:
                    stream.raise_for_status()
                    with tmp_onnx.open("wb") as fh:
                        async for chunk in stream.aiter_bytes(chunk_size=262144):
                            fh.write(chunk)
                tmp_onnx.replace(onnx_path)

        return onnx_path, json_path

    async def _get_loaded_voice(self, dialect_code: str, voice_key: str) -> Any:
        cache_key = self._settings.piper_model_path or voice_key
        if cache_key in self._voices:
            return self._voices[cache_key]

        onnx_path, json_path = await self._ensure_model_files(dialect_code, voice_key)

        def _load() -> Any:
            from piper import PiperVoice

            return PiperVoice.load(str(onnx_path), config_path=str(json_path), use_cuda=False)

        loaded = await asyncio.to_thread(_load)
        self._voices[cache_key] = loaded
        return loaded

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
            raise asyncio.CancelledError("Piper TTS cancelled before synthesis")

        cleaned = text.strip()
        if not cleaned:
            return b""

        dialect_code, voice_key = resolve_swissdial_voice_key(
            voice or dialect,
            default_dialect=self._settings.piper_default_dialect,
        )
        piper_voice = await self._get_loaded_voice(dialect_code, voice_key)
        rate = max(0.5, min(2.0, speaking_rate or self._settings.tts_speaking_rate))
        length_scale = 1.0 / rate

        def _run_piper() -> bytes:
            # Support streaming chunk API in piper-tts >= 1.3 (piper1-gpl) with mid-sentence cancellation
            try:
                from piper import SynthesisConfig

                syn_config = SynthesisConfig(length_scale=length_scale)
                pcm_parts = bytearray()
                sample_rate = 22050
                channels = 1
                for audio_chunk in piper_voice.synthesize(cleaned, syn_config=syn_config):
                    if cancel_event is not None and cancel_event.is_set():
                        return b""
                    pcm_parts.extend(audio_chunk.audio_int16_bytes)
                    sample_rate = getattr(audio_chunk, "sample_rate", sample_rate)
                    channels = getattr(audio_chunk, "sample_channels", channels)
                return pcm16_to_wav(bytes(pcm_parts), sample_rate=sample_rate, channels=channels)
            except ImportError:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    piper_voice.synthesize(cleaned, wf, length_scale=length_scale)
                return buf.getvalue()

        async with self._synth_lock:
            wav_bytes = await asyncio.to_thread(_run_piper)
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("Piper TTS cancelled during synthesis")
        return wav_bytes
