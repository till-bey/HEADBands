"""Tests for the dedicated CPU faster-whisper ASR microservice HTTP contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.speech.asr.faster_whisper import FasterWhisperProvider
from app.speech.audio_utils import synthesize_sine_wav

ASR_MAIN_PATH = Path(__file__).resolve().parents[3] / "asr_service" / "app" / "main.py"


def _load_asr_service_module():
    spec = importlib.util.spec_from_file_location("asr_service_main", ASR_MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asr_microservice_health_and_transcribe_contract() -> None:
    asr_mod = _load_asr_service_module()

    fake_seg = MagicMock()
    fake_seg.text = "Grüezi üs Leukerbad, s Wätter isch güet."
    fake_seg.avg_logprob = -0.15
    fake_info = MagicMock()
    fake_info.language = "de"
    fake_info.language_probability = 0.96

    fake_whisper = MagicMock()
    fake_whisper.transcribe.return_value = ([fake_seg], fake_info)
    asr_mod.holder._model = fake_whisper

    with TestClient(asr_mod.app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["device"] == "cpu"
        assert h.json()["compute_type"] == "int8"

        wav = synthesize_sine_wav(duration_ms=200, sample_rate=22050)
        r = client.post(
            "/v1/transcribe",
            data={"language": "gsw", "preserve_dialect": "true"},
            files={"audio": ("sample.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["language"] == "gsw"
        assert "Leukerbad" in body["text"]
        passed_audio = fake_whisper.transcribe.call_args[0][0]
        assert isinstance(passed_audio, np.ndarray)
        assert passed_audio.dtype == np.float32
        assert len(passed_audio) == 3200  # 200 ms resampled from 22.05 kHz -> 16 kHz


@pytest.mark.asyncio
async def test_faster_whisper_provider_remote_microservice_mode() -> None:
    asr_mod = _load_asr_service_module()

    fake_seg = MagicMock()
    fake_seg.text = "Grüezi mitenand z Leukerbad!"
    fake_seg.avg_logprob = -0.12
    fake_info = MagicMock()
    fake_info.language = "de"
    fake_info.language_probability = 0.95

    fake_whisper = MagicMock()
    fake_whisper.transcribe.return_value = ([fake_seg], fake_info)
    asr_mod.holder._model = fake_whisper

    settings = Settings(
        asr_endpoint="http://asr-service.internal:8081",
        asr_auth_enabled=False,
        whisper_device="cpu",
        whisper_compute_type="int8",
    )
    provider = FasterWhisperProvider(settings)
    provider._http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=asr_mod.app),
        base_url="http://asr-service.internal:8081",
    )

    try:
        wav = synthesize_sine_wav(duration_ms=200)
        res = await provider.transcribe(wav, language="gsw")
        assert res.provider == "faster_whisper"
        assert res.language == "gsw"
        assert res.detected_dialect == "vs"
        assert "Leukerbad" in res.text
        assert res.normalized_text is not None
    finally:
        await provider.close()
