#!/usr/bin/env python3
"""Reproducible Swiss German ASR & TTS Benchmark Suite (README §12).

Evaluates:
- Word Error Rate (WER)
- Dialect preservation (detects unwanted conversion into Standard German)
- End-to-end ASR latency (ms) and hardware profile
across the 5 mandatory Swiss German dialects:
  vs (Wallisertiitsch), zh (Züridütsch), be (Bärndütsch), bs (Baseldytsch), sg (Ostschwiizerdütsch).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import Settings
from app.speech.asr.faster_whisper import FasterWhisperProvider
from app.speech.asr.google_stt import GoogleASRProvider
from app.speech.language.detection import SWISS_GERMAN_MARKERS, tokenize_words
from app.speech.tts.piper import PiperTTSProvider


def normalize_for_wer(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\säöüÄÖÜ]", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return round(dp[len(ref)][len(hyp)] / float(len(ref)), 4)


def preserves_swiss_dialect(hypothesis: str) -> bool:
    tokens = set(tokenize_words(hypothesis))
    if "ß" in hypothesis:
        return False
    return len(tokens & SWISS_GERMAN_MARKERS) > 0


async def run_benchmark(whisper_model: str, include_google_stt: bool) -> dict[str, Any]:
    manifest_path = REPO_ROOT / "data" / "benchmarks" / "metadata" / "benchmark_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["dialects_covered"]

    gcp_project = os.getenv("GCP_PROJECT_ID", "")
    settings = Settings(
        app_env="development",
        gcp_project_id=gcp_project,
        whisper_model=whisper_model,
        whisper_device="cpu",
        whisper_compute_type="int8",
        whisper_cpu_threads=min(8, os.cpu_count() or 4),
        piper_models_dir=REPO_ROOT / "backend" / "models" / "piper",
        allow_non_commercial_models=True,
    )

    piper = PiperTTSProvider(settings)
    fw_provider = FasterWhisperProvider(settings)
    # Warm up faster-whisper model before timing per-utterance latency
    await fw_provider._ensure_local_model()

    google_provider = GoogleASRProvider(settings) if (include_google_stt and gcp_project) else None

    audio_dir = REPO_ROOT / "data" / "benchmarks" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = REPO_ROOT / "data" / "benchmarks" / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for entry in items:
        code = entry["code"]
        dialect_name = entry["name"]
        ref_text = entry["utterance"]

        # Persist reference transcript metadata
        (transcripts_dir / f"{code}_reference.txt").write_text(ref_text + "\n", encoding="utf-8")

        # Use speaker wav if provided in audio/, else synthesize dialect reference with SwissDial Piper
        wav_path = audio_dir / f"{code}_sample.wav"
        if wav_path.is_file():
            wav_bytes = wav_path.read_bytes()
            audio_source = "recorded_wav"
        else:
            wav_bytes = await piper.synthesize(text=ref_text, language="gsw", dialect=code)
            wav_path.write_bytes(wav_bytes)
            audio_source = f"piper_swissdial_{code}"

        fw_res = await fw_provider.transcribe(wav_bytes, language="gsw")
        row: dict[str, Any] = {
            "dialect_code": code,
            "dialect_name": dialect_name,
            "audio_source": audio_source,
            "reference": ref_text,
            "faster_whisper": {
                "model": fw_res.model,
                "hypothesis": fw_res.text,
                "wer": compute_wer(ref_text, fw_res.text),
                "dialect_preserved": preserves_swiss_dialect(fw_res.text),
                "latency_ms": fw_res.latency_ms,
            },
        }

        if google_provider is not None:
            g_res = await google_provider.transcribe(wav_bytes, language="de-CH")
            row["google_stt_de_ch"] = {
                "model": g_res.model,
                "hypothesis": g_res.text,
                "wer": compute_wer(ref_text, g_res.text),
                "dialect_preserved": preserves_swiss_dialect(g_res.text),
                "latency_ms": g_res.latency_ms,
            }

        records.append(row)

    await fw_provider.close()

    avg_fw_wer = round(sum(r["faster_whisper"]["wer"] for r in records) / len(records), 4)
    avg_fw_lat = round(sum(r["faster_whisper"]["latency_ms"] for r in records) / len(records), 1)
    fw_dialect_rate = round(
        sum(1 for r in records if r["faster_whisper"]["dialect_preserved"]) / len(records), 2
    )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "gpu_used": False,
        },
        "whisper_config": {
            "model": whisper_model,
            "device": "cpu",
            "compute_type": "int8",
            "license": "MIT",
        },
        "aggregate": {
            "faster_whisper_avg_wer": avg_fw_wer,
            "faster_whisper_avg_latency_ms": avg_fw_lat,
            "faster_whisper_dialect_preservation_ratio": fw_dialect_rate,
        },
        "dialects": records,
    }

    if google_provider is not None and records and "google_stt_de_ch" in records[0]:
        summary["aggregate"]["google_de_ch_avg_wer"] = round(
            sum(r["google_stt_de_ch"]["wer"] for r in records) / len(records), 4
        )
        summary["aggregate"]["google_de_ch_avg_latency_ms"] = round(
            sum(r["google_stt_de_ch"]["latency_ms"] for r in records) / len(records), 1
        )
        summary["aggregate"]["google_de_ch_dialect_preservation_ratio"] = round(
            sum(1 for r in records if r["google_stt_de_ch"]["dialect_preserved"]) / len(records), 2
        )

    out_path = REPO_ROOT / "data" / "benchmarks" / "results" / "latest_benchmark.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Swiss German multi-dialect ASR benchmark.")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--include-google-stt", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(run_benchmark(args.whisper_model, args.include_google_stt))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
