#!/usr/bin/env python3
"""Download open-source Swiss German TTS (Piper SwissDial) and CPU Whisper models after license check."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import SWISS_DIALECT_LABELS, Settings
from app.speech.tts.piper import PiperTTSProvider, resolve_swissdial_voice_key


async def download_piper_dialects(dialects: list[str], models_dir: Path, allow_nc: bool) -> None:
    if not allow_nc:
        raise SystemExit(
            "ERROR: d41k4n/piper-swissdial models are licensed under CC BY-NC 4.0 (non-commercial). "
            "Pass --acknowledge-cc-by-nc-4.0 for non-commercial development/sandbox evaluation."
        )
    settings = Settings(
        app_env="development",
        piper_models_dir=models_dir,
        allow_non_commercial_models=True,
    )
    provider = PiperTTSProvider(settings)
    for code in dialects:
        d_code, voice_key = resolve_swissdial_voice_key(code)
        onnx_path, _json_path = await provider._ensure_model_files(d_code, voice_key)
        print(f"[OK] {d_code} ({SWISS_DIALECT_LABELS[d_code]}): {onnx_path.name} ({onnx_path.stat().st_size // 1024} KB)")


def preload_whisper_cpu(model_name: str) -> None:
    from faster_whisper import WhisperModel

    print(f"[INFO] Pre-caching faster-whisper '{model_name}' (MIT license, device=cpu, compute_type=int8)...")
    WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    print(f"[OK] faster-whisper '{model_name}' ready on CPU int8.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Piper SwissDial & CPU Whisper models safely.")
    parser.add_argument(
        "--dialects",
        nargs="+",
        default=["vs"],
        choices=list(SWISS_DIALECT_LABELS.keys()) + ["all"],
        help="SwissDial dialect codes to download (default: vs for Wallisertiitsch / Leukerbad).",
    )
    parser.add_argument("--whisper-model", default="small", help="Whisper model size (e.g. small, large-v3-turbo, large-v3).")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument(
        "--acknowledge-cc-by-nc-4.0",
        action="store_true",
        help="Acknowledge non-commercial CC BY-NC 4.0 license for SwissDial Piper checkpoints.",
    )
    args = parser.parse_args()

    selected = list(SWISS_DIALECT_LABELS.keys()) if "all" in args.dialects else args.dialects
    models_dir = REPO_ROOT / "backend" / "models" / "piper"
    asyncio.run(download_piper_dialects(selected, models_dir, args.acknowledge_cc_by_nc_4_0))

    if not args.skip_whisper:
        preload_whisper_cpu(args.whisper_model)


if __name__ == "__main__":
    main()
