# Swiss German Evaluation & Benchmark Dataset Directory

This directory holds the reproducible Swiss German ASR & TTS benchmark structure (`README.md` §12):

```text
data/
  benchmarks/
    audio/        # Local WAV recordings (16 kHz 16-bit mono) — GITIGNORED for speaker privacy
    transcripts/  # Verified reference dialect transcripts (.json / .txt)
    metadata/     # Speaker consent, dialect labels, noise conditions, and model license records
    results/      # Benchmark run JSON reports (WER, dialect preservation, latency) — GITIGNORED
```

## Privacy & Licensing Rules
1. **Never commit personal speaker audio** (`*.wav`, `*.mp3`, `*.flac` are blocked in `.gitignore`).
2. Obtain explicit informed speaker consent before recording real voices.
3. Verify the license of every model and voice corpus independently before commercial use (`d41k4n/piper-swissdial` is **CC BY-NC 4.0** non-commercial; ETH Zürich `SwissDial v1.1` research terms apply).

## Running the Benchmark
```bash
# Run CPU int8 faster-whisper benchmark across all 5 required Swiss German dialects (vs, zh, be, bs, sg)
python scripts/benchmark_swiss_german.py --whisper-model small --include-google-stt
```
