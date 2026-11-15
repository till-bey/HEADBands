# Swiss German Speech Recognition & Synthesis Benchmark

## 1. Why Dedicated Swiss German Benchmarking Is Mandatory

Standard German (`de-DE`) and Google Cloud's `de-CH` locale are **not** equivalent to conversational Swiss German (`gsw` / Schwiizerdütsch):
1. **Unstandardized Orthography & Phonology:** Spoken Swiss German exhibits major vowel/diphthong shifts (*huus* vs *Haus*, *ziit* vs *Zeit*), consonantal shifts (*chuchichäschtli*), and regional lexical divergence across Valais (*Wallisertiitsch*), Bern (*Bärndütsch*), Zurich (*Züridütsch*), Basel (*Baseldytsch*), and Eastern Switzerland (*Ostschwiizerdütsch*).
2. **Silent Hochdeutsch Normalization Risk:** Standard ASR models trained to map Swiss speech into Standard German text silently erase dialect structure unless prompted and benchmarked for **dialect preservation**.
3. **Model Availability on Managed Cloud STT:** Google Cloud Speech-to-Text V2 provides `chirp_3` for `de-DE` and `fr-FR` on `eu-speech.googleapis.com`, whereas `de-CH` exposes `long` / `telephony` baselines that must be empirically compared against self-hosted Whisper variants.

---

## 2. Benchmark Protocol (`scripts/benchmark_swiss_german.py`)

### Dialects Covered
1. **`vs` — Wallisertiitsch (Valais / Leukerbad)**
2. **`zh` — Züridütsch (Zürich)**
3. **`be` — Bärndütsch (Bern)**
4. **`bs` — Baseldytsch (Basel)**
5. **`sg` — Ostschwiizerdütsch (St. Gallen)**

### Tracked Metrics
- **Word Error Rate (WER):** Token-level Levenshtein edit distance normalized by reference length (lower is better).
- **Dialect Preservation:** Verifies that Swiss German lexical/orthographic markers are preserved and never converted to German `ß` or purely Hochdeutsch phrasing.
- **Inference Latency (`ms`):** End-to-end CPU `int8` recognition latency after model warm-up.
- **Hardware & Model Provenance:** Exact CPU architecture, thread count, `CTranslate2` quantization (`int8`), and license metadata.

---

## 3. CPU-Only Hardware & Licensing Matrix

| Component | Candidate | Quantization / Runtime | License | Commercial Deployment Permitted? |
|---|---|---|---|---|
| Self-Hosted ASR | `faster-whisper` (`small` / `large-v3-turbo` / `large-v3`) | `CPU` + `int8` (`CTranslate2`) | `MIT` | **Yes** |
| Managed Baseline ASR | Google Cloud Speech-to-Text V2 (`de-CH` `long` / `telephony`) | Managed EU Endpoint (`eu-speech.googleapis.com`) | Google Cloud ToS | **Yes** |
| Swiss German Dialect TTS | `d41k4n/piper-swissdial` (`vs`, `zh`, `be`, `bs`, `sg`, `ag`, `gr`, `lu`) | `CPU` ONNX (`piper-tts`, `22,050 Hz`) | **`CC BY-NC 4.0`** | **No** (Non-commercial research/sandbox only; blocked by `Settings` when `APP_ENV=production`) |
| Commercial Fallback TTS | `rhasspy/piper-voices` (`de_DE-thorsten-medium`) / Google Cloud TTS | `CPU` ONNX / Managed EU Endpoint | `CC0` / GCP ToS | **Yes** |

---

## 4. Measured Live GCP Cloud Run Benchmark Results (`europe-west6`, CPU-Only)

Executed live against the deployed Zurich Cloud Run services (`swiss-voice-backend` + IAM-authenticated 4 vCPU `swiss-asr-service` running `faster-whisper` `small-int8-cpu` vs Google Cloud Speech-to-Text V2 `eu-speech.googleapis.com` `de-CH` `telephony` fallback) across all 5 mandatory SwissDial dialects (`22,050 Hz` `d41k4n/piper-swissdial` synthesis):

### Aggregate Summary

| Provider & Model | Endpoint / Hardware | Avg Dialect WER | Dialect Preservation Ratio | Avg Warm Latency (`ms`) |
|---|---|---:|---:|---:|
| **`faster-whisper` (`small-int8-cpu`)** | Cloud Run `europe-west6` (4 vCPU, 0 GPU) | **`0.7334`** | **`0.40`** | `2,990.9 ms` |
| **Google Cloud STT V2 (`de-CH` `telephony`)** | `eu-speech.googleapis.com` (EU Multi-Region) | `0.8065` | `0.40` | **`963.7 ms`** |

### Per-Dialect Breakdown

| Dialect | Reference Utterance | `faster-whisper` (`small-int8-cpu`) Hypothesis & WER | Google STT V2 (`de-CH`) Hypothesis & WER |
|---|---|---|---|
| **`vs`** (Wallisertiitsch / Leukerbad) | *Grüezi mitenand, wier sind hüt z Leukerbad und s Wätter isch güet.* | *"Tröt sie miteinander, wer ist in Schwiizerdütsch, Leukerbad und es Wetter ist, gut."* (`WER=0.9167`, `3,215 ms`, keeps *Leukerbad*) | *"Trotzdem miteinand, wir sind heute lecker geworden. Das Wetter ist gut."* (`WER=0.9167`, `877 ms`, misrecognizes *z Leukerbad*) |
| **`zh`** (Züridütsch / Zürich) | *Grüezi wohl, mir händ z Züri hüt vil z schaffe mit em Roboter.* | *"3 Zivolen, wir hänts Zürich und Wilt schaff'n mit dem Roboter."* (`WER=0.8462`, `2,963 ms`) | *"Grüezi wohl wir haben Zürich schaffen mit dem Roboter."* (`WER=0.6923`, `1,082 ms`) |
| **`be`** (Bärndütsch / Bern) | *Grüessech mitenand, ig ha nid gwüsst wie schnäu das geit.* | *"Grüeziich miteinander, ich hab nicht gewusst wie schnell das geht."* (`WER=0.8000`, `2,820 ms`) | *"30 miteinand, ich habe nicht gewusst, wie schnell das Date."* (`WER=0.8000`, `844 ms`) |
| **`bs`** (Baseldytsch / Basel) | *Sali zämme, mir luege hüt z Basel wie dr Assischtänt funktioniert.* | *"Salli zusammen, wir schauen jetzt, wie der Assistent funktioniert."* (`WER=0.8182`, `2,766 ms`) | *"Soll ich zusammen funktioniert?"* (`WER=0.9091`, `961 ms`, drops middle clause) |
| **`sg`** (Ostschwiizerdütsch / St. Gallen) | *Grüezi wohl, chasch du mir bitte säge wie s Wätter z Sankt Galle isch?* | *"Grüezi vor, kannst du mir bitte säge, wie s Wetter z Sankt Gali isch?"* (`WER=0.2857`, `3,191 ms`, preserves *säge*, *s*, *z*, *isch*) | *"Kannst du mir bitte sagen, wie das Wetter St. Gallen ist?"* (`WER=0.7143`, `1,055 ms`, normalizes to Hochdeutsch) |

### Key Empirical Findings
1. **Orthographic WER vs Semantic Preservation:** Because Swiss German lacks standardized orthography, generic `whisper-small` (`int8` CPU) partially normalizes pronouns/verbs (*ig ha nid* -> *ich hab nicht*, *mir luege* -> *wir schauen*), inflating strict dialect-orthography WER (`0.7334`) even when semantic meaning is preserved (`Basel` and `Bern` hypotheses are 100% intelligible to Gemini).
2. **Place-Name & Clause Retention:** `faster-whisper` (`small-int8-cpu`) outperformed Google Cloud STT `de-CH` overall (`0.7334` vs `0.8065` WER), notably retaining *Leukerbad* on Wallisertiitsch (`vs`), retaining the full clause on Baseldytsch (`bs`), and achieving `0.2857` WER with strong dialect retention on Ostschwiizerdütsch (`sg`).
3. **Drop-In Accuracy Upgrade Path:** `WHISPER_MODEL` is externally configurable without code changes (e.g., `large-v3-turbo` or a Swiss-German fine-tuned CTranslate2 checkpoint such as `Flurin17/whisper-large-v3-turbo-swiss-german`) when higher CPU RAM or GPU capacity is provisioned.

---

## 5. Reproducing Benchmarks Locally or Against GCP Sandbox

```bash
# 1. CPU-only local benchmark across all 5 dialects
.venv/Scripts/python.exe scripts/benchmark_swiss_german.py --whisper-model small

# 2. Side-by-side comparison including Google Cloud STT V2 de-CH (requires GCP_PROJECT_ID + ADC)
set GCP_PROJECT_ID=your-sandbox-project-id
.venv/Scripts/python.exe scripts/benchmark_swiss_german.py --whisper-model small --include-google-stt
```

Results are written automatically to `data/benchmarks/results/latest_benchmark.json` (gitignored to protect any speaker evaluation recordings).
