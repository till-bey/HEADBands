# Swiss Multilingual Voice & Robotics Assistant (Leukerbad Platform)

Real-time multilingual voice and embodied-robotics assistant built on **Google Cloud Platform (`europe-west6` — Zurich)** with **default-on automatic language detection (`auto`)** across **Swiss German (Schwiizerdütsch, including Wallisertiitsch / Valais)**, **Standard German (`de-DE` / `de-CH`)**, **French (`fr-FR` / `fr-CH`)**, **English (`en-US` / `en-GB`)**, **Spanish (`es-ES` / `es-US`)**, **Czech (`cs-CZ`)**, and **Mandarin Chinese (`zh-CN` / `zh-TW`)**, integrated with **Gemini** and **Gemini Robotics ER 2 + ROS 2**.

> **CPU-Only GCP Sandbox Ready:** Engineered to run with **zero GPUs** (`WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8` on a private Cloud Run CPU microservice in `europe-west6`, plus CPU ONNX `PiperVoice` TTS).

---

## Architecture Overview

```text
Browser / Microphone (16 kHz 16-bit Mono PCM)
        │  WebSocket (/ws/voice) + Barge-In Interruption + VAD
        ▼
Cloud Run: voice-backend (europe-west6, Zurich)
  ├── Language & Dialect Router (auto [default] | gsw | de-CH | de-DE | fr-CH | fr-FR | en-US | en-GB | es-ES | es-US | cs-CZ | zh-CN | zh-TW)
  ├── STT Providers (ASRProvider interface)
  │     ├── Auto / DE / FR / EN / ES / CS / ZH ──► Google Cloud Speech-to-Text V2 (Chirp 3 auto-detect, EU endpoint)
  │     └── Swiss German                       ──► Private Cloud Run CPU Service: asr-service
  │                                                (faster-whisper int8 CPU, europe-west6) + Google de-CH baseline
  ├── Assistant Orchestrator (LLMProvider interface)
  │     └── Vertex AI Gemini (google-genai SDK, streaming + function calling)
  ├── Embodied Robotics Reasoning (RoboticsReasoningProvider interface)
  │     └── Gemini Robotics ER 2 (gemini-robotics-er-2-preview, global endpoint)
  │           ├── On-Demand Camera Trigger (/camera/color/image_raw, Orbbec Gemini 2 auto-launch)
  │           └── Deterministic Safety Validator + Human Confirmation Gate
  │                 └── Outbound Authenticated WS (/ws/robot-bridge) ──► UniX AI Panther ROS 2 Bridge (CycloneDDS, Domain 15)
  └── TTS Providers (TTSProvider interface, sentence-chunked streaming)
        ├── DE / FR / EN / ES / CS / ZH ──► Google Cloud Text-to-Speech (Chirp 3 HD / Neural2 / WaveNet, EU endpoint)
        └── Swiss German                ──► Piper ONNX CPU (d41k4n/piper-swissdial: vs, zh, be, bs, sg, ag, gr, lu)
```

---

## Regional Data Processing & Licensing Disclosures

1. **Infrastructure Residency (`europe-west6` Zurich):** Cloud Run (`voice-backend` and `asr-service`), Artifact Registry, Firestore, Cloud Storage, Pub/Sub, and Secret Manager deploy directly in `europe-west6` (Zurich, Switzerland).
2. **Managed Speech APIs (`eu` multi-region):** Google Cloud Speech-to-Text V2 (`eu-speech.googleapis.com`) and Text-to-Speech (`eu-texttospeech.googleapis.com`) process data within the **European Union multi-region**, not Switzerland-only.
3. **Vertex AI Gemini (`GCP_LLM_LOCATION`, default `global` for `gemini-3-flash-preview`):** `gemini-3-flash-preview` (`ThinkingLevel.LOW` for low-latency voice turns) and `gemini-3.1-pro-preview` run on Vertex AI's `global` endpoint in the GCP sandbox (`GCP_LLM_LOCATION` remains independently configurable to regional EU hubs when regional GA models are enabled).
4. **Gemini Robotics ER 2 (`global`):** `gemini-robotics-er-2-preview` uses Google's `global` endpoint and does not guarantee Switzerland-only processing.
5. **SwissDial Piper TTS License (`CC BY-NC 4.0`):** The `d41k4n/piper-swissdial` voice models are licensed under **Creative Commons Attribution-NonCommercial 4.0**. Startup validation blocks these models when `APP_ENV=production` unless replaced by a commercially licensed checkpoint (`PIPER_MODEL_LICENSE=commercial`).

---

## Quickstart (Local Development)

```bash
# 1. Create Python 3.12 virtualenv and install backend + test dependencies
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e "backend[whisper,dev]"

# 2. Copy configuration template
copy .env.example .env

# 3. Run automated test suite
pytest backend/tests

# 4. Start backend locally (http://localhost:8080)
cd backend
uvicorn app.main:app --reload --port 8080
```

See:
- [docs/architecture.md](docs/architecture.md) — Detailed system & regional residency architecture
- [docs/speech-benchmarks.md](docs/speech-benchmarks.md) — Swiss German ASR benchmark methodology & results
- [docs/robotics-safety.md](docs/robotics-safety.md) — Mandatory 10-rule ROS 2 safety architecture
- [docs/deployment.md](docs/deployment.md) — Step-by-step GCP Sandbox & Terraform deployment guide
