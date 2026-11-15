# Architecture & Regional Data Processing Matrix

## 1. Design Adaptation for CPU-Only GCP Sandbox (No GPUs)

The initial brief (`readme.txt`) proposed a Compute Engine `G2` NVIDIA L4 GPU instance for `faster-whisper` (`WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`). Because the target GCP sandbox provides **no GPU quota**, the platform makes two clean architectural adaptations while keeping all provider interfaces 100% compatible with future GPU instances:

1. **Dedicated Serverless CPU `int8` ASR Service (`swiss-asr-service` on Cloud Run in `europe-west6`):**
   - Runs `faster-whisper` (`CTranslate2`) with `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8` across 4 vCPUs.
   - Deployed directly in **Zurich (`europe-west6`)** — whereas Cloud Run GPU is unavailable in Zurich, Cloud Run CPU runs natively in `europe-west6` with scale-to-zero pay-as-you-go billing!
   - Protected by Cloud Run IAM (`--no-allow-unauthenticated`); `swiss-voice-backend` authenticates service-to-service using Google-signed OIDC ID tokens (`mint_cloud_run_id_token`).
   -Startup validation in `Settings` explicitly rejects invalid `WHISPER_DEVICE=cpu` + `WHISPER_COMPUTE_TYPE=float16` combinations.
2. **CPU ONNX Swiss German TTS (`PiperTTSProvider`):**
   - Uses `piper-tts` (`ONNX Runtime` CPU) with `d41k4n/piper-swissdial` (`22,050 Hz` medium VITS checkpoints across all 8 Swiss dialects, defaulting to `vs` — Wallisertiitsch for Leukerbad).
   - Synthesizes speech in `< 50 ms` per sentence on 2 vCPUs without any GPU.

---

## 2. Independent Regional & Data-Residency Matrix

Not every managed Google Cloud AI API processes requests exclusively inside Switzerland. Each external dependency is documented below:

| Component | Service / Model | Configured Endpoint / Region | Residency Guarantee |
|---|---|---|---|
| Main Voice & Robotics Gateway | Cloud Run (`swiss-voice-backend`) | `europe-west6` (Zurich) | **Switzerland (`europe-west6`)** |
| Self-Hosted Swiss German ASR | Cloud Run (`swiss-asr-service`, CPU `int8`) | `europe-west6` (Zurich) | **Switzerland (`europe-west6`)** |
| Container Images | Artifact Registry (`swiss-voice-repo`) | `europe-west6` (Zurich) | **Switzerland (`europe-west6`)** |
| Secrets | Secret Manager | `europe-west6` user-managed replica | **Switzerland (`europe-west6`)** |
| Robot Telemetry Messaging | Cloud Pub/Sub (`robot-telemetry`) | `europe-west6` storage policy | **Switzerland (`europe-west6`)** |
| Optional Session DB | Firestore Native | `europe-west6` (Zurich) | **Switzerland (`europe-west6`)** |
| Optional Audio Storage | Cloud Storage (`AUDIO_STORAGE_ENABLED=false`) | `europe-west6` (Zurich) | **Switzerland (`europe-west6`)** |
| Managed STT (`auto`, `de-DE`, `fr-FR`, `en-US`, `es-ES`, `es-US`, `cs-CZ`, `cmn-Hans-CN`, `de-CH`) | Cloud Speech-to-Text V2 (`chirp_3` `auto` / `long`) | `eu-speech.googleapis.com` | **EU Multi-Region** (not Switzerland-only) |
| Managed TTS (`de-DE`, `fr-FR`, `en-US`, `en-GB`, `es-ES`, `es-US`, `cs-CZ`, `cmn-CN`, `cmn-TW`) | Cloud Text-to-Speech (`Chirp3-HD` / `Neural2` / `WaveNet`) | `eu-texttospeech.googleapis.com` | **EU Multi-Region** (not Switzerland-only) |
| Conversational LLM | Vertex AI Gemini (`gemini-3-flash-preview`) | `GCP_LLM_LOCATION=global` (independently configurable) | **Global Endpoint** (`gemini-3-flash-preview` preview offering; switchable to EU regional endpoints via `GCP_LLM_LOCATION` + `GEMINI_MODEL`) |
| Embodied Robotics Reasoning | Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) | `global` | **Global Endpoint** (preview offering; not Switzerland-only) |

---

## 3. Replaceable Provider Interfaces

All external AI integrations sit behind clean async Python interfaces:
- `ASRProvider` (`backend/app/speech/asr/base.py`): `GoogleASRProvider`, `FasterWhisperProvider`, `MockASRProvider`
- `TTSProvider` (`backend/app/speech/tts/base.py`): `GoogleTTSProvider`, `PiperTTSProvider`, `MockTTSProvider`
- `LLMProvider` (`backend/app/assistant/orchestrator.py`): `GeminiLLMProvider`, `MockLLMProvider`
- `RoboticsReasoningProvider` (`backend/app/robotics/er2_client.py`): `GeminiRoboticsER2Client`, `MockRoboticsER2Client`

---

## 4. Privacy & Audio Handling

- Raw microphone audio (`16 kHz`, `16-bit`, `mono` PCM) is processed strictly **in memory** (`StreamingAudioBuffer`).
- `AUDIO_STORAGE_ENABLED` defaults to `false`. When explicitly enabled, objects carry retention metadata and GCS bucket lifecycle rules automatically delete objects after `AUDIO_RETENTION_DAYS`.
- `CloudLoggingJSONFormatter` (`backend/app/core/logging.py`) automatically scrubs binary audio buffers, tokens, and API keys before emitting structured Cloud Logging JSON.

---

## 5. Two-Tier Voice & Embodied Robotics Pipeline (`Gemini Flash` + `Gemini Robotics ER 2`)

To achieve sub-second voice responsiveness while supporting arbitrary multi-step robotics instructions across all 7 language families (`en`, `fr`, `de`, `gsw`, `es`, `cs`, `zh`):

1. **System 1 — `Gemini Flash` (`gemini-3-flash-preview`, `ThinkingLevel.MINIMAL`):**
   - Handles fast conversational turns, cross-language translation, status checks (`get_robot_status`), and immediate Emergency Stop (`emergency_stop_robot`).
   - Whenever the user gives any physical robot instruction in any language/dialect, `Gemini Flash` emits `plan_robotics_task_with_er2(english_instruction=..., quick_spoken_ack=...)`.
   - Over `/ws/voice`, `quick_spoken_ack` (e.g., *"Yeah sure, on it!"*, *"Bien sûr !"*, *"Ja sicher!"*, *"¡Claro, enseguida!"*, *"Jasně, hned to naplánuji!"*, *"好的，没问题！"*) is streamed immediately with concurrent TTS audio while System 2 plans in parallel.
2. **System 2 — `Gemini Robotics ER 2` (`gemini-robotics-er-2-preview`, `ThinkingLevel.LOW`):**
   - Receives the translated `english_instruction`, robot telemetry, and optional camera frame, decomposing single- and multi-step instructions (e.g., *"Move forward 5 m and turn around."* -> `[move_to(x=5.0m), rotate(yaw_rad=3.14159)]`) into ordered high-level proposals.
   - Every proposal is validated by `ActionSafetyValidator` and queued in `RobotSafetyGateway` awaiting mandatory human confirmation.
