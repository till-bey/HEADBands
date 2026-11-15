# Swiss Multilingual Voice & Robotics Assistant

> Project brief and implementation instructions for OpenAI Codex

**Status:** Architecture / initial implementation  
**Primary cloud:** Google Cloud Platform (GCP)  
**Primary deployment region:** `europe-west6` (Zurich, Switzerland)  
**Languages:** Swiss German, Standard German and French  
**Robotics:** Gemini Robotics ER 2 + ROS 2  
**Main goals:** Low latency, modular architecture, low operating costs, open-source models where practical, and pay-as-you-go cloud services.

---

## 1. Project overview

Build a real-time, multilingual voice assistant that can communicate naturally in:

1. Swiss German (Schweizerdeutsch / Schwiizerdütsch).
2. Standard German (Hochdeutsch).
3. French (Français).

The assistant should understand spoken language, generate responses, speak naturally and eventually interact with a physical robot.

The entire system should use Google Cloud wherever practical. Open-source models should be preferred for capabilities that Google Cloud does not cover adequately, especially Swiss German speech recognition and dialect speech synthesis.

The system should be modular so that speech recognition, TTS, LLMs and robotics models can be changed without rewriting the application.

### Main design principles

- Google Cloud first.
- Prefer usage-based services over long-term subscriptions.
- Prefer open-source and self-hostable models where quality is sufficient.
- Support streaming audio and low-latency conversation.
- Support language and dialect switching during a conversation.
- Keep robot control separate from language-model reasoning.
- Never let an LLM directly control motors or safety-critical hardware.
- Make all external model integrations replaceable through interfaces.
- Keep personal audio data private and avoid storing recordings by default.

---

## 2. Recommended architecture

```text
                         USER
                          |
                    Microphone / UI
                          |
                    WebSocket Audio
                          |
                          v
                GOOGLE CLOUD RUN
                Voice Gateway / API
                          |
                Language Detection
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
          Swiss DE     Standard DE   French
              |           |           |
              v           v           v
         Local ASR     Google STT   Google STT
         on GCE GPU    Chirp 3      Chirp 3
              |           |           |
              +-----------+-----------+
                          |
                    Transcription
                          |
                          v
                   GEMINI ASSISTANT
                   Vertex / Agent Platform
                          |
                 Conversation / Tools
                          |
              +-----------+-----------+
              |                       |
              v                       v
         Normal reply            Robot reasoning
              |                 Gemini Robotics ER 2
              |                       |
              |                 Structured proposal
              |                       |
              |                       v
              |                ROS 2 Safety Gateway
              |                       |
              |                       v
              |                    ROBOT
              |
         Language-aware TTS
              |
       +------+-------+
       |              |
       v              v
    Google TTS     Piper SwissDial
    German/French  Swiss German
       |              |
       +------+-------+
              |
         Audio response
              |
            USER
```

### GCP infrastructure

Use the following services:

| Component | Technology | Purpose |
|---|---|---|
| API / backend | Cloud Run | Main assistant backend |
| Audio transport | WebSockets | Real-time audio communication |
| STT German/French | Cloud Speech-to-Text V2 | Speech recognition |
| STT Swiss German | faster-whisper / other benchmarked model | Swiss German recognition |
| STT hosting | Compute Engine G2 | GPU inference |
| TTS German/French | Cloud Text-to-Speech | Voice generation |
| TTS Swiss German | Piper-compatible model | Dialect voice generation |
| Main LLM | Gemini through Google Cloud | Conversation and reasoning |
| Robotics reasoning | Gemini Robotics ER 2 | Visual/spatial reasoning |
| Cloud storage | Cloud Storage | Optional audio and dataset storage |
| Database | Firestore | Conversations and configuration |
| Messaging | Pub/Sub | Robot telemetry and asynchronous events |
| Secrets | Secret Manager | API credentials |
| Container registry | Artifact Registry | Docker images |
| CI/CD | Cloud Build | Build and deployment |
| Monitoring | Cloud Logging / Monitoring | Observability |

Avoid Kubernetes initially. Cloud Run plus one GPU-enabled Compute Engine instance should be sufficient for the first prototype.

---

## 3. GCP configuration

### Default region

```env
GCP_PROJECT_ID=your-project-id
GCP_REGION=europe-west6
GCP_LOCATION=europe-west6
GCP_STT_LOCATION=eu
GCP_TTS_LOCATION=eu
```

Use Zurich (`europe-west6`) for infrastructure wherever supported.

Important: Not every managed AI endpoint supports Swiss-only processing. In particular, the Speech-to-Text EU endpoint and Chirp 3 HD EU endpoint are European multi-region services, not guarantees of Switzerland-only processing. Robotics ER 2 currently has a global model endpoint. Document these exceptions clearly.

### Compute Engine

Use a G2 instance with an NVIDIA L4 GPU for the self-hosted speech recognition service.

Preferred zones:

```text
europe-west6-b
europe-west6-c
```

The GPU service should:

- Run as a Docker container.
- Expose a private inference API.
- Support streaming audio where possible.
- Load the ASR model once and keep it available.
- Be independently deployable from the main API.
- Be configurable so it can be disabled or replaced.

Do not use Cloud Run GPU for the initial Swiss-hosted ASR deployment because its published GPU region availability does not include Zurich.

---

## 4. Speech recognition (STT)

This is one of the most important components.

The application must distinguish Swiss German from Standard German. `de-CH` is not equivalent to unrestricted recognition of conversational Swiss German.

### Standard German

Use Google Cloud Speech-to-Text V2.

Preferred configuration:

```env
STT_STANDARD_DE_PROVIDER=google
STT_STANDARD_DE_LANGUAGE=de-DE
STT_STANDARD_DE_MODEL=chirp_3
```

Use the EU endpoint for Chirp 3 where supported.

### French

Use Google Cloud Speech-to-Text V2.

```env
STT_FR_PROVIDER=google
STT_FR_LANGUAGE=fr-FR
STT_FR_MODEL=chirp_3
```

Support French (Switzerland) as an optional locale:

```text
fr-CH
```

Test both French (France) and French (Switzerland). Do not assume that a Swiss locale always has the same model availability as `fr-FR`.

### Swiss German

Test several options instead of assuming that one model is best.

Preferred starting candidates:

1. faster-whisper with Whisper large-v3.
2. Whisper variants fine-tuned for Swiss German, subject to model-card and licence checks.
3. Google Speech-to-Text V2 `de-CH` as a baseline.
4. Other Swiss German ASR models from Hugging Face, if licences and quality are acceptable.

The Google `de-CH` locale currently lists telephony and telephony-short models, not Chirp 3. They must be benchmarked on the actual microphone recordings and conversation use case.

Create a configurable ASR provider interface.

Example:

```python
class ASRProvider:
    async def transcribe(
        self,
        audio: bytes,
        language: str,
    ) -> "Transcription":
        raise NotImplementedError
```

Implement at least:

```text
GoogleASRProvider
FasterWhisperProvider
```

The interface must allow further models to be added later.

### Language handling

Support these language settings:

```text
auto
gsw
de-CH
de-DE
fr-CH
fr-FR
```

`gsw` is an application-level identifier for Swiss German. It is not a claim that every provider supports that language code.

Allow both manual language selection and automatic language detection.

For Swiss German, preserve the original dialect transcription. If a normalized Standard German version is useful for reasoning, store it separately rather than replacing the original.

---

## 5. Text-to-speech (TTS)

### Standard German

Use Google Cloud Text-to-Speech.

Preferred locale:

```text
de-DE
```

Test available Chirp 3 HD and Neural2 voices.

### French

Use Google Cloud Text-to-Speech.

Preferred locale:

```text
fr-FR
```

Test available Chirp 3 HD, Studio and Neural2 voices.

### Swiss German

Google Cloud does not currently list a dedicated Swiss German (`de-CH`) dialect TTS voice in its standard voice catalogue.

Use a provider interface and benchmark Piper-compatible Swiss German voices.

Candidate:

```text
d41k4n/piper-swissdial
```

It includes eight dialect variants:

- Aargau
- Bern
- Basel
- Graubünden
- Luzern
- St. Gallen
- Valais
- Zürich

For example, the Valais voice is:

```text
de_CH-swissdial_vs-medium
```

Important licensing restriction:

The published SwissDial Piper models use CC BY-NC 4.0. This is a non-commercial licence. Do not use these models in a commercial deployment without obtaining appropriate permissions or replacing them with commercially licensed alternatives.

Create a TTS interface:

```python
class TTSProvider:
    async def synthesize(
        self,
        text: str,
        language: str,
        voice: str | None = None,
    ) -> bytes:
        raise NotImplementedError
```

Implement:

```text
GoogleTTSProvider
PiperTTSProvider
```

The TTS system should support:

- Voice selection.
- Language selection.
- Dialect selection.
- Audio streaming or sentence-level chunking.
- Interruptions and cancellation.
- WAV or PCM audio output.
- Configurable speaking speed where supported.

When the user speaks Swiss German, the assistant should reply in the same dialect when a suitable voice is configured.

Do not silently convert every Swiss German response into Standard German.

---

## 6. Main assistant and LLM

Use Google Gemini as the main reasoning and conversation engine.

Preferred hosting and authentication: Google Cloud / Vertex AI or the current Google Cloud Gemini Enterprise Agent Platform APIs.

The assistant should support:

- Multilingual conversation.
- Context retention during a session.
- Function calling.
- Structured tool outputs.
- Language-aware responses.
- Robot capability discovery.
- Conversation interruption.
- Streaming text responses.

The LLM must not be responsible for low-level robot safety.

Keep conversation orchestration separate from the speech recognition and TTS implementations.

Create an interface:

```python
class LLMProvider:
    async def generate_response(
        self,
        messages: list,
        tools: list | None = None,
    ) -> "AssistantResponse":
        raise NotImplementedError
```

Use environment configuration for the selected Gemini model and location. Never hardcode preview model identifiers throughout the code.

---

## 7. Robotics integration: Gemini Robotics ER 2

The robotics target is **Gemini Robotics ER 2**, Google's embodied reasoning model.

Model identifier:

```text
gemini-robotics-er-2-preview
```

Use the Google Cloud Gemini Enterprise Agent Platform integration where possible.

The model is currently a preview offering. Its documented model endpoint is global, so do not promise Swiss-only processing for ER 2 requests.

Use ER 2 for high-level reasoning tasks such as:

- Understanding camera images.
- Interpreting the robot's environment.
- Spatial reasoning.
- Identifying objects.
- Understanding visual task progress.
- Generating structured task proposals.
- Orchestrating allowed robot tools.

Do not mistake ER 2 for a low-level motor controller.

### ROS 2 integration

Build a separate Python ROS 2 bridge.

The bridge should run on the robot or a local computer connected to the robot.

Responsibilities:

- Receive validated action requests.
- Publish allowed ROS 2 commands.
- Return robot status.
- Publish telemetry.
- Report action completion or failure.
- Enforce local limits independently of cloud availability.

Recommended logical structure:

```text
Gemini Robotics ER 2
          |
          v
   Robot Action Planner
          |
          v
   Structured Action API
          |
          v
    Safety Validator
          |
          v
     ROS 2 Bridge
          |
          v
        Robot
```

### Safety rules

These are mandatory:

1. Never execute arbitrary code generated by an LLM.
2. Never allow direct arbitrary ROS topic publishing from the LLM.
3. Only allow explicitly defined robot actions.
4. Validate all incoming action parameters.
5. Enforce local speed, position and movement limits.
6. Require human confirmation for potentially dangerous actions.
7. Implement timeouts and command expiration.
8. Implement a watchdog.
9. Keep emergency stop local and independent of cloud connectivity.
10. Reject commands whenever the robot is in an unsafe state.

The robot must remain safe when the cloud API, internet connection or LLM is unavailable.

### Example action schema

```json
{
  "action": "move_to",
  "target": {
    "x": 0.5,
    "y": 0.2,
    "z": 0.0
  },
  "speed": 0.1,
  "requires_confirmation": true,
  "command_id": "unique-id"
}
```

This is an example application-level schema, not a direct ROS message or a model-specific API contract.

All actions must be converted and validated locally before execution.

Use Pub/Sub for asynchronous telemetry and status updates. For interactive command delivery, prefer a persistent authenticated outbound connection from the robot bridge to the cloud backend.

Do not expose the robot's ROS 2 network directly to the public internet.

---

## 8. Backend architecture

Use:

```text
Python 3.12+
FastAPI
WebSockets
Pydantic
Google Cloud Python SDKs
Google Gen AI SDK
```

The main backend should be divided into independent services/modules.

Suggested modules:

```text
backend/
  app/
    main.py

    api/
      voice.py
      conversation.py
      robotics.py
      health.py

    core/
      config.py
      logging.py
      security.py

    speech/
      asr/
        base.py
        google_stt.py
        faster_whisper.py

      tts/
        base.py
        google_tts.py
        piper.py

      language/
        detection.py
        routing.py
        normalization.py

    assistant/
      orchestrator.py
      prompts.py
      memory.py
      tools.py

    robotics/
      er2_client.py
      action_schema.py
      action_validator.py
      robot_gateway.py

    infrastructure/
      storage.py
      firestore.py
      pubsub.py

  tests/
    unit/
    integration/
    speech/
    robotics/
```

The frontend should be a separate application and must not contain cloud credentials.

---

## 9. Suggested repository structure

```text
.
├── README.md
├── .env.example
├── .gitignore
├── docker-compose.yml
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── app/
│
├── frontend/
│   └── README.md
│
├── robot_bridge/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── ros2_ws/
│
├── infrastructure/
│   └── terraform/
│
├── scripts/
│   ├── setup_gcp.sh
│   ├── deploy.sh
│   └── download_models.sh
│
├── data/
│   └── README.md
│
├── tests/
│
└── docs/
    ├── architecture.md
    ├── speech-benchmarks.md
    ├── robotics-safety.md
    └── deployment.md
```

Do not commit:

- Audio recordings.
- Model weights.
- API keys.
- Service account private keys.
- Personal information.
- Production credentials.

---

## 10. Configuration

Create an `.env.example` file.

Example:

```env
# General
APP_ENV=development
LOG_LEVEL=INFO

# GCP
GCP_PROJECT_ID=
GCP_REGION=europe-west6
GCP_STT_LOCATION=eu
GCP_TTS_LOCATION=eu

# STT
STT_DEFAULT_PROVIDER=google
STT_SWISS_GERMAN_PROVIDER=faster_whisper
STT_STANDARD_DE_MODEL=chirp_3
STT_FRENCH_MODEL=chirp_3

# Swiss German ASR
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
ASR_ENDPOINT=

# TTS
TTS_DEFAULT_PROVIDER=google
TTS_SWISS_GERMAN_PROVIDER=piper

PIPER_MODEL_PATH=
PIPER_CONFIG_PATH=

GOOGLE_TTS_LANGUAGE_DE=de-DE
GOOGLE_TTS_LANGUAGE_FR=fr-FR

GOOGLE_TTS_VOICE_DE=
GOOGLE_TTS_VOICE_FR=

# LLM
LLM_PROVIDER=google
GEMINI_MODEL=

# Robotics
ROBOTICS_ENABLED=false
ROBOTICS_PROVIDER=gemini_robotics_er
ROBOT_GATEWAY_URL=
ROBOT_REQUIRE_CONFIRMATION=true

# Storage
AUDIO_STORAGE_ENABLED=false
GCS_AUDIO_BUCKET=

# Database
FIRESTORE_ENABLED=false
```

All settings must be validated at application startup.

Never put secrets in `.env.example`. In production, use Secret Manager and service identities.

---

## 11. Audio processing requirements

The application must support real-time or near-real-time voice interactions.

Initial audio format:

```text
PCM
16 kHz
16-bit
Mono
```

Implement:

- Microphone capture.
- Voice activity detection.
- Audio buffering.
- Streaming recognition where supported.
- Partial transcription.
- End-of-utterance detection.
- Streaming or chunked TTS.
- Playback interruption.
- Cancellation of pending generation.

Avoid writing complete audio files to Cloud Storage by default.

Keep audio in memory while processing unless explicit recording is enabled.

When recording is enabled, support automatic deletion and configurable retention.

---

## 12. Swiss German evaluation

Do not assume that a model works well for Swiss German because it understands Standard German.

Create a dedicated benchmark.

Test at least these dialects:

- Wallisertiitsch (Valais).
- Züridütsch.
- Bärndütsch.
- Baseldytsch.
- Ostschwiizerdütsch.

Use real recordings from consenting speakers.

Evaluate:

| Metric | Goal |
|---|---|
| Word error rate | Lower is better |
| Sentence understanding | Correct interpretation |
| Dialect preservation | Avoid unwanted conversion |
| Latency | Low conversational delay |
| Noise robustness | Usable in real environments |
| Accent robustness | Multiple speakers |
| Proper nouns | Recognize Swiss names and places |

Keep a benchmark dataset containing:

```text
audio/
transcripts/
metadata/
results/
```

Each recording should have a corresponding verified transcript and dialect label.

Compare:

1. Google STT `de-CH`.
2. Whisper large-v3.
3. Swiss German fine-tuned ASR candidates.

Record the exact model version, language configuration, hardware and latency for each test.

For commercial use, check the licence of the model, its training data and every voice dataset separately.

---

## 13. Security and privacy

Privacy should be a core part of the implementation.

Requirements:

- GCP IAM and least-privilege service accounts.
- Secret Manager for credentials.
- HTTPS and secure WebSockets.
- Authenticated robot connections.
- No credentials in frontend code.
- No raw audio in standard application logs.
- Optional audio storage only.
- Configurable retention periods.
- Restricted access to stored recordings.
- Clear separation between development and production environments.

Use Zurich resources for application hosting and storage where supported.

Do not claim that every managed API processes data exclusively in Switzerland. Document the region and processing characteristics of each external service independently.

---

## 14. Implementation roadmap

Implement this project incrementally. A working voice assistant is more valuable than building every component simultaneously.

### Phase 1 — Basic infrastructure

- Create the Python backend.
- Create a GCP project configuration.
- Set up FastAPI.
- Configure Cloud Run.
- Add health-check endpoints.
- Add structured logging.
- Add local Docker development.

**Deliverable:** Backend deployable locally and on Cloud Run.

### Phase 2 — Standard German and French

- Integrate Google Speech-to-Text V2.
- Integrate Google Text-to-Speech.
- Add audio upload and WebSocket endpoints.
- Add Gemini conversation orchestration.
- Implement basic language selection.

**Deliverable:** A working voice assistant in German and French.

### Phase 3 — Swiss German

- Set up a Zurich G2 GPU instance.
- Deploy faster-whisper.
- Benchmark Swiss German recognition.
- Add the Piper TTS adapter.
- Test SwissDial voices where licensing permits.
- Add dialect selection and preservation.

**Deliverable:** A multilingual assistant with a demonstrable Swiss German pipeline.

### Phase 4 — Conversational improvements

- Add interruption handling.
- Add voice activity detection.
- Improve language switching.
- Add conversation context.
- Optimize audio streaming and response latency.

**Deliverable:** Natural voice conversations.

### Phase 5 — Robotics

- Create the ROS 2 bridge.
- Define the allowed robot action schema.
- Implement local action validation.
- Integrate Gemini Robotics ER 2.
- Add visual inputs.
- Add human confirmation.
- Add telemetry.

**Deliverable:** Assistant can understand a robot-related request, reason about visual input and safely propose allowed actions.

### Phase 6 — Production readiness

- Add Terraform infrastructure.
- Add Cloud Build.
- Add automated tests.
- Add monitoring and alerts.
- Add budget controls.
- Review security and model licences.

**Deliverable:** Reproducible and documented deployment.

---

## 15. Definition of done

The initial project is successful when:

- [ ] Backend runs locally using Docker.
- [ ] Backend deploys to Cloud Run.
- [ ] User can speak Standard German.
- [ ] User can speak French.
- [ ] User can speak at least one Swiss German dialect.
- [ ] User can switch languages during a conversation.
- [ ] The assistant can answer in the selected language.
- [ ] TTS produces playable audio.
- [ ] Speech recognition supports streaming or low-latency interaction.
- [ ] Swiss German ASR has a documented benchmark.
- [ ] Providers can be switched through configuration.
- [ ] The assistant can receive a robot image.
- [ ] Robotics reasoning can return structured proposals.
- [ ] ROS 2 commands pass through local safety validation.
- [ ] Emergency stop works independently of the cloud.
- [ ] No secrets are committed to the repository.

---

## 16. Instructions for Codex

You are building this project as a production-oriented but initially minimal prototype.

Follow these rules:

1. First inspect the existing repository before creating or changing files.
2. Do not overwrite existing functionality without checking it.
3. Follow the architecture in this README unless there is a clear technical reason not to.
4. Prefer simple, maintainable solutions over unnecessary abstractions.
5. Use Python type hints and Pydantic models.
6. Keep services modular and independently testable.
7. Use asynchronous I/O for API and audio operations where appropriate.
8. Put all secrets and deployment-specific settings in environment variables or Google Secret Manager.
9. Do not invent model IDs, cloud APIs, ROS 2 topics or robot capabilities.
10. Verify current official documentation before implementing an external API integration.
11. Check model licences before downloading or deploying model weights.
12. Do not use non-commercial models in a commercial deployment without appropriate permission.
13. Do not implement actual robot motor control before the safety gateway and action validation have been tested.
14. Write unit tests for language routing, provider selection, tool validation and robot safety.
15. Document all important implementation decisions.

### Work incrementally

Do not attempt to implement the entire architecture in one step.

Start with:

**Phase 1: Backend foundation and GCP deployment.**

Once this is working, continue with Phase 2.

After each phase:

- Run tests.
- Fix errors.
- Update documentation.
- Report which parts work.
- Identify missing credentials or external hardware dependencies.
- Do not claim a feature works until it has actually been tested.

If external credentials or a robot are unavailable, implement an interface and a mock provider instead of pretending the real integration works.

### Final priority

The priority order is:

1. Reliable voice communication.
2. Genuine Swiss German understanding.
3. Natural multilingual speech generation.
4. Fast response times.
5. Reliable Gemini integration.
6. Safe robotics integration.

The final product should feel like one intelligent assistant, not three independent speech systems connected together.

---

## 17. Official documentation and model references

Use these sources as the starting point. Always recheck model availability, API versions, pricing and licensing before deployment.

### Google Cloud

- [Google Cloud Speech-to-Text V2](https://cloud.google.com/speech-to-text/v2/docs)
- [Supported STT languages and models](https://cloud.google.com/speech-to-text/docs/speech-to-text-supported-languages)
- [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech/docs)
- [TTS supported voices](https://cloud.google.com/text-to-speech/docs/list-voices-and-types)
- [TTS regional endpoints](https://cloud.google.com/text-to-speech/docs/endpoints)
- [Cloud Run](https://cloud.google.com/run/docs)
- [Compute Engine GPU locations](https://cloud.google.com/compute/docs/regions-zones/gpu-regions-zones)
- [Vertex AI / Gemini Enterprise Agent Platform](https://cloud.google.com/vertex-ai/generative-ai/docs/overview)

### Robotics

- [Gemini Robotics ER 2](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/robotics-er-2)
- [Gemini Robotics ER documentation](https://ai.google.dev/gemini-api/docs/robotics-overview)
- [ROS 2 documentation](https://docs.ros.org/)

### Open-source speech

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Whisper](https://github.com/openai/whisper)
- [Piper](https://github.com/OHF-Voice/piper1-gpl)
- [Piper SwissDial voices](https://huggingface.co/d41k4n/piper-swissdial)

**Documentation baseline checked:** 14 November 2026.

**Important:** SwissDial Piper models are currently published under CC BY-NC 4.0. Do not treat them as commercially unrestricted.