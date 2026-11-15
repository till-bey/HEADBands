# Deployment Guide (Local, Docker, Cloud Run & Terraform — CPU-Only Sandbox)

## 1. Prerequisites

- **Google Cloud Project** (GCP Sandbox with Billing enabled; **no GPUs required**).
- **Google Cloud SDK (`gcloud`)** authenticated:
  ```powershell
  gcloud auth login
  gcloud auth application-default login
  gcloud config set project YOUR_SANDBOX_PROJECT_ID
  ```
- **Python 3.12+** (for local execution and test suite).

---

## 2. Automated Deployment via Scripts (Recommended for GCP Sandbox)

Cloud Build compiles the Docker images remotely in GCP, so **local Docker Desktop is not even required** for cloud deployment:

### Windows PowerShell
```powershell
# Step 1: Enable GCP APIs, create Zurich Artifact Registry, Service Accounts, IAM, and Pub/Sub
.\scripts\setup_gcp.ps1 -ProjectId "YOUR_SANDBOX_PROJECT_ID" -Region "europe-west6"

# Step 2: Build images in Cloud Build & deploy private CPU ASR + main Voice Backend to Cloud Run (europe-west6)
.\scripts\deploy.ps1 -ProjectId "YOUR_SANDBOX_PROJECT_ID" -Region "europe-west6" -LlmLocation "global" -GeminiModel "gemini-3-flash-preview"
```

### Linux / macOS / Cloud Shell
```bash
export GCP_PROJECT_ID="YOUR_SANDBOX_PROJECT_ID"
bash scripts/setup_gcp.sh
bash scripts/deploy.sh
```

---

## 3. Infrastructure-as-Code Deployment (Terraform)

After building images once via `gcloud builds submit --config=cloudbuild.yaml .`:

```bash
cd infrastructure/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project_id and image URIs

terraform init
terraform plan
terraform apply
```

Terraform provisions:
- Least-privilege Service Accounts (`swiss-voice-backend-sa`, `swiss-asr-cpu-sa`)
- Zurich Artifact Registry (`swiss-voice-repo`)
- Zurich Secret Manager secret (`robot-bridge-shared-token`)
- Zurich Pub/Sub topic & subscription (`robot-telemetry`)
- Private IAM-protected Cloud Run service `swiss-asr-service` (`europe-west6`, 4 vCPU, 4 GiB RAM, `faster-whisper` `int8` CPU)
- Public/IAM Cloud Run service `swiss-voice-backend` (`europe-west6`, session affinity + 3600s WebSocket timeout)
- Cloud Monitoring HTTPS health uptime check & optional Cloud Billing monthly budget alert

---

## 4. Local Docker Compose Development

```bash
docker compose up --build
```
- Main Voice & Robotics UI + API: `http://localhost:8080`
- Local CPU `faster-whisper` ASR microservice: `http://localhost:8081/health`

---

## 5. Commercial Licensing Checklist Before Production (`APP_ENV=production`)

1. **`d41k4n/piper-swissdial` Voices (`CC BY-NC 4.0`):** Strictly **non-commercial**. Setting `APP_ENV=production` automatically blocks startup if `PIPER_MODEL_LICENSE=cc-by-nc-4.0`. For commercial production, either obtain a commercial license from ETH Zürich / model authors or set `PIPER_MODEL_LICENSE=cc0` (`de_DE-thorsten-medium`) / replace with a commercially licensed Swiss German TTS model.
2. **`faster-whisper` / OpenAI Whisper (`MIT`):** Commercially permissive.
3. **Google Cloud STT V2, TTS, Vertex AI Gemini, Gemini Robotics ER 2:** Governed by Google Cloud Terms of Service.
