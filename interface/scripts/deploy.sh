#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-europe-west6}"
LLM_LOCATION="${GCP_LLM_LOCATION:-global}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"
REPO="swiss-voice-repo"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: Set GCP_PROJECT_ID or run 'gcloud config set project YOUR_PROJECT_ID'" >&2
  exit 1
fi

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
DEFAULT_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
CUSTOM_SA_ROLES="$(gcloud projects get-iam-policy "${PROJECT_ID}" --flatten='bindings[].members' --filter="bindings.members:swiss-voice-backend-sa@${PROJECT_ID}.iam.gserviceaccount.com" --format='value(bindings.role)' 2>/dev/null || true)"

if [[ -n "${CUSTOM_SA_ROLES}" ]]; then
  BACKEND_SA="swiss-voice-backend-sa@${PROJECT_ID}.iam.gserviceaccount.com"
  ASR_SA="swiss-asr-cpu-sa@${PROJECT_ID}.iam.gserviceaccount.com"
else
  BACKEND_SA="${DEFAULT_SA}"
  ASR_SA="${DEFAULT_SA}"
fi

echo "==> Submitting container builds to Google Cloud Build (${REGION})..."
gcloud builds submit --config=cloudbuild.yaml --project="${PROJECT_ID}" --quiet .

ASR_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/asr-service:latest"
BACKEND_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/voice-backend:latest"

echo "==> Deploying private CPU faster-whisper ASR service (swiss-asr-service)..."
gcloud run deploy swiss-asr-service \
  --image="${ASR_IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --service-account="${ASR_SA}" \
  --no-allow-unauthenticated \
  --cpu=4 \
  --memory=4Gi \
  --concurrency=4 \
  --min-instances=0 \
  --max-instances=3 \
  --cpu-boost \
  --timeout=120 \
  --set-env-vars="WHISPER_MODEL=small,WHISPER_DEVICE=cpu,WHISPER_COMPUTE_TYPE=int8,WHISPER_CPU_THREADS=4,ASR_PRELOAD_ON_STARTUP=true" \
  --quiet

gcloud run services add-iam-policy-binding swiss-asr-service \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${BACKEND_SA}" \
  --role="roles/run.invoker" \
  --quiet >/dev/null

ASR_URL="$(gcloud run services describe swiss-asr-service --region="${REGION}" --project="${PROJECT_ID}" --format='value(status.url)')"

SECRET_ARGS=()
if gcloud secrets describe robot-bridge-shared-token --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud secrets add-iam-policy-binding robot-bridge-shared-token \
    --project="${PROJECT_ID}" \
    --member="serviceAccount:${BACKEND_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null || true
  SECRET_ARGS=("--set-secrets=ROBOT_BRIDGE_SHARED_TOKEN=robot-bridge-shared-token:latest")
fi

echo "==> Deploying Swiss Voice & Robotics Backend (swiss-voice-backend)..."
gcloud run deploy swiss-voice-backend \
  --image="${BACKEND_IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --service-account="${BACKEND_SA}" \
  --allow-unauthenticated \
  --session-affinity \
  --cpu=2 \
  --memory=4Gi \
  --concurrency=40 \
  --min-instances=0 \
  --max-instances=5 \
  --cpu-boost \
  --timeout=3600 \
  --set-env-vars="APP_ENV=development,GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},GCP_LOCATION=${REGION},GCP_LLM_LOCATION=${LLM_LOCATION},GCP_ROBOTICS_LOCATION=global,GCP_STT_LOCATION=eu,GCP_TTS_LOCATION=eu,STT_DEFAULT_PROVIDER=google,STT_SWISS_GERMAN_PROVIDER=faster_whisper,ASR_ENDPOINT=${ASR_URL},ASR_AUTH_ENABLED=true,TTS_DEFAULT_PROVIDER=google,GOOGLE_TTS_VOICE_DE=de-DE-Neural2-F,GOOGLE_TTS_VOICE_ES=es-ES-Neural2-A,GOOGLE_TTS_VOICE_CS=cs-CZ-Wavenet-A,TTS_SWISS_GERMAN_PROVIDER=piper,PIPER_DEFAULT_DIALECT=vs,ALLOW_NON_COMMERCIAL_MODELS=true,LLM_PROVIDER=google,GEMINI_MODEL=${GEMINI_MODEL},ROBOTICS_ENABLED=true,ROBOTICS_PROVIDER=gemini_robotics_er,GEMINI_ROBOTICS_MODEL=gemini-robotics-er-2-preview,PUBSUB_ENABLED=true,PUBSUB_TELEMETRY_TOPIC=robot-telemetry,SERVE_FRONTEND=true" \
  "${SECRET_ARGS[@]}" \
  --quiet || true

BACKEND_URL="$(gcloud run services describe swiss-voice-backend --region="${REGION}" --project="${PROJECT_ID}" --format='value(status.url)')"
echo "==> Deployment complete! Live URL: ${BACKEND_URL}"
