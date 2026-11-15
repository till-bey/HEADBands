#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-europe-west6}"
REPO="swiss-voice-repo"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: Set GCP_PROJECT_ID or run 'gcloud config set project YOUR_PROJECT_ID'" >&2
  exit 1
fi

echo "==> Configuring GCP Project: ${PROJECT_ID} (Region: ${REGION}, CPU-only)"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  speech.googleapis.com \
  texttospeech.googleapis.com \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  pubsub.googleapis.com \
  firestore.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet

gcloud artifacts repositories describe "${REPO}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Swiss Multilingual Voice & Robotics containers" \
    --project="${PROJECT_ID}" \
    --quiet

for SA in "swiss-voice-backend-sa" "swiss-asr-cpu-sa"; do
  gcloud iam service-accounts describe "${SA}@${PROJECT_ID}.iam.gserviceaccount.com" --project="${PROJECT_ID}" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "${SA}" --display-name="${SA}" --project="${PROJECT_ID}" --quiet || true
done

BACKEND_SA="swiss-voice-backend-sa@${PROJECT_ID}.iam.gserviceaccount.com"
if gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${BACKEND_SA}" --role="roles/aiplatform.user" --quiet >/dev/null 2>&1; then
  for ROLE in "roles/speech.client" "roles/pubsub.publisher" "roles/logging.logWriter" "roles/monitoring.metricWriter"; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
      --member="serviceAccount:${BACKEND_SA}" \
      --role="${ROLE}" \
      --quiet >/dev/null
  done
else
  echo "    Sandbox restricts resourcemanager.projects.setIamPolicy; deploy.sh will use default Compute SA."
fi

gcloud pubsub topics describe "robot-telemetry" --project="${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud pubsub topics create "robot-telemetry" \
    --message-storage-policy-allowed-regions="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet

gcloud pubsub subscriptions describe "robot-telemetry-sub" --project="${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud pubsub subscriptions create "robot-telemetry-sub" \
    --topic="robot-telemetry" \
    --project="${PROJECT_ID}" \
    --quiet

gcloud secrets describe "robot-bridge-shared-token" --project="${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud secrets create "robot-bridge-shared-token" \
    --replication-policy=user-managed \
    --locations="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet

echo "==> GCP sandbox foundation ready in ${REGION}."
