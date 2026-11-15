param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [Parameter(Mandatory = $false)]
    [string]$Region = "europe-west6",
    [Parameter(Mandatory = $false)]
    [string]$LlmLocation = "global",
    [Parameter(Mandatory = $false)]
    [string]$GeminiModel = "gemini-3-flash-preview"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    $sdkBin = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin"
    if (Test-Path $sdkBin) { $env:PATH = "$sdkBin;$env:PATH" }
}

if (-not $ProjectId) {
    $ProjectId = (gcloud config get-value project 2>$null).Trim()
}
if (-not $ProjectId) {
    throw "Set GCP_PROJECT_ID or pass -ProjectId <your-gcp-sandbox-project-id>."
}

$Repo = "swiss-voice-repo"
$AsrImage = "$Region-docker.pkg.dev/$ProjectId/$Repo/asr-service:latest"
$BackendImage = "$Region-docker.pkg.dev/$ProjectId/$Repo/voice-backend:latest"

$ProjectNumber = (gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()
$DefaultComputeSa = "$ProjectNumber-compute@developer.gserviceaccount.com"

# Determine whether custom SA has project IAM bindings or if sandbox requires default Compute SA
$ErrorActionPreference = "Continue"
$CustomSaBound = (gcloud projects get-iam-policy $ProjectId --flatten="bindings[].members" --filter="bindings.members:swiss-voice-backend-sa@$ProjectId.iam.gserviceaccount.com" --format="value(bindings.role)" 2>$null)
$ErrorActionPreference = "Stop"

if ($CustomSaBound) {
    $BackendSa = "swiss-voice-backend-sa@$ProjectId.iam.gserviceaccount.com"
    $AsrSa = "swiss-asr-cpu-sa@$ProjectId.iam.gserviceaccount.com"
} else {
    $BackendSa = $DefaultComputeSa
    $AsrSa = $DefaultComputeSa
}
Write-Host "==> Using runtime service account: $BackendSa"

Write-Host "==> Submitting Cloud Build for asr-service and voice-backend..."
gcloud builds submit --config=cloudbuild.yaml --project=$ProjectId --quiet .

Write-Host "==> Deploying private CPU faster-whisper service (swiss-asr-service) in $Region..."
gcloud run deploy swiss-asr-service `
    --image=$AsrImage `
    --region=$Region `
    --project=$ProjectId `
    --service-account=$AsrSa `
    --no-allow-unauthenticated `
    --cpu=4 `
    --memory=4Gi `
    --concurrency=4 `
    --min-instances=0 `
    --max-instances=3 `
    --cpu-boost `
    --timeout=120 `
    --set-env-vars="WHISPER_MODEL=small,WHISPER_DEVICE=cpu,WHISPER_COMPUTE_TYPE=int8,WHISPER_CPU_THREADS=4,ASR_PRELOAD_ON_STARTUP=true" `
    --quiet

gcloud run services add-iam-policy-binding swiss-asr-service `
    --region=$Region `
    --project=$ProjectId `
    --member="serviceAccount:$BackendSa" `
    --role="roles/run.invoker" `
    --quiet | Out-Null

$AsrUrl = (gcloud run services describe swiss-asr-service --region=$Region --project=$ProjectId --format="value(status.url)").Trim()

Write-Host "==> Deploying Swiss Voice & Robotics Backend (swiss-voice-backend) in $Region..."
$EnvList = @(
    "APP_ENV=development",
    "GCP_PROJECT_ID=$ProjectId",
    "GCP_REGION=$Region",
    "GCP_LOCATION=$Region",
    "GCP_LLM_LOCATION=$LlmLocation",
    "GCP_ROBOTICS_LOCATION=global",
    "GCP_STT_LOCATION=eu",
    "GCP_TTS_LOCATION=eu",
    "STT_DEFAULT_PROVIDER=google",
    "STT_SWISS_GERMAN_PROVIDER=faster_whisper",
    "ASR_ENDPOINT=$AsrUrl",
    "ASR_AUTH_ENABLED=true",
    "TTS_DEFAULT_PROVIDER=google",
    "GOOGLE_TTS_VOICE_DE=de-DE-Neural2-F",
    "GOOGLE_TTS_VOICE_ES=es-ES-Neural2-A",
    "GOOGLE_TTS_VOICE_CS=cs-CZ-Wavenet-A",
    "TTS_SWISS_GERMAN_PROVIDER=piper",
    "PIPER_DEFAULT_DIALECT=vs",
    "ALLOW_NON_COMMERCIAL_MODELS=true",
    "LLM_PROVIDER=google",
    "GEMINI_MODEL=$GeminiModel",
    "ROBOTICS_ENABLED=true",
    "ROBOTICS_PROVIDER=gemini_robotics_er",
    "GEMINI_ROBOTICS_MODEL=gemini-robotics-er-2-preview",
    "PUBSUB_ENABLED=true",
    "PUBSUB_TELEMETRY_TOPIC=robot-telemetry",
    "SERVE_FRONTEND=true"
) -join ","

$ErrorActionPreference = "Continue"
$SecretArgs = @()
$SecretExists = (gcloud secrets describe robot-bridge-shared-token --project=$ProjectId --format="value(name)" 2>$null)
if ($SecretExists) {
    gcloud secrets add-iam-policy-binding robot-bridge-shared-token `
        --project=$ProjectId `
        --member="serviceAccount:$BackendSa" `
        --role="roles/secretmanager.secretAccessor" `
        --quiet | Out-Null
    $SecretArgs = @("--set-secrets=ROBOT_BRIDGE_SHARED_TOKEN=robot-bridge-shared-token:latest")
}

gcloud run deploy swiss-voice-backend `
    --image=$BackendImage `
    --region=$Region `
    --project=$ProjectId `
    --service-account=$BackendSa `
    --allow-unauthenticated `
    --session-affinity `
    --cpu=2 `
    --memory=4Gi `
    --concurrency=40 `
    --min-instances=0 `
    --max-instances=5 `
    --cpu-boost `
    --timeout=3600 `
    --set-env-vars=$EnvList `
    @SecretArgs `
    --quiet
$ErrorActionPreference = "Stop"

$BackendUrl = (gcloud run services describe swiss-voice-backend --region=$Region --project=$ProjectId --format="value(status.url)").Trim()
Write-Host "==> Deployment complete! Live URL: $BackendUrl"
