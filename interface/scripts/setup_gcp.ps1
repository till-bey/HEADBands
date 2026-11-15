param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [Parameter(Mandatory = $false)]
    [string]$Region = "europe-west6"
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

Write-Host "==> Enabling required GCP APIs on '$ProjectId'..."
gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    speech.googleapis.com `
    texttospeech.googleapis.com `
    aiplatform.googleapis.com `
    secretmanager.googleapis.com `
    pubsub.googleapis.com `
    firestore.googleapis.com `
    logging.googleapis.com `
    monitoring.googleapis.com `
    --project=$ProjectId `
    --quiet

Write-Host "==> Ensuring Zurich Artifact Registry repository 'swiss-voice-repo' ($Region)..."
$ErrorActionPreference = "Continue"
gcloud artifacts repositories describe swiss-voice-repo --location=$Region --project=$ProjectId --quiet *> $null
if ($LASTEXITCODE -ne 0) {
    $ErrorActionPreference = "Stop"
    gcloud artifacts repositories create swiss-voice-repo `
        --repository-format=docker `
        --location=$Region `
        --description="Swiss Multilingual Voice & Robotics containers" `
        --project=$ProjectId `
        --quiet
}
$ErrorActionPreference = "Stop"

Write-Host "==> Configuring Service Accounts (with automatic GCP Sandbox fallback)..."
$ErrorActionPreference = "Continue"
foreach ($sa in @("swiss-voice-backend-sa", "swiss-asr-cpu-sa")) {
    $saEmail = "$sa@$ProjectId.iam.gserviceaccount.com"
    gcloud iam service-accounts describe $saEmail --project=$ProjectId --quiet *> $null
    if ($LASTEXITCODE -ne 0) {
        gcloud iam service-accounts create $sa --display-name=$sa --project=$ProjectId --quiet *> $null
    }
}

$backendSa = "swiss-voice-backend-sa@$ProjectId.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding $ProjectId `
    --member="serviceAccount:$backendSa" `
    --role="roles/aiplatform.user" `
    --quiet *> $null

if ($LASTEXITCODE -eq 0) {
    foreach ($role in @("roles/speech.client", "roles/pubsub.publisher", "roles/logging.logWriter", "roles/monitoring.metricWriter")) {
        gcloud projects add-iam-policy-binding $ProjectId `
            --member="serviceAccount:$backendSa" `
            --role=$role `
            --quiet *> $null
    }
    Write-Host "    Bound least-privilege project IAM roles to $backendSa."
} else {
    $projNum = (gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()
    Write-Host "    Sandbox restricts resourcemanager.projects.setIamPolicy; deploy scripts will use $projNum-compute@developer.gserviceaccount.com."
}

Write-Host "==> Ensuring Zurich Pub/Sub topic 'robot-telemetry' and Secret Manager secret..."
gcloud pubsub topics describe robot-telemetry --project=$ProjectId --quiet *> $null
if ($LASTEXITCODE -ne 0) {
    gcloud pubsub topics create robot-telemetry `
        --message-storage-policy-allowed-regions=$Region `
        --project=$ProjectId `
        --quiet
}

gcloud pubsub subscriptions describe robot-telemetry-sub --project=$ProjectId --quiet *> $null
if ($LASTEXITCODE -ne 0) {
    gcloud pubsub subscriptions create robot-telemetry-sub `
        --topic=robot-telemetry `
        --project=$ProjectId `
        --quiet
}

gcloud secrets describe robot-bridge-shared-token --project=$ProjectId --quiet *> $null
if ($LASTEXITCODE -ne 0) {
    gcloud secrets create robot-bridge-shared-token `
        --replication-policy=user-managed `
        --locations=$Region `
        --project=$ProjectId `
        --quiet
}
$ErrorActionPreference = "Stop"

Write-Host "==> GCP sandbox setup complete for '$ProjectId' ($Region)."
