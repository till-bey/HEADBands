# Least-privilege Service Accounts

resource "google_service_account" "voice_backend" {
  account_id   = "swiss-voice-backend-sa"
  display_name = "Swiss Voice & Robotics Backend Service Account"
  depends_on   = [google_project_service.required_apis]
}

resource "google_service_account" "asr_service" {
  account_id   = "swiss-asr-cpu-sa"
  display_name = "Swiss German CPU ASR Inference Service Account"
  depends_on   = [google_project_service.required_apis]
}

locals {
  backend_project_roles = concat(
    [
      "roles/aiplatform.user",
      "roles/speech.client",
      "roles/pubsub.publisher",
      "roles/logging.logWriter",
      "roles/monitoring.metricWriter",
    ],
    var.enable_firestore ? ["roles/datastore.user"] : [],
  )
}

resource "google_project_iam_member" "backend_roles" {
  for_each = toset(local.backend_project_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.voice_backend.email}"
}

# Restrict Secret Manager access strictly to the robot-bridge-shared-token secret
resource "google_secret_manager_secret_iam_member" "backend_robot_secret_access" {
  secret_id = google_secret_manager_secret.robot_bridge_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.voice_backend.email}"
}

# Restrict private swiss-asr-service Cloud Run invocation strictly to voice-backend-sa
resource "google_cloud_run_v2_service_iam_member" "backend_invokes_private_asr" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.asr_service.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.voice_backend.email}"
}

# Optional public browser access to voice-backend in sandbox
resource "google_cloud_run_v2_service_iam_member" "public_backend_invoker" {
  count    = var.allow_public_backend_access ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.voice_backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Optional least-privilege bucket access when audio storage is explicitly enabled
resource "google_storage_bucket_iam_member" "backend_audio_bucket_access" {
  count  = var.enable_audio_bucket ? 1 : 0
  bucket = google_storage_bucket.audio_recordings[0].name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.voice_backend.email}"
}
