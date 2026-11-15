locals {
  required_services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "speech.googleapis.com",
    "texttospeech.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "pubsub.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "firestore.googleapis.com",
    "billingbudgets.googleapis.com",
  ]
}

resource "google_project_service" "required_apis" {
  for_each           = toset(local.required_services)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry in Zurich (europe-west6)
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "swiss-voice-repo"
  description   = "Container images for Swiss Multilingual Voice & Robotics Assistant"
  format        = "DOCKER"
  depends_on    = [google_project_service.required_apis]
}

# Secret Manager secret replicated exclusively in Zurich (europe-west6)
resource "google_secret_manager_secret" "robot_bridge_token" {
  secret_id = "robot-bridge-shared-token"
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
  depends_on = [google_project_service.required_apis]
}

# Pub/Sub topic & subscription for asynchronous robot telemetry (Zurich storage policy)
resource "google_pubsub_topic" "robot_telemetry" {
  name = "robot-telemetry"
  message_storage_policy {
    allowed_persistence_regions = [var.region]
  }
  depends_on = [google_project_service.required_apis]
}

resource "google_pubsub_subscription" "robot_telemetry_sub" {
  name                       = "robot-telemetry-sub"
  topic                      = google_pubsub_topic.robot_telemetry.name
  message_retention_duration = "86400s"
  ack_deadline_seconds       = 20
}

# Optional Zurich GCS bucket for audio recordings with mandatory lifecycle deletion
resource "google_storage_bucket" "audio_recordings" {
  count                       = var.enable_audio_bucket ? 1 : 0
  name                        = "${var.project_id}-swiss-voice-audio"
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  lifecycle_rule {
    condition {
      age = var.audio_retention_days
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required_apis]
}

# Optional Firestore Native database in Zurich (europe-west6)
resource "google_firestore_database" "conversations_db" {
  count       = var.enable_firestore ? 1 : 0
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  depends_on  = [google_project_service.required_apis]
}

# Private CPU-only Swiss German ASR microservice on Cloud Run (europe-west6 — zero GPUs required)
resource "google_cloud_run_v2_service" "asr_service" {
  name     = "swiss-asr-service"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account                  = google_service_account.asr_service.email
    max_instance_request_concurrency = 4
    timeout                          = "120s"

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    containers {
      image = var.asr_image

      resources {
        limits = {
          cpu    = "4"
          memory = "4Gi"
        }
        cpu_idle = true
      }

      env {
        name  = "WHISPER_MODEL"
        value = var.whisper_model
      }
      env {
        name  = "WHISPER_DEVICE"
        value = "cpu"
      }
      env {
        name  = "WHISPER_COMPUTE_TYPE"
        value = "int8"
      }
      env {
        name  = "WHISPER_CPU_THREADS"
        value = "4"
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 3
        period_seconds        = 5
        failure_threshold     = 12
      }
    }
  }

  depends_on = [google_project_service.required_apis]
}

# Main Voice & Robotics Gateway on Cloud Run (europe-west6)
resource "google_cloud_run_v2_service" "voice_backend" {
  name     = "swiss-voice-backend"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account                  = google_service_account.voice_backend.email
    session_affinity                 = true
    max_instance_request_concurrency = 40
    timeout                          = "3600s"

    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    containers {
      image = var.backend_image

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
        cpu_idle = true
      }

      env {
        name  = "APP_ENV"
        value = var.app_env
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.region
      }
      env {
        name  = "GCP_LOCATION"
        value = var.region
      }
      env {
        name  = "GCP_LLM_LOCATION"
        value = var.llm_location
      }
      env {
        name  = "GCP_ROBOTICS_LOCATION"
        value = "global"
      }
      env {
        name  = "GCP_STT_LOCATION"
        value = "eu"
      }
      env {
        name  = "GCP_TTS_LOCATION"
        value = "eu"
      }
      env {
        name  = "STT_DEFAULT_PROVIDER"
        value = "google"
      }
      env {
        name  = "STT_SWISS_GERMAN_PROVIDER"
        value = "faster_whisper"
      }
      env {
        name  = "ASR_ENDPOINT"
        value = google_cloud_run_v2_service.asr_service.uri
      }
      env {
        name  = "ASR_AUTH_ENABLED"
        value = "true"
      }
      env {
        name  = "TTS_DEFAULT_PROVIDER"
        value = "google"
      }
      env {
        name  = "TTS_SWISS_GERMAN_PROVIDER"
        value = "piper"
      }
      env {
        name  = "PIPER_DEFAULT_DIALECT"
        value = "vs"
      }
      env {
        name  = "ALLOW_NON_COMMERCIAL_MODELS"
        value = tostring(var.allow_non_commercial_models)
      }
      env {
        name  = "LLM_PROVIDER"
        value = "google"
      }
      env {
        name  = "GEMINI_MODEL"
        value = var.gemini_model
      }
      env {
        name  = "ROBOTICS_ENABLED"
        value = "true"
      }
      env {
        name  = "ROBOTICS_PROVIDER"
        value = "gemini_robotics_er"
      }
      env {
        name  = "GEMINI_ROBOTICS_MODEL"
        value = var.gemini_robotics_model
      }
      env {
        name  = "PUBSUB_ENABLED"
        value = "true"
      }
      env {
        name  = "PUBSUB_TELEMETRY_TOPIC"
        value = google_pubsub_topic.robot_telemetry.name
      }
      env {
        name  = "FIRESTORE_ENABLED"
        value = tostring(var.enable_firestore)
      }
      env {
        name  = "AUDIO_STORAGE_ENABLED"
        value = tostring(var.enable_audio_bucket)
      }
      env {
        name  = "GCS_AUDIO_BUCKET"
        value = var.enable_audio_bucket ? google_storage_bucket.audio_recordings[0].name : ""
      }
      env {
        name = "ROBOT_BRIDGE_SHARED_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.robot_bridge_token.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 3
        period_seconds        = 5
        failure_threshold     = 12
      }
    }
  }

  depends_on = [google_cloud_run_v2_service.asr_service]
}

# Cloud Monitoring uptime check for main voice gateway
resource "google_monitoring_uptime_check_config" "backend_health" {
  display_name = "swiss-voice-backend-health"
  timeout      = "10s"
  period       = "300s"

  http_check {
    path         = "/health"
    port         = 443
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = replace(google_cloud_run_v2_service.voice_backend.uri, "https://", "")
    }
  }
}

# Optional Cloud Billing budget alert
resource "google_billing_budget" "sandbox_budget" {
  count           = var.billing_account_id != "" ? 1 : 0
  billing_account = var.billing_account_id
  display_name    = "${var.project_id}-swiss-voice-monthly-budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "CHF"
      units         = tostring(var.monthly_budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.8
  }
  threshold_rules {
    threshold_percent = 1.0
  }
}
