output "voice_backend_url" {
  description = "HTTPS/WSS URL of the main Swiss Voice & Robotics Backend on Cloud Run (europe-west6)."
  value       = google_cloud_run_v2_service.voice_backend.uri
}

output "asr_service_private_url" {
  description = "Private IAM-authenticated URL of the CPU faster-whisper ASR service on Cloud Run (europe-west6)."
  value       = google_cloud_run_v2_service.asr_service.uri
}

output "artifact_registry_repo" {
  description = "Artifact Registry Docker repository path in Zurich."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "telemetry_pubsub_topic" {
  description = "Pub/Sub topic for robot telemetry."
  value       = google_pubsub_topic.robot_telemetry.id
}
