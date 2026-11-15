variable "project_id" {
  description = "GCP Project ID for the Swiss Multilingual Voice & Robotics platform."
  type        = string
}

variable "region" {
  description = "Primary Swiss GCP region (Zurich)."
  type        = string
  default     = "europe-west6"
}

variable "llm_location" {
  description = "Vertex AI endpoint for Gemini foundation models (global for gemini-3-flash-preview, or regional EU endpoint when enabled)."
  type        = string
  default     = "global"
}

variable "gemini_model" {
  description = "Main conversational Gemini model identifier."
  type        = string
  default     = "gemini-3-flash-preview"
}

variable "gemini_robotics_model" {
  description = "Embodied reasoning Gemini Robotics ER 2 model identifier (global endpoint)."
  type        = string
  default     = "gemini-robotics-er-2-preview"
}

variable "whisper_model" {
  description = "CPU int8 faster-whisper model size on swiss-asr-service (small, large-v3-turbo, large-v3)."
  type        = string
  default     = "small"
}

variable "backend_image" {
  description = "Artifact Registry container image URI for swiss-voice-backend."
  type        = string
}

variable "asr_image" {
  description = "Artifact Registry container image URI for swiss-asr-service (CPU int8)."
  type        = string
}

variable "allow_public_backend_access" {
  description = "Allow unauthenticated HTTPS/WSS access to swiss-voice-backend for browser testing in sandbox."
  type        = bool
  default     = true
}

variable "allow_non_commercial_models" {
  description = "Enable CC BY-NC 4.0 SwissDial Piper voices in non-commercial sandbox/development deployments."
  type        = bool
  default     = true
}

variable "app_env" {
  description = "Environment mode (development, staging, production)."
  type        = string
  default     = "development"
}

variable "enable_audio_bucket" {
  description = "Create optional Zurich GCS bucket with automatic retention deletion (disabled by default for privacy)."
  type        = bool
  default     = false
}

variable "audio_retention_days" {
  description = "Automatic deletion age (days) when optional audio storage is enabled."
  type        = number
  default     = 1
}

variable "enable_firestore" {
  description = "Provision Firestore Native database in europe-west6 for conversation persistence."
  type        = bool
  default     = false
}

variable "billing_account_id" {
  description = "Optional Billing Account ID to provision automated monthly cost budget alerts."
  type        = string
  default     = ""
}

variable "monthly_budget_amount" {
  description = "Monthly budget threshold in billing currency when billing_account_id is set."
  type        = number
  default     = 50
}
