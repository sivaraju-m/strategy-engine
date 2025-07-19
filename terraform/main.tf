# Strategy Engine Terraform Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "ai-trading-terraform-state"
    prefix = "strategy-engine"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "strategy_engine" {
  location      = var.region
  repository_id = "strategy-engine"
  description   = "Strategy Engine Docker repository"
  format        = "DOCKER"

  labels = {
    component   = "strategy-engine"
    environment = var.environment
    managed-by  = "terraform"
  }
}

# Service Account for Strategy Engine
resource "google_service_account" "strategy_engine" {
  account_id   = "strategy-engine-sa"
  display_name = "Strategy Engine Service Account"
  description  = "Service account for Strategy Engine"
}

# IAM bindings for service account
resource "google_project_iam_member" "strategy_engine_bigquery_user" {
  project = var.project_id
  role    = "roles/bigquery.user"
  member  = "serviceAccount:${google_service_account.strategy_engine.email}"
}

resource "google_project_iam_member" "strategy_engine_bigquery_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.strategy_engine.email}"
}

resource "google_project_iam_member" "strategy_engine_storage_object_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.strategy_engine.email}"
}

resource "google_project_iam_member" "strategy_engine_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.strategy_engine.email}"
}

resource "google_project_iam_member" "strategy_engine_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.strategy_engine.email}"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "strategy_engine" {
  name     = "strategy-engine"
  location = var.region
  
  deletion_protection = false

  template {
    service_account = google_service_account.strategy_engine.email
    
    timeout = "3600s"
    
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/strategy-engine/strategy-engine:latest"
      
      ports {
        container_port = 8080
      }
      
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
      
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
      
      env {
        name  = "REGION"
        value = var.region
      }

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
      }
      
      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        timeout_seconds       = 10
        period_seconds        = 10
        failure_threshold     = 3
      }
      
      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 60
        timeout_seconds       = 10
        period_seconds        = 30
        failure_threshold     = 3
      }
    }
    
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    
    labels = {
      component   = "strategy-engine"
      environment = var.environment
      managed-by  = "terraform"
    }
  }
  
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  depends_on = [google_artifact_registry_repository.strategy_engine]
}

# IAM policy for Cloud Run service
resource "google_cloud_run_service_iam_member" "strategy_engine_invoker" {
  service  = google_cloud_run_v2_service.strategy_engine.name
  location = google_cloud_run_v2_service.strategy_engine.location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.strategy_engine.email}"
}

# BigQuery dataset for strategy results
resource "google_bigquery_dataset" "strategy_results" {
  dataset_id    = "strategy_results"
  friendly_name = "Strategy Results Dataset"
  description   = "Dataset for storing strategy execution results and analytics"
  location      = var.bigquery_location

  labels = {
    component   = "strategy-engine"
    environment = var.environment
    managed-by  = "terraform"
  }

  delete_contents_on_destroy = false

  access {
    role          = "OWNER"
    user_by_email = google_service_account.strategy_engine.email
  }
}

# Cloud Scheduler job for strategy execution
resource "google_cloud_scheduler_job" "strategy_execution" {
  name        = "strategy-execution-job"
  description = "Scheduled strategy execution"
  schedule    = var.strategy_execution_schedule
  time_zone   = var.timezone
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.strategy_engine.uri}/execute"
    
    oidc_token {
      service_account_email = google_service_account.strategy_engine.email
    }
    
    headers = {
      "Content-Type" = "application/json"
    }
    
    body = base64encode(jsonencode({
      strategy_type = "all"
      market_hours  = true
    }))
  }
}

# Cloud Scheduler job for strategy optimization
resource "google_cloud_scheduler_job" "strategy_optimization" {
  name        = "strategy-optimization-job"
  description = "Weekly strategy optimization"
  schedule    = var.strategy_optimization_schedule
  time_zone   = var.timezone
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.strategy_engine.uri}/optimize"
    
    oidc_token {
      service_account_email = google_service_account.strategy_engine.email
    }
    
    headers = {
      "Content-Type" = "application/json"
    }
    
    body = base64encode(jsonencode({
      optimization_type = "full"
      lookback_days    = 30
    }))
  }
}

# Monitoring alert policy for strategy execution failures
resource "google_monitoring_alert_policy" "strategy_execution_failures" {
  display_name = "Strategy Execution Failures"
  combiner     = "OR"
  
  conditions {
    display_name = "Strategy Engine Error Rate"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"strategy-engine\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.1
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields = ["resource.labels.service_name"]
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  alert_strategy {
    auto_close = "86400s"
  }
}

# Budget alert for strategy engine costs
resource "google_billing_budget" "strategy_engine_budget" {
  count = var.enable_budget_alerts ? 1 : 0
  
  billing_account = var.billing_account_id
  display_name    = "Strategy Engine Budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
    labels = {
      component = "strategy-engine"
    }
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_budget)
    }
  }

  threshold_rules {
    threshold_percent = 0.8
    spend_basis      = "CURRENT_SPEND"
  }

  threshold_rules {
    threshold_percent = 1.0
    spend_basis      = "CURRENT_SPEND"
  }
}
