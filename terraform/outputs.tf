# Strategy Engine Terraform Outputs

output "service_account_email" {
  description = "Email of the Strategy Engine service account"
  value       = google_service_account.strategy_engine.email
}

output "cloud_run_service_url" {
  description = "URL of the Strategy Engine Cloud Run service"
  value       = google_cloud_run_v2_service.strategy_engine.uri
}

output "cloud_run_service_name" {
  description = "Name of the Strategy Engine Cloud Run service"
  value       = google_cloud_run_v2_service.strategy_engine.name
}

output "artifact_registry_repository" {
  description = "Artifact Registry repository for Docker images"
  value       = google_artifact_registry_repository.strategy_engine.name
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.strategy_engine.repository_id}"
}

output "bigquery_dataset_id" {
  description = "BigQuery dataset ID for strategy results"
  value       = google_bigquery_dataset.strategy_results.dataset_id
}

output "bigquery_dataset_location" {
  description = "BigQuery dataset location"
  value       = google_bigquery_dataset.strategy_results.location
}

output "scheduler_job_names" {
  description = "Names of Cloud Scheduler jobs"
  value = [
    google_cloud_scheduler_job.strategy_execution.name,
    google_cloud_scheduler_job.strategy_optimization.name
  ]
}

output "monitoring_alert_policy_name" {
  description = "Name of the monitoring alert policy"
  value       = google_monitoring_alert_policy.strategy_execution_failures.name
}

output "budget_name" {
  description = "Name of the budget alert (if enabled)"
  value       = var.enable_budget_alerts ? google_billing_budget.strategy_engine_budget[0].display_name : null
}

# Configuration outputs for other services
output "strategy_engine_config" {
  description = "Strategy Engine configuration for other services"
  value = {
    service_url           = google_cloud_run_v2_service.strategy_engine.uri
    service_account_email = google_service_account.strategy_engine.email
    dataset_id           = google_bigquery_dataset.strategy_results.dataset_id
    execution_schedule   = var.strategy_execution_schedule
    optimization_schedule = var.strategy_optimization_schedule
  }
  sensitive = false
}

# Deployment information
output "deployment_info" {
  description = "Deployment information and next steps"
  value = {
    docker_image_url = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.strategy_engine.repository_id}/strategy-engine:latest"
    health_check_url = "${google_cloud_run_v2_service.strategy_engine.uri}/health"
    execution_endpoint = "${google_cloud_run_v2_service.strategy_engine.uri}/execute"
    optimization_endpoint = "${google_cloud_run_v2_service.strategy_engine.uri}/optimize"
  }
}
