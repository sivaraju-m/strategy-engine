# Strategy Engine Terraform Variables

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "bigquery_location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

variable "timezone" {
  description = "Timezone for scheduled jobs"
  type        = string
  default     = "America/New_York"
}

# Cloud Run Configuration
variable "cloud_run_cpu" {
  description = "CPU allocation for Cloud Run"
  type        = string
  default     = "2"
}

variable "cloud_run_memory" {
  description = "Memory allocation for Cloud Run"
  type        = string
  default     = "4Gi"
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

# Scheduling Configuration
variable "strategy_execution_schedule" {
  description = "Cron schedule for strategy execution (market hours)"
  type        = string
  default     = "0 9,12,15 * * 1-5"  # 9 AM, 12 PM, 3 PM EST on weekdays
}

variable "strategy_optimization_schedule" {
  description = "Cron schedule for strategy optimization"
  type        = string
  default     = "0 6 * * 1"  # 6 AM EST every Monday
}

# Monitoring Configuration
variable "notification_channels" {
  description = "List of notification channel IDs for alerts"
  type        = list(string)
  default     = []
}

# Budget Configuration
variable "enable_budget_alerts" {
  description = "Enable budget alerts"
  type        = bool
  default     = true
}

variable "billing_account_id" {
  description = "Billing account ID for budget alerts"
  type        = string
  default     = ""
}

variable "monthly_budget" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 500
}

# Security Configuration
variable "allowed_ingress" {
  description = "Allowed ingress configuration for Cloud Run"
  type        = string
  default     = "INGRESS_TRAFFIC_INTERNAL_ONLY"
}

variable "vpc_connector" {
  description = "VPC connector for private networking"
  type        = string
  default     = ""
}

# Performance Configuration
variable "concurrency" {
  description = "Maximum concurrent requests per instance"
  type        = number
  default     = 100
}

variable "execution_environment" {
  description = "Execution environment (EXECUTION_ENVIRONMENT_GEN1 or EXECUTION_ENVIRONMENT_GEN2)"
  type        = string
  default     = "EXECUTION_ENVIRONMENT_GEN2"
}

# Data Configuration
variable "historical_data_start_date" {
  description = "Start date for historical data processing"
  type        = string
  default     = "2010-01-01"
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 2555  # ~7 years
}

# Strategy Configuration
variable "enabled_strategies" {
  description = "List of enabled strategy types"
  type        = list(string)
  default     = ["momentum", "mean_reversion", "arbitrage", "market_making"]
}

variable "risk_limits" {
  description = "Risk limits configuration"
  type = object({
    max_position_size = number
    max_daily_loss    = number
    max_drawdown      = number
  })
  default = {
    max_position_size = 100000
    max_daily_loss    = 10000
    max_drawdown      = 0.05
  }
}
