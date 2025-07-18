# Shared Services Infrastructure

This directory contains the Terraform infrastructure for the shared-services subproject. 

## Components

- Common storage buckets
- BigQuery datasets
- Secret Manager resources
- IAM policies and permissions
- VPC networking (if needed)

## Deployment

```bash
# Initialize Terraform
terraform init

# Create a workspace (if needed)
terraform workspace new dev

# Plan the deployment
terraform plan -out=tfplan

# Apply the changes
terraform apply tfplan
```

## Environment Variables

The following environment variables need to be set before deploying:

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to the service account key file
- `TF_VAR_project_id`: GCP project ID
- `TF_VAR_region`: GCP region (e.g., us-central1)

## Modules

The infrastructure is organized into the following modules:

- `gcs`: Storage buckets
- `bigquery`: BigQuery datasets and tables
- `iam`: IAM roles and policies
- `secrets`: Secret Manager resources
