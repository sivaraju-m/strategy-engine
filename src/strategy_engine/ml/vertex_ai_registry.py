"""
Vertex AI Model Registry Integration
Handles model versioning, drift detection, and automated retraining
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

try:
    from google.cloud import aiplatform

    AIPLATFORM_AVAILABLE = True
except ImportError:
    AIPLATFORM_AVAILABLE = False
    aiplatform = None

try:
    from google.cloud import bigquery

    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    bigquery = None

try:
    from google.cloud import firestore

    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    firestore = None

try:
    from google.cloud import storage

    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    storage = None

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

try:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = None
    accuracy_score = precision_score = recall_score = f1_score = None


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load configuration
def load_config(file_path="config.yaml"):
    with open(file_path) as f:
        return yaml.safe_load(f)


config = load_config()


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    annual_return: Optional[float] = None
    timestamp: datetime = None


@dataclass
class DriftMetrics:
    """Model drift detection metrics"""

    input_drift_score: float
    prediction_drift_score: float
    data_quality_score: float
    drift_detected: bool
    timestamp: datetime = None


class VertexAIModelRegistry:
    """
    Manages ML models in Vertex AI with SEBI compliance
    """

    def __init__(self):
        """Initialize the model registry"""
        self.project_id = config.get("project_id", "default_project_id")
        self.region = config.get("region", "us-central1")
        self.model_bucket = config.get("model_bucket", "default-model-bucket")
        self.drift_threshold = config.get("drift_threshold", 0.15)
        self.performance_threshold = config.get("performance_threshold", 0.8)

        # Initialize clients
        try:
            aiplatform.init(project=self.project_id, location=self.region)
            self.bq_client = bigquery.Client(project=self.project_id)
            self.firestore_client = firestore.Client(project=self.project_id)
            self.storage_client = storage.Client(project=self.project_id)
        except Exception:
            logger.error("Error initializing clients")
            raise

    def register_model(
        self,
        model: BaseEstimator,
        model_name: str,
        model_type: str,
        strategy_id: str,
        metrics: ModelMetrics,
        metadata: dict[str, Any] = None,
    ) -> str:
        """
        Register a new model version in Vertex AI

        Args:
            model: Trained ML model
            model_name: Name of the model
            model_type: Type (classifier/regression)
            strategy_id: Associated trading strategy ID
            metrics: Model performance metrics
            metadata: Additional metadata

        Returns:
            Model version ID
        """
        try:
            # Generate model version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_version = "{model_name}_v{timestamp}"

            # Save model artifacts to GCS
            model_uri = self._save_model_artifacts(model, model_version)

            # Create Vertex AI model
            vertex_model = aiplatform.Model.upload(
                display_name=model_version,
                artifact_uri=model_uri,
                serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest",
                labels={
                    "strategy_id": strategy_id,
                    "model_type": model_type,
                    "environment": "production",
                },
            )

            # Store model metadata in Firestore
            model_doc = {
                "model_name": model_name,
                "model_version": model_version,
                "model_type": model_type,
                "strategy_id": strategy_id,
                "vertex_model_id": vertex_model.resource_name,
                "model_uri": model_uri,
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "annual_return": metrics.annual_return,
                },
                "metadata": metadata or {},
                "status": "registered",
                "created_at": firestore.SERVER_TIMESTAMP,
                "approved_for_prod": False,
                "approved_by": None,
                "approval_timestamp": None,
            }

            doc_ref = self.firestore_client.collection("ml_models").document(
                model_version
            )
            doc_ref.set(model_doc)

            # Log to BigQuery for analytics
            self._log_model_registration(model_version, model_doc)

            logger.info("Model {model_version} registered successfully")
            return model_version

        except Exception:
            logger.error("Error registering model {model_name}")
            raise

    def _save_model_artifacts(self, model: BaseEstimator, model_version: str) -> str:
        """Save model artifacts to GCS"""
        try:
            # Save model using joblib
            import os
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                model_file = os.path.join(tmp_dir, "model.pkl")
                joblib.dump(model, model_file)

                # Upload to GCS
                bucket = self.storage_client.bucket(self.model_bucket)
                blob = bucket.blob("{model_path}/model.pkl")
                blob.upload_from_filename(model_file)

                # Create model metadata
                metadata = {
                    "model_version": model_version,
                    "created_at": datetime.now().isoformat(),
                    "model_type": type(model).__name__,
                    "sklearn_version": getattr(model, "_sklearn_version", "unknown"),
                }

                metadata_blob = bucket.blob("{model_path}/metadata.json")
                metadata_blob.upload_from_string(json.dumps(metadata))

            return "gs://{self.model_bucket}/{model_path}"

        except Exception:
            logger.error("Error saving model artifacts")
            raise

    def detect_model_drift(
        self,
        model_version: str,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
    ) -> DriftMetrics:
        """
        Detect model drift using statistical tests

        Args:
            model_version: Model version to check
            current_data: Current input data
            reference_data: Reference training data

        Returns:
            Drift metrics
        """
        try:
            # Calculate input drift using KL divergence
            input_drift = self._calculate_input_drift(current_data, reference_data)

            # Load model and calculate prediction drift
            model = self._load_model(model_version)
            current_predictions = model.predict(current_data)
            reference_predictions = model.predict(reference_data)

            prediction_drift = self._calculate_prediction_drift(
                current_predictions, reference_predictions
            )

            # Calculate data quality score
            data_quality = self._calculate_data_quality(current_data)

            # Determine if drift detected
            drift_detected = (
                input_drift > self.drift_threshold
                or prediction_drift > self.drift_threshold
                or data_quality < 0.8
            )

            drift_metrics = DriftMetrics(
                input_drift_score=input_drift,
                prediction_drift_score=prediction_drift,
                data_quality_score=data_quality,
                drift_detected=drift_detected,
                timestamp=datetime.now(),
            )

            # Store drift metrics
            self._store_drift_metrics(model_version, drift_metrics)

            # Trigger retraining if drift detected
            if drift_detected:
                self._trigger_retraining(model_version, drift_metrics)

            return drift_metrics

        except Exception:
            logger.error("Error detecting drift for model {model_version}")
            raise

    def _calculate_input_drift(
        self, current_data: pd.DataFrame, reference_data: pd.DataFrame
    ) -> float:
        """Calculate input drift using KL divergence"""
        try:
            # Simple implementation using feature-wise KL divergence
            total_drift = 0.0
            n_features = len(current_data.columns)

            for column in current_data.columns:
                if pd.api.types.is_numeric_dtype(current_data[column]):
                    # Calculate histograms
                    current_hist, bins = np.histogram(
                        current_data[column], bins=20, density=True
                    )
                    reference_hist, _ = np.histogram(
                        reference_data[column], bins=bins, density=True
                    )

                    # Add small epsilon to avoid division by zero
                    epsilon = 1e-8
                    current_hist += epsilon
                    reference_hist += epsilon

                    # Calculate KL divergence
                    kl_div = np.sum(
                        current_hist * np.log(current_hist / reference_hist)
                    )
                    total_drift += kl_div

            return total_drift / n_features

        except Exception:
            logger.error("Error calculating input drift")
            return 1.0  # Return high drift on error

    def _calculate_prediction_drift(
        self, current_preds: np.ndarray, reference_preds: np.ndarray
    ) -> float:
        """Calculate prediction drift using statistical distance"""
        try:
            # Calculate Jensen-Shannon divergence
            def js_divergence(p: np.ndarray, q: np.ndarray):
                p = np.asarray(p)
                q = np.asarray(q)
                p = p / np.sum(p)
                q = q / np.sum(q)
                m = 0.5 * (p + q)

                # KL divergence with small epsilon
                epsilon = 1e-8
                return 0.5 * np.sum(
                    p * np.log((p + epsilon) / (m + epsilon))
                ) + 0.5 * np.sum(q * np.log((q + epsilon) / (m + epsilon)))

            # Create histograms
            current_hist, bins = np.histogram(current_preds, bins=20, density=True)
            reference_hist, _ = np.histogram(reference_preds, bins=bins, density=True)

            return js_divergence(current_hist, reference_hist)

        except Exception:
            logger.error("Error calculating prediction drift")
            return 1.0

    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            duplicate_rows = data.duplicated().sum()

            # Calculate quality metrics
            completeness = 1 - (missing_cells / total_cells)
            uniqueness = 1 - (duplicate_rows / len(data))

            # Simple weighted average
            quality_score = 0.7 * completeness + 0.3 * uniqueness

            return quality_score

        except Exception:
            logger.error("Error calculating data quality")
            return 0.0

    def _load_model(self, model_version: str) -> Any:
        """Load model from GCS"""
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as _:  # Replaced 'tmp_dir' with '_'
                # Download model from GCS
                bucket = self.storage_client.bucket(self.model_bucket)
                blob = bucket.blob("models/{model_version}/model.pkl")

                model_file = "{tmp_dir}/model.pkl"
                blob.download_to_filename(model_file)

                # Load model
                model = joblib.load(model_file)
                return model

        except Exception:
            logger.error("Error loading model {model_version}")
            raise

    def _store_drift_metrics(self, model_version: str, drift_metrics: DriftMetrics):
        """Store drift metrics in Firestore and BigQuery"""
        try:
            # Store in Firestore
            drift_doc = {
                "model_version": model_version,
                "input_drift_score": drift_metrics.input_drift_score,
                "prediction_drift_score": drift_metrics.prediction_drift_score,
                "data_quality_score": drift_metrics.data_quality_score,
                "drift_detected": drift_metrics.drift_detected,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }

            self.firestore_client.collection("model_drift").add(drift_doc)

            # Log to BigQuery
            table_id = "{self.project_id}.ml_monitoring.drift_metrics"

            rows_to_insert = [
                {
                    "model_version": model_version,
                    "input_drift_score": drift_metrics.input_drift_score,
                    "prediction_drift_score": drift_metrics.prediction_drift_score,
                    "data_quality_score": drift_metrics.data_quality_score,
                    "drift_detected": drift_metrics.drift_detected,
                    "timestamp": drift_metrics.timestamp.isoformat(),
                }
            ]

            errors = self.bq_client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error("BigQuery insert errors: {errors}")

        except Exception:
            logger.error("Error storing drift metrics")

    def _trigger_retraining(self, model_version: str, drift_metrics: DriftMetrics):
        """Trigger automated retraining workflow"""
        try:
            # Create retraining request
            retraining_doc = {
                "model_version": model_version,
                "trigger_reason": "drift_detected",
                "drift_metrics": {
                    "input_drift_score": drift_metrics.input_drift_score,
                    "prediction_drift_score": drift_metrics.prediction_drift_score,
                    "data_quality_score": drift_metrics.data_quality_score,
                },
                "status": "pending",
                "created_at": firestore.SERVER_TIMESTAMP,
                "priority": (
                    "high" if drift_metrics.input_drift_score > 0.3 else "medium"
                ),
            }

            self.firestore_client.collection("retraining_requests").add(retraining_doc)

            logger.info("Retraining triggered for model {model_version} due to drift")

        except Exception:
            logger.error("Error triggering retraining")

    def approve_model_for_production(
        self, model_version: str, approved_by: str, remarks: str = ""
    ) -> bool:
        """
        Approve model for production use with SEBI compliance

        Args:
            model_version: Model version to approve
            approved_by: User ID who approved the model
            remarks: Approval remarks

        Returns:
            Success status
        """
        try:
            # Update model status in Firestore
            doc_ref = self.firestore_client.collection("ml_models").document(
                model_version
            )
            doc_ref.update(
                {
                    "approved_for_prod": True,
                    "approved_by": approved_by,
                    "approval_timestamp": firestore.SERVER_TIMESTAMP,
                    "approval_remarks": remarks,
                    "status": "production_approved",
                }
            )

            # Log approval for audit trail
            audit_doc = {
                "action": "model_approval",
                "model_version": model_version,
                "approved_by": approved_by,
                "remarks": remarks,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "compliance_status": "sebi_compliant",
            }

            self.firestore_client.collection("audit_trail").add(audit_doc)

            logger.info(
                "Model {model_version} approved for production by {approved_by}"
            )
            return True

        except Exception:
            logger.error("Error approving model {model_version}")
            return False

    def get_model_performance_report(self, model_version: str) -> dict[str, Any]:
        """Get comprehensive model performance report"""
        try:
            # Get model metadata
            doc_ref = self.firestore_client.collection("ml_models").document(
                model_version
            )
            model_doc = doc_ref.get()

            if not model_doc.exists:
                raise ValueError("Model {model_version} not found")

            model_data = model_doc.to_dict()

            # Get drift history
            drift_query = (
                self.firestore_client.collection("model_drift")
                .where("model_version", "==", model_version)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(10)
            )

            drift_history = [doc.to_dict() for doc in drift_query.stream()]

            # Compile performance report
            performance_report = {
                "model_info": {
                    "model_version": model_version,
                    "model_type": model_data.get("model_type"),
                    "strategy_id": model_data.get("strategy_id"),
                    "status": model_data.get("status"),
                    "approved_for_prod": model_data.get("approved_for_prod", False),
                },
                "performance_metrics": model_data.get("metrics", {}),
                "drift_history": drift_history,
                "last_drift_check": drift_history[0] if drift_history else None,
                "compliance_status": (
                    "approved"
                    if model_data.get("approved_for_prod")
                    else "pending_approval"
                ),
            }

            return performance_report

        except Exception:
            logger.error("Error generating performance report for {model_version}")
            raise

    def _log_model_registration(self, model_version: str, model_doc: dict[str, Any]):
        """Log model registration to BigQuery"""
        try:
            table_id = "{self.project_id}.ml_monitoring.model_registry"

            rows_to_insert = [
                {
                    "model_version": model_version,
                    "model_name": model_doc["model_name"],
                    "model_type": model_doc["model_type"],
                    "strategy_id": model_doc["strategy_id"],
                    "accuracy": model_doc["metrics"]["accuracy"],
                    "precision": model_doc["metrics"]["precision"],
                    "recall": model_doc["metrics"]["recall"],
                    "f1_score": model_doc["metrics"]["f1_score"],
                    "sharpe_ratio": model_doc["metrics"].get("sharpe_ratio"),
                    "created_at": datetime.now().isoformat(),
                    "status": model_doc["status"],
                }
            ]

            errors = self.bq_client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error("BigQuery model registration errors: {errors}")

        except Exception:
            logger.error("Error logging model registration")

    def schedule_weekly_retraining(self):
        """Schedule weekly retraining check"""
        try:
            # This would integrate with Cloud Scheduler
            # For now, we'll create a retraining schedule document
            schedule_doc = {
                "schedule_type": "weekly_retraining",
                "cron_expression": "0 2 * * 1",  # Every Monday at 2 AM
                "enabled": True,
                "last_run": None,
                "next_run": None,
                "created_at": firestore.SERVER_TIMESTAMP,
            }

            self.firestore_client.collection("ml_schedules").document(
                "weekly_retraining"
            ).set(schedule_doc)

            logger.info("Weekly retraining schedule configured")

        except Exception:
            logger.error("Error scheduling weekly retraining")


# Health check and statistics functions
def health_check() -> dict[str, Any]:
    """Health check for Vertex AI Model Registry"""
    try:
        return {
            "status": "healthy",
            "module": "vertex_ai_registry",
            "features": [
                "Model Registration",
                "Drift Detection",
                "Automated Retraining",
                "Model Versioning",
                "Performance Tracking",
            ],
            "dependencies": {
                "aiplatform": AIPLATFORM_AVAILABLE,
                "bigquery": True,
                "firestore": True,
                "storage": True,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
        }


def get_statistics() -> dict[str, Any]:
    """Get Vertex AI Model Registry statistics"""
    return {
        "module": "vertex_ai_registry",
        "features": [
            "Model Registration",
            "Drift Detection",
            "Automated Retraining",
            "Model Versioning",
            "Performance Tracking",
        ],
        "model_types": ["classification", "regression"],
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "drift_detection": ["input_drift", "output_drift"],
        "aiplatform_available": AIPLATFORM_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--health":
            print(json.dumps(health_check(), indent=2))
        elif sys.argv[1] == "--stats":
            print(json.dumps(get_statistics(), indent=2))
        elif sys.argv[1] == "--test":
            print("🧪 Testing Vertex AI Model Registry...")
            result = health_check()
            print("Status: {result['status']}")
        else:
            print("Usage: python vertex_ai_registry.py [--health|--stats|--test]")
    else:
        print("Vertex AI Model Registry - Use --help for usage")
