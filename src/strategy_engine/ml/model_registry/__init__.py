"""
Model registry module for AI Trading Machine.

This module handles model versioning, storage, and retrieval using
either Google Cloud Storage (GCS) or Vertex AI Model Registry.
"""

import datetime
import json
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Base class for model registry implementations.
    """

    def save_model(self, model_path: str, metadata: dict) -> str:
        """
        Save a model to the registry.

        Args:
            model_path: Path to the model file
            metadata: Dictionary of metadata about the model

        Returns:
            Registry path or identifier for the saved model
        """
        raise NotImplementedError("Subclasses must implement save_model method")

    def load_model(self, model_id: str, destination_path: str) -> str:
        """
        Load a model from the registry.

        Args:
            model_id: Registry identifier for the model
            destination_path: Local path to save the model

        Returns:
            Local path to the loaded model
        """
        raise NotImplementedError("Subclasses must implement load_model method")

    def list_models(self) -> list[dict]:
        """
        List all models in the registry.

        Returns:
            List of dictionaries with model information
        """
        raise NotImplementedError("Subclasses must implement list_models method")


class GCSModelRegistry(ModelRegistry):
    """
    Model registry implementation using Google Cloud Storage (GCS).
    """

    def __init__(self, bucket_name: str, base_path: str = "models"):
        """
        Initialize the GCS model registry.

        Args:
            bucket_name: Name of the GCS bucket
            base_path: Base path within the bucket for storing models
        """
        self.bucket_name = bucket_name
        self.base_path = base_path

        # Import Google Cloud Storage client
        try:
            from google.cloud import storage

            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)
        except ImportError:
            logger.warning(
                "google-cloud-storage not installed. GCS functionality will be limited."
            )
            self.storage_client = None
            self.bucket = None

    def save_model(self, model_path: str, metadata: dict) -> str:
        """
        Save a model to GCS.

        Args:
            model_path: Path to the model file
            metadata: Dictionary of metadata about the model

        Returns:
            GCS path to the saved model
        """
        if self.storage_client is None:
            raise ImportError("google-cloud-storage not installed")

        # Generate a timestamp-based version
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path)
        model_name_without_ext = os.path.splitext(model_name)[0]

        # Create a version directory
        version_dir = "{self.base_path}/{model_name_without_ext}/{timestamp}"

        # Upload the model file
        model_gcs_path = "{version_dir}/{model_name}"
        blob = self.bucket.blob(model_gcs_path)
        blob.upload_from_filename(model_path)

        # Save metadata
        metadata_with_timestamp = metadata.copy()
        metadata_with_timestamp["timestamp"] = timestamp
        metadata_with_timestamp["model_path"] = model_gcs_path

        metadata_gcs_path = "{version_dir}/metadata.json"
        metadata_blob = self.bucket.blob(metadata_gcs_path)
        metadata_blob.upload_from_string(json.dumps(metadata_with_timestamp, indent=2))

        logger.info("Model saved to gs://{self.bucket_name}/{model_gcs_path}")

        return "gs://{self.bucket_name}/{model_gcs_path}"

    def load_model(self, model_id: str, destination_path: str) -> str:
        """
        Load a model from GCS.

        Args:
            model_id: GCS path to the model
            destination_path: Local path to save the model

        Returns:
            Local path to the loaded model
        """
        if self.storage_client is None:
            raise ImportError("google-cloud-storage not installed")

        # Extract bucket and blob names from model_id
        # model_id format: gs://bucket_name/path/to/model
        if model_id.startswith("gs://"):
            model_id = model_id[5:]

        parts = model_id.split("/", 1)
        if len(parts) != 2 or parts[0] != self.bucket_name:
            raise ValueError("Invalid model_id format: {model_id}")

        blob_name = parts[1]

        # Download the model file
        blob = self.bucket.blob(blob_name)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        blob.download_to_filename(destination_path)

        logger.info(
            "Model downloaded from gs://{self.bucket_name}/{blob_name} to {destination_path}"
        )

        return destination_path

    def list_models(self) -> list[dict]:
        """
        List all models in the GCS registry.

        Returns:
            List of dictionaries with model information
        """
        if self.storage_client is None:
            raise ImportError("google-cloud-storage not installed")

        # List all metadata files
        blobs = self.bucket.list_blobs(prefix="{self.base_path}/")
        metadata_blobs = [
            blob for blob in blobs if blob.name.endswith("/metadata.json")
        ]

        models = []
        for blob in metadata_blobs:
            # Download and parse metadata
            metadata_content = blob.download_as_text()
            metadata = json.loads(metadata_content)

            # Add to results
            models.append(metadata)

        return models


class VertexAIModelRegistry(ModelRegistry):
    """
    Model registry implementation using Vertex AI Model Registry.
    """

    def __init__(self, project_id: str, region: str = "us-central1"):
        """
        Initialize the Vertex AI model registry.

        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
        """
        self.project_id = project_id
        self.region = region

        # Import Vertex AI client
        try:
            from google.cloud import aiplatform

            aiplatform.init(project=project_id, location=region)
            self.aiplatform = aiplatform
        except ImportError:
            logger.warning(
                "google-cloud-aiplatform not installed. Vertex AI functionality will be limited."
            )
            self.aiplatform = None

    def save_model(self, model_path: str, metadata: dict) -> str:
        """
        Save a model to Vertex AI Model Registry.

        Args:
            model_path: Path to the model file
            metadata: Dictionary of metadata about the model

        Returns:
            Vertex AI model ID
        """
        if self.aiplatform is None:
            raise ImportError("google-cloud-aiplatform not installed")

        # First upload the model to a temporary GCS location
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path)
        model_display_name = "{metadata.get('name', 'trading_model')}_{timestamp}"

        # Create a Cloud Storage staging location
        gcs_directory = (
            "gs://{self.project_id}-model-artifacts/vertex-staging/{model_display_name}"
        )

        # Upload to GCS first
        from google.cloud import storage

        storage_client = storage.Client()
        bucket_name = "{self.project_id}-model-artifacts"
        bucket = storage_client.bucket(bucket_name)
        blob_name = "vertex-staging/{model_display_name}/{model_name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(model_path)

        # Import the model to Vertex AI
        model = self.aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri="{gcs_directory}",
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
            metadata=metadata,
        )

        logger.info("Model saved to Vertex AI with ID: {model.resource_name}")

        return model.resource_name

    def load_model(self, model_id: str, destination_path: str) -> str:
        """
        Load a model from Vertex AI Model Registry.

        Args:
            model_id: Vertex AI model ID
            destination_path: Local path to save the model

        Returns:
            Local path to the loaded model
        """
        if self.aiplatform is None:
            raise ImportError("google-cloud-aiplatform not installed")

        # Get the model from Vertex AI
        model = self.aiplatform.Model(model_id)

        # Download artifacts
        gcs_uri = model.gcs_uri

        # Use GCS client to download
        from google.cloud import storage

        storage_client = storage.Client()

        # Parse the GCS URI
        if gcs_uri.startswith("gs://"):
            gcs_uri = gcs_uri[5:]

        bucket_name, blob_prefix = gcs_uri.split("/", 1)
        bucket = storage_client.bucket(bucket_name)

        # List all blobs in the model directory
        blobs = list(bucket.list_blobs(prefix=blob_prefix))

        # Find the model file (usually a .pkl or .joblib file)
        model_blob = next(
            (blob for blob in blobs if blob.name.endswith((".pkl", ".joblib"))), None
        )

        if model_blob is None:
            raise ValueError("No model file found in {gcs_uri}")

        # Download the model file
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        model_blob.download_to_filename(destination_path)

        logger.info("Model downloaded from Vertex AI to {destination_path}")

        return destination_path

    def list_models(self) -> list[dict]:
        """
        List all models in the Vertex AI Model Registry.

        Returns:
            List of dictionaries with model information
        """
        if self.aiplatform is None:
            raise ImportError("google-cloud-aiplatform not installed")

        # List models from Vertex AI
        models = self.aiplatform.Model.list()

        results = []
        for model in models:
            model_info = {
                "id": model.resource_name,
                "display_name": model.display_name,
                "create_time": model.create_time.isoformat(),
                "update_time": model.update_time.isoformat(),
                "metadata": model.metadata,
            }
            results.append(model_info)

        return results


def get_model_registry(registry_type: str = "gcs", **kwargs) -> ModelRegistry:
    """
    Get an instance of a model registry.

    Args:
        registry_type: Type of registry ('gcs' or 'vertex')
        **kwargs: Additional arguments for the registry constructor

    Returns:
        An instance of a ModelRegistry implementation
    """
    if registry_type.lower() == "gcs":
        if "bucket_name" not in kwargs:
            raise ValueError("bucket_name is required for GCS registry")
        return GCSModelRegistry(**kwargs)

    elif registry_type.lower() == "vertex":
        if "project_id" not in kwargs:
            raise ValueError("project_id is required for Vertex AI registry")
        return VertexAIModelRegistry(**kwargs)

    else:
        raise ValueError("Unsupported registry type: {registry_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage AI Trading Machine models")
    parser.add_argument(
        "--action",
        choices=["save", "load", "list"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument(
        "--registry-type",
        choices=["gcs", "vertex"],
        default="gcs",
        help="Type of model registry",
    )
    parser.add_argument("--bucket-name", help="GCS bucket name (for GCS registry)")
    parser.add_argument("--project-id", help="GCP project ID (for Vertex AI registry)")
    parser.add_argument("--model-path", help="Path to the model file (for save action)")
    parser.add_argument("--model-id", help="Model ID or path (for load action)")
    parser.add_argument("--destination", help="Destination path (for load action)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create registry
    registry_kwargs = {}
    if args.registry_type == "gcs" and args.bucket_name:
        registry_kwargs["bucket_name"] = args.bucket_name
    elif args.registry_type == "vertex" and args.project_id:
        registry_kwargs["project_id"] = args.project_id

    try:
        registry = get_model_registry(args.registry_type, **registry_kwargs)

        if args.action == "save":
            if not args.model_path:
                raise ValueError("--model-path is required for save action")

            # Sample metadata
            metadata = {
                "name": os.path.splitext(os.path.basename(args.model_path))[0],
                "description": "Trading model saved from command line",
                "framework": "scikit-learn",
                "created_by": os.environ.get("USER", "unknown"),
            }

            registry.save_model(args.model_path, metadata)

        elif args.action == "load":
            if not args.model_id or not args.destination:
                raise ValueError(
                    "--model-id and --destination are required for load action"
                )

            registry.load_model(args.model_id, args.destination)

        elif args.action == "list":
            models = registry.list_models()
            print("Found {len(models)} models:")
            for model in models:
                print(
                    "- {model.get('display_name', model.get('name', 'Unknown'))}: {model.get('id', model.get('model_path', 'Unknown'))}"
                )

    except ImportError as e:
        logger.error("Required package not installed: {str(e)}")
        logger.info(
            "Install requirements with: pip install google-cloud-storage google-cloud-aiplatform"
        )
    except Exception as e:
        logger.error("Error: {str(e)}")
