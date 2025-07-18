"""
AI Trading Machine – ML Model Trainer

This module is responsible for training machine learning models using features and signals.
It handles model creation, training, evaluation, and saving to the model registry.

The pipeline:
1. Loads features and signals from cleaned GCS data, BigQuery, or local files (CSV/Parquet)
2. Engineers features like RSI, momentum, volatility, etc. using helper methods
3. Trains an XGBoost or LightGBM model to predict Buy/Sell/Hold signals
4. Saves the model as a .pkl file or pushes to GCS/Vertex AI Model Registry
5. Supports time-based train/test split for financial data
6. Evaluates model performance with appropriate metrics (accuracy, precision, recall, F1)
7. Tracks feature importance for model interpretability

Usage:
    python -m src.ai_trading_machine.ml.train_model --source-type csv \
        --source-path data/cleaned/INFY.NS.csv \
        --ticker INFY.NS \
        --target-column signal \
        --output-model models/INFY.NS_model.pkl \
        --model-type lightgbm \
        --test-split-date 2024-01-01
"""

import logging
import os
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Set up logging
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model trainer class for AI Trading Machine.

    This class is responsible for training and evaluating machine learning models
    for predicting market signals (Buy/Sell/Hold).
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        model_params: Optional[dict] = None,
        test_size: float = 0.2,
        test_split_date: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model to train (random_forest, xgboost, lightgbm)
            model_params: Parameters for the model
            test_size: Size of the test set (used if test_split_date is None)
            test_split_date: Date to split train/test data (takes precedence over test_size)
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.test_size = test_size
        self.test_split_date = test_split_date
        self.random_state = random_state
        self.model = None
        self.feature_importance = None

    def _initialize_model(self) -> BaseEstimator:
        """
        Initialize the model based on model_type.

        Returns:
            An initialized model
        """
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                max_depth=self.model_params.get("max_depth", None),
                random_state=self.random_state,
                **{
                    k: v
                    for k, v in self.model_params.items()
                    if k not in ["n_estimators", "max_depth"]
                },
            )

        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    n_estimators=self.model_params.get("n_estimators", 100),
                    max_depth=self.model_params.get("max_depth", 6),
                    learning_rate=self.model_params.get("learning_rate", 0.1),
                    objective=self.model_params.get("objective", "multi:softprob"),
                    num_class=self.model_params.get("num_class", 3),  # Buy/Sell/Hold
                    random_state=self.random_state,
                    **{
                        k: v
                        for k, v in self.model_params.items()
                        if k
                        not in [
                            "n_estimators",
                            "max_depth",
                            "learning_rate",
                            "objective",
                            "num_class",
                        ]
                    },
                )
            except ImportError:
                logger.warning("XGBoost not installed. Falling back to RandomForest.")
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                )

        elif self.model_type == "lightgbm":
            try:
                import lightgbm as lgb

                return lgb.LGBMClassifier(
                    n_estimators=self.model_params.get("n_estimators", 100),
                    max_depth=self.model_params.get("max_depth", 6),
                    learning_rate=self.model_params.get("learning_rate", 0.1),
                    objective=self.model_params.get("objective", "multiclass"),
                    num_class=self.model_params.get("num_class", 3),  # Buy/Sell/Hold
                    random_state=self.random_state,
                    **{
                        k: v
                        for k, v in self.model_params.items()
                        if k
                        not in [
                            "n_estimators",
                            "max_depth",
                            "learning_rate",
                            "objective",
                            "num_class",
                        ]
                    },
                )
            except ImportError:
                logger.warning(
                    "LightGBM not installed. Falling back to XGBoost or RandomForest."
                )
                try:
                    import xgboost as xgb

                    return xgb.XGBClassifier(
                        n_estimators=100, max_depth=6, random_state=self.random_state
                    )
                except ImportError:
                    from sklearn.ensemble import RandomForestClassifier

                    return RandomForestClassifier(
                        n_estimators=100, random_state=self.random_state
                    )

        else:
            raise ValueError("Unsupported model type: {self.model_type}")

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        date_column: Optional[str] = None,
    ) -> tuple[BaseEstimator, dict[str, float]]:
        """
        Train the model on the provided features and labels.

        Args:
            features: DataFrame containing the features
            labels: Series containing the labels (e.g., -1=Sell, 0=Hold, 1=Buy)
            date_column: Name of date column for time-based train/test split

        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        # Initialize the model
        self.model = self._initialize_model()

        # Split the data into training and testing sets
        if self.test_split_date and date_column and date_column in features.columns:
            logger.info(
                "Using time-based split with test data after {self.test_split_date}"
            )
            train_mask = features[date_column] < self.test_split_date
            X_train = features[train_mask].drop(columns=[date_column])
            X_test = features[~train_mask].drop(columns=[date_column])
            y_train = labels[train_mask]
            y_test = labels[~train_mask]
        else:
            logger.info("Using random split with test_size={self.test_size}")
            if date_column and date_column in features.columns:
                features_no_date = features.drop(columns=[date_column])
            else:
                features_no_date = features

            X_train, X_test, y_train, y_test = train_test_split(
                features_no_date,
                labels,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        # Train the model
        logger.info(
            "Training {self.model_type} model with {X_train.shape[0]} samples..."
        )
        self.model.fit(X_train, y_train)

        # Store feature importance if available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            logger.info("Top 10 important features:")
            logger.info(self.feature_importance.head(10))

        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)

        # Log the results
        logger.info("Model training complete. Metrics: {metrics}")

        return self.model, metrics

    def _calculate_metrics(
        self, y_train, y_train_pred, y_test, y_test_pred
    ) -> dict[str, float]:
        """Calculate and return evaluation metrics."""
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
        }

        # For multi-class, we use micro/macro averaging
        metrics.update(
            {
                "train_precision": precision_score(
                    y_train, y_train_pred, average="weighted"
                ),
                "test_precision": precision_score(
                    y_test, y_test_pred, average="weighted"
                ),
                "train_recall": recall_score(y_train, y_train_pred, average="weighted"),
                "test_recall": recall_score(y_test, y_test_pred, average="weighted"),
                "train_f1": f1_score(y_train, y_train_pred, average="weighted"),
                "test_f1": f1_score(y_test, y_test_pred, average="weighted"),
            }
        )

        return metrics

    def save_model(
        self,
        model_path: str,
        save_to_registry: bool = False,
        registry_options: Optional[dict] = None,
    ) -> str:
        """
        Save the trained model to the specified path and optionally to a model registry.

        Args:
            model_path: Path to save the model
            save_to_registry: Whether to save the model to a registry (GCS or Vertex AI)
            registry_options: Options for the model registry
                - registry_type: 'gcs' or 'vertex'
                - bucket_name: For GCS registry
                - project_id: For Vertex AI registry

        Returns:
            Path to the saved model
        """
        import joblib

        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the model
        joblib.dump(self.model, model_path)
        logger.info("Model saved to {model_path}")

        # Save to registry if requested
        if save_to_registry:
            registry_path = self._save_to_registry(model_path, registry_options or {})
            logger.info("Model also saved to registry: {registry_path}")
            return registry_path

        return model_path

    def _save_to_registry(self, model_path: str, registry_options: dict) -> str:
        """Save model to GCS or Vertex AI Model Registry."""
        try:
            # Import here to avoid dependency issues
            from strategy_engine.ml.model_registry import get_model_registry

            registry_type = registry_options.get("registry_type", "gcs")

            # Create model metadata
            metadata = {
                "name": os.path.basename(model_path),
                "model_type": self.model_type,
                "model_params": self.model_params,
                "feature_importance": (
                    self.feature_importance.to_dict()
                    if self.feature_importance is not None
                    else None
                ),
                "created_at": pd.Timestamp.now().isoformat(),
            }

            # Get registry instance
            registry = get_model_registry(
                registry_type=registry_type, **registry_options
            )

            # Save to registry
            return registry.save_model(model_path, metadata)

        except (ImportError, Exception) as e:
            logger.error("Failed to save model to registry: {str(e)}")
            logger.info("Model was saved locally only.")
            return model_path


def load_data(
    source_type: str,
    source_path: str,
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    project_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data from various sources (CSV, Parquet, GCS, BigQuery).

    Args:
        source_type: Type of source ('csv', 'parquet', 'gcs', 'bigquery')
        source_path: Path or query for the data source
        ticker: Ticker symbol (used for filtering)
        start_date: Start date for filtering
        end_date: End date for filtering
        project_id: GCP project ID (for BigQuery)

    Returns:
        DataFrame with the loaded data
    """
    logger.info("Loading data from {source_type} source: {source_path}")

    if source_type == "csv":
        data = pd.read_csv(source_path)

    elif source_type == "parquet":
        data = pd.read_parquet(source_path)

    elif source_type == "gcs":
        # Import here to avoid dependency issues
        try:
            from google.cloud import storage

            # Extract bucket and blob names
            bucket_name, blob_name = source_path.replace("gs://", "").split("/", 1)

            # Download to temp file
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            temp_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "temp_data"
            )
            blob.download_to_filename(temp_file)

            # Determine file type and load
            if blob_name.endswith(".csv"):
                data = pd.read_csv(temp_file)
            elif blob_name.endswith(".parquet"):
                data = pd.read_parquet(temp_file)
            else:
                raise ValueError("Unsupported file type: {blob_name}")

            # Clean up
            os.remove(temp_file)

        except ImportError:
            logger.error("google-cloud-storage not installed")
            raise

    elif source_type == "bigquery":
        # Import here to avoid dependency issues
        try:
            from google.cloud import bigquery

            client = bigquery.Client(project=project_id)
            query = source_path

            # Add filters if provided
            if ticker or start_date or end_date:
                if "WHERE" not in query.upper():
                    query += " WHERE"
                else:
                    query += " AND"

                filters = []
                if ticker:
                    filters.append("ticker = '{ticker}'")
                if start_date:
                    filters.append("date >= '{start_date}'")
                if end_date:
                    filters.append("date <= '{end_date}'")

                query += " " + " AND ".join(filters)

            data = client.query(query).to_dataframe()

        except ImportError:
            logger.error("google-cloud-bigquery not installed")
            raise
    else:
        raise ValueError("Unsupported source type: {source_type}")

    # Apply filters for local files
    if source_type in ["csv", "parquet"]:
        if ticker and "ticker" in data.columns:
            data = data[data["ticker"] == ticker]

        if start_date and "date" in data.columns:
            data = data[data["date"] >= start_date]

        if end_date and "date" in data.columns:
            data = data[data["date"] <= end_date]

    logger.info("Loaded {len(data)} rows with {len(data.columns)} columns")
    return data


def main(
    source_type: str,
    source_path: str,
    target_column: str,
    output_model_path: str,
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    project_id: Optional[str] = None,
    date_column: Optional[str] = "date",
    test_split_date: Optional[str] = None,
    model_type: str = "xgboost",
    model_params: Optional[dict] = None,
    save_to_registry: bool = False,
    registry_options: Optional[dict] = None,
) -> str:
    """
    Main function to train a model from a dataset.

    Args:
        source_type: Type of source ('csv', 'parquet', 'gcs', 'bigquery')
        source_path: Path or query for the data source
        target_column: Name of the target column
        output_model_path: Path to save the trained model
        ticker: Ticker symbol (used for filtering)
        start_date: Start date for filtering
        end_date: End date for filtering
        project_id: GCP project ID (for BigQuery)
        date_column: Name of the date column for time-based splits
        test_split_date: Date to use for train/test split
        model_type: Type of model to train
        model_params: Parameters for the model
        save_to_registry: Whether to save model to registry
        registry_options: Options for the model registry

    Returns:
        Path to the saved model
    """
    # Load the data
    data = load_data(
        source_type=source_type,
        source_path=source_path,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        project_id=project_id,
    )

    # Check if we need to generate additional features
    if model_params and model_params.get("generate_features", False):
        try:
            # Try to import feature generator
            from strategy_engine.ml.features import create_default_pipeline

            logger.info("Generating additional technical features")

            # Create and run feature pipeline
            pipeline = create_default_pipeline()
            data = pipeline.generate_all(data)
        except ImportError:
            logger.warning(
                "Could not import feature generator. Using existing features only."
            )

    # Split features and labels
    if target_column not in data.columns:
        raise ValueError("Target column '{target_column}' not found in data")

    features = data.drop(columns=[target_column])
    labels = data[target_column]

    # Initialize and train the model
    trainer = ModelTrainer(
        model_type=model_type,
        model_params=model_params,
        test_split_date=test_split_date,
    )

    model, metrics = trainer.train(features, labels, date_column)

    # Save the model
    saved_model_path = trainer.save_model(
        output_model_path,
        save_to_registry=save_to_registry,
        registry_options=registry_options,
    )

    return saved_model_path


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Train an ML model for trading signals"
    )

    # Data source arguments
    parser.add_argument(
        "--source-type",
        choices=["csv", "parquet", "gcs", "bigquery"],
        default="csv",
        help="Type of data source",
    )
    parser.add_argument(
        "--source-path", required=True, help="Path or query for the data source"
    )
    parser.add_argument("--ticker", help="Ticker symbol (e.g., 'INFY.NS')")
    parser.add_argument("--start-date", help="Start date for filtering (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for filtering (YYYY-MM-DD)")
    parser.add_argument("--project-id", help="GCP project ID (for BigQuery)")
    parser.add_argument("--date-column", default="date", help="Name of date column")

    # Target and output arguments
    parser.add_argument(
        "--target-column", required=True, help="Name of the target column"
    )
    parser.add_argument(
        "--output-model", required=True, help="Path to save the trained model"
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost", "lightgbm"],
        default="xgboost",
        help="Type of model to train",
    )
    parser.add_argument(
        "--model-params",
        type=json.loads,
        default={},
        help="JSON string of model parameters",
    )
    parser.add_argument(
        "--test-split-date", help="Date to use for train/test split (YYYY-MM-DD)"
    )

    # Registry arguments
    parser.add_argument(
        "--save-to-registry", action="store_true", help="Save model to registry"
    )
    parser.add_argument(
        "--registry-type",
        choices=["gcs", "vertex"],
        default="gcs",
        help="Type of model registry",
    )
    parser.add_argument("--bucket-name", help="GCS bucket name (for GCS registry)")
    parser.add_argument("--vertex-project", help="GCP project ID (for Vertex registry)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Prepare registry options if needed
    registry_options = None
    if args.save_to_registry:
        registry_options = {"registry_type": args.registry_type}
        if args.registry_type == "gcs" and args.bucket_name:
            registry_options["bucket_name"] = args.bucket_name
        elif args.registry_type == "vertex" and args.vertex_project:
            registry_options["project_id"] = args.vertex_project

    # Run the main function
    try:
        model_path = main(
            source_type=args.source_type,
            source_path=args.source_path,
            target_column=args.target_column,
            output_model_path=args.output_model,
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            project_id=args.project_id,
            date_column=args.date_column,
            test_split_date=args.test_split_date,
            model_type=args.model_type,
            model_params=args.model_params,
            save_to_registry=args.save_to_registry,
            registry_options=registry_options,
        )

        logger.info("Successfully trained and saved model to: {model_path}")

    except Exception as e:
        logger.error("Error training model: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        exit(1)
