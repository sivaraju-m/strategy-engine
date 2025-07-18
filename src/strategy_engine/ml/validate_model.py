"""
AI Trading Machine – Step 2: Model Validation Pipeline

This module is responsible for validating trained machine learning models
by evaluating their performance on test data.

It handles:
1. Loading a trained model from disk
2. Loading validation/test set features and actual signals
3. Evaluating the model using multiple metrics
4. Visualizing performance (confusion matrix, feature importance)
5. Logging results to structured files and optionally to BigQuery

Usage:
    python -m src.ai_trading_machine.ml.validate_model --model-path models/INFY.NS_model.pkl --data-path data/test/INFY.NS.csv --ticker INFY.NS
"""

import argparse
import datetime
import json
import logging
import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Class for validating trained machine learning models.
    """

    def __init__(
        self,
        model_path: str,
        ticker: str,
        output_dir: str = "logs/model_metrics",
        visualize: bool = True,
    ):
        """
        Initialize the model validator.

        Args:
            model_path: Path to the trained model file
            ticker: Ticker symbol for the model
            output_dir: Directory to save validation results
            visualize: Whether to generate visualizations
        """
        self.model_path = model_path
        self.ticker = ticker
        self.output_dir = output_dir
        self.visualize = visualize
        self.model = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load the model
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        logger.info("Loading model from {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)

            # Get model type
            self.model_type = type(self.model).__name__
            logger.info("Loaded {self.model_type} model successfully")
        except Exception as e:
            logger.error("Failed to load model: {str(e)}")
            raise

    def validate(
        self,
        features: pd.DataFrame,
        actual_signals: pd.Series,
        return_series: Optional[pd.Series] = None,
    ) -> dict:
        """
        Validate the model using test data.

        Args:
            features: Features for validation
            actual_signals: Actual signals (ground truth)
            return_series: Optional return series for financial metrics

        Returns:
            Dictionary of validation metrics
        """
        logger.info("Validating {self.model_type} model for {self.ticker}")

        # Cache features for later use (e.g., ROC-AUC calculation)
        self.last_features = features

        # Make predictions
        predicted_signals = self.model.predict(features)

        # Calculate classification metrics
        metrics = self._calculate_classification_metrics(
            actual_signals, predicted_signals
        )

        # Calculate financial metrics if return series is provided
        if return_series is not None:
            financial_metrics = self._calculate_financial_metrics(
                predicted_signals, return_series
            )
            metrics.update(financial_metrics)

        # Get feature importance if available
        if hasattr(self.model, "feature_importances_"):
            metrics["feature_importance"] = self._get_feature_importance(
                features.columns
            )

        # Add metadata
        metrics.update(
            {
                "ticker": self.ticker,
                "model": self.model_type,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

        # Create visualizations if enabled
        if self.visualize:
            self._create_visualizations(
                actual_signals, predicted_signals, features.columns
            )

        logger.info(
            "Validation complete: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
        )
        return metrics

    def _calculate_classification_metrics(
        self, actual: pd.Series, predicted: np.ndarray
    ) -> dict:
        """Calculate classification performance metrics."""
        metrics = {
            "accuracy": accuracy_score(actual, predicted),
            "precision": precision_score(actual, predicted, average="weighted"),
            "recall": recall_score(actual, predicted, average="weighted"),
            "f1_score": f1_score(actual, predicted, average="weighted"),
        }

        # Calculate ROC-AUC if applicable (for multi-class, need to binarize)
        try:
            if hasattr(self.model, "predict_proba"):
                # For multi-class classification
                y_prob = self.model.predict_proba(self.last_features)
                metrics["roc_auc"] = roc_auc_score(
                    pd.get_dummies(actual),
                    y_prob,
                    multi_class="ovr",
                    average="weighted",
                )
        except Exception as e:
            logger.warning("Could not calculate ROC-AUC: {str(e)}")

        # Get classification report as a string
        metrics["classification_report"] = classification_report(actual, predicted)

        # Calculate confusion matrix
        cm = confusion_matrix(actual, predicted)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    def _calculate_financial_metrics(
        self, signals: np.ndarray, returns: pd.Series
    ) -> dict:
        """Calculate financial performance metrics."""
        # Convert signals to positions (-1, 0, 1)
        positions = signals

        # Calculate strategy returns (position * next day's return)
        strategy_returns = positions * returns.shift(-1)

        # Drop NaN values
        strategy_returns = strategy_returns.dropna()

        # Calculate Sharpe ratio (annualized)
        mean_daily_return = strategy_returns.mean()
        std_daily_return = strategy_returns.std()

        if std_daily_return > 0:
            sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
        else:
            sharpe_ratio = float("nan")

        # Calculate cumulative return
        cumulative_return = (1 + strategy_returns).cumprod().iloc[-1] - 1

        return {
            "sharpe_ratio": sharpe_ratio,
            "mean_daily_return": mean_daily_return,
            "cumulative_return": cumulative_return,
        }

    def _get_feature_importance(self, feature_names: list[str]) -> dict:
        """Get feature importance from the model."""
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_

            # Create sorted dictionary of feature importances
            feature_importance = {"features": [], "importance": []}

            # Sort features by importance
            indices = np.argsort(importance)[::-1]

            for idx in indices:
                feature_importance["features"].append(feature_names[idx])
                feature_importance["importance"].append(float(importance[idx]))

            return feature_importance

        return {}

    def _create_visualizations(
        self, actual: pd.Series, predicted: np.ndarray, feature_names: list[str]
    ) -> None:
        """Create and save visualizations of model performance."""
        # Create output directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. Confusion Matrix
        self._plot_confusion_matrix(actual, predicted, vis_dir)

        # 2. Feature Importance
        if hasattr(self.model, "feature_importances_"):
            self._plot_feature_importance(feature_names, vis_dir)

    def _plot_confusion_matrix(
        self, actual: pd.Series, predicted: np.ndarray, vis_dir: str
    ) -> None:
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))

        # Get confusion matrix
        cm = confusion_matrix(actual, predicted)

        # Get unique classes and their labels
        classes = sorted(actual.unique())
        class_labels = {-1: "Sell", 0: "Hold", 1: "Buy"}
        labels = [class_labels.get(c, str(c)) for c in classes]

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.title("Confusion Matrix for {self.ticker} - {self.model_type}")
        plt.ylabel("Actual Signal")
        plt.xlabel("Predicted Signal")

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "{self.ticker}_confusion_matrix.png"))
        plt.close()

    def _plot_feature_importance(self, feature_names: list[str], vis_dir: str) -> None:
        """Plot and save feature importance."""
        if not hasattr(self.model, "feature_importances_"):
            return

        plt.figure(figsize=(12, 8))

        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]

        # Plot only top 15 features if there are many
        n_features = min(15, len(feature_names))
        top_indices = indices[:n_features]

        plt.barh(range(n_features), importance[top_indices], align="center")

        plt.yticks(range(n_features), [feature_names[i] for i in top_indices])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance for {self.ticker} - {self.model_type}")

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "{self.ticker}_feature_importance.png"))
        plt.close()

    def save_results(self, metrics: dict) -> str:
        """Save validation results to a JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(self.output_dir, "{self.ticker}_{timestamp}.json")

        # Create a JSON-serializable version of the metrics
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, np.float64) or isinstance(value, np.float32):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.int64) or isinstance(value, np.int32):
                serializable_metrics[key] = int(value)
            else:
                serializable_metrics[key] = value

        # Save to JSON file
        with open(output_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        logger.info("Validation results saved to {output_file}")
        return output_file

    def push_to_bigquery(
        self, metrics: dict, project_id: str, dataset_id: str, table_id: str
    ) -> bool:
        """
        Push validation results to BigQuery.

        Args:
            metrics: Dictionary of validation metrics
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID

        Returns:
            True if successful, False otherwise
        """
        try:
            from google.cloud import bigquery

            # Create BigQuery client
            client = bigquery.Client(project=project_id)

            # Create a reference to the table
            table_ref = "{project_id}.{dataset_id}.{table_id}"

            # Convert metrics to DataFrame for easier insertion
            metrics_df = pd.DataFrame([metrics])

            # Load the data into BigQuery
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
            )

            job = client.load_table_from_dataframe(
                metrics_df, table_ref, job_config=job_config
            )

            # Wait for the job to complete
            job.result()

            logger.info("Validation results pushed to BigQuery: {table_ref}")
            return True

        except Exception as e:
            logger.error("Failed to push results to BigQuery: {str(e)}")
            return False


def load_test_data(
    data_path: str,
    ticker: Optional[str] = None,
    signal_column: str = "signal",
    return_column: Optional[str] = None,
    date_column: Optional[str] = "date",
) -> tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Load test data for model validation.

    Args:
        data_path: Path to the test data file
        ticker: Optional ticker symbol for filtering
        signal_column: Name of the signal column
        return_column: Optional name of the return column
        date_column: Optional name of the date column

    Returns:
        Tuple of (features, actual signals, returns)
    """
    logger.info("Loading test data from {data_path}")

    # Load data based on file extension
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    elif data_path.endswith(".parquet"):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format: {data_path}")

    # Filter by ticker if provided
    if ticker and "ticker" in data.columns:
        data = data[data["ticker"] == ticker]

    # Extract date column if present (for time series validation)
    if date_column and date_column in data.columns:
        date_series = data[date_column]
        data = data.drop(columns=[date_column])
    else:
        date_series = None

    # Extract signal column
    if signal_column not in data.columns:
        raise ValueError("Signal column '{signal_column}' not found in data")

    signals = data[signal_column]

    # Extract return column if provided
    if return_column and return_column in data.columns:
        returns = data[return_column]
    else:
        returns = None

    # Features are all columns except signal, date, ticker, and return
    exclude_cols = [
        c for c in [signal_column, "ticker", return_column] if c in data.columns
    ]
    features = data.drop(columns=exclude_cols)

    logger.info("Loaded {len(features)} samples with {len(features.columns)} features")

    return features, signals, returns


def main() -> None:
    """Main function to run model validation."""
    parser = argparse.ArgumentParser(description="Validate ML models for trading")

    # Required arguments
    parser.add_argument(
        "--model-path", required=True, help="Path to the trained model file"
    )
    parser.add_argument("--data-path", required=True, help="Path to the test data file")
    parser.add_argument("--ticker", required=True, help="Ticker symbol for the model")

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        default="logs/model_metrics",
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--signal-column",
        default="signal",
        help="Name of the signal column in test data",
    )
    parser.add_argument(
        "--return-column", help="Name of the return column in test data"
    )
    parser.add_argument(
        "--date-column", default="date", help="Name of the date column in test data"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )

    # BigQuery arguments
    parser.add_argument(
        "--push-to-bigquery", action="store_true", help="Push results to BigQuery"
    )
    parser.add_argument("--project-id", help="GCP project ID for BigQuery")
    parser.add_argument(
        "--dataset-id", default="trading_ml", help="BigQuery dataset ID"
    )
    parser.add_argument(
        "--table-id", default="ml_validation_results", help="BigQuery table ID"
    )

    args = parser.parse_args()

    try:
        # Load test data
        features, signals, returns = load_test_data(
            args.data_path,
            args.ticker,
            args.signal_column,
            args.return_column,
            args.date_column,
        )

        # Initialize validator
        validator = ModelValidator(
            args.model_path, args.ticker, args.output_dir, args.visualize
        )

        # Validate model
        metrics = validator.validate(features, signals, returns)

        # Save results
        output_file = validator.save_results(metrics)

        # Push to BigQuery if requested
        if args.push_to_bigquery:
            if not args.project_id:
                logger.warning("--project-id is required for BigQuery integration")
            else:
                validator.push_to_bigquery(
                    metrics, args.project_id, args.dataset_id, args.table_id
                )

        # Print summary
        print("\n=== Model Validation Summary ===")
        print("Ticker: {args.ticker}")
        print("Model: {metrics['model']}")
        print("Accuracy: {metrics['accuracy']:.4f}")
        print("Precision: {metrics['precision']:.4f}")
        print("Recall: {metrics['recall']:.4f}")
        print("F1 Score: {metrics['f1_score']:.4f}")

        if "sharpe_ratio" in metrics:
            print("Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")

        if "feature_importance" in metrics:
            print("\nTop Features:")
            for i, feature in enumerate(metrics["feature_importance"]["features"][:5]):
                importance = metrics["feature_importance"]["importance"][i]
                print("  {feature}: {importance:.4f}")

        print("\nDetailed results saved to: {output_file}")

    except Exception as e:
        logger.error("Validation failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
