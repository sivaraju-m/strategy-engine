"""
Prediction module for AI Trading Machine.

This module provides functionality for making predictions with trained models,
either in batch mode or through an API.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Model predictor class for AI Trading Machine.

    This class is responsible for loading trained models and making predictions.
    """

    def __init__(self, model_path: str):
        """
        Initialize the model predictor.

        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the model from the specified path.

        Returns:
            The loaded model
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model file not found: {self.model_path}")

        logger.info("Loading model from {self.model_path}")
        return joblib.load(self.model_path)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.

        Args:
            features: DataFrame containing the features

        Returns:
            Array of predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet.")

        logger.info("Making predictions for {len(features)} samples")
        return self.model.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the loaded model.

        Args:
            features: DataFrame containing the features

        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet.")

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support probability predictions")

        logger.info("Making probability predictions for {len(features)} samples")
        return self.model.predict_proba(features)


def batch_predict(
    model_path: str,
    feature_data_path: str,
    output_predictions_path: str,
    include_probabilities: bool = False,
) -> str:
    """
    Perform batch prediction on a dataset.

    Args:
        model_path: Path to the trained model
        feature_data_path: Path to the feature data CSV
        output_predictions_path: Path to save the predictions
        include_probabilities: Whether to include prediction probabilities

    Returns:
        Path to the saved predictions
    """
    # Load the data
    logger.info("Loading data from {feature_data_path}")
    features = pd.read_csv(feature_data_path)

    # Initialize the predictor
    predictor = ModelPredictor(model_path)

    # Make predictions
    predictions = predictor.predict(features)

    # Create a DataFrame with the predictions
    results = features.copy()
    results["prediction"] = predictions

    # Add probability predictions if requested
    if include_probabilities:
        try:
            proba = predictor.predict_proba(features)

            # Add each class probability as a column
            for i in range(proba.shape[1]):
                results["probability_class_{i}"] = proba[:, i]

        except AttributeError:
            logger.warning("Model does not support probability predictions. Skipping.")

    # Save the predictions
    os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
    results.to_csv(output_predictions_path, index=False)
    logger.info("Predictions saved to {output_predictions_path}")

    return output_predictions_path


# API Endpoint for Predictions
def predict_api(model_path: str, features_dict: dict) -> dict:
    """
    Make a prediction for API requests.

    Args:
        model_path: Path to the trained model
        features_dict: Dictionary of feature values

    Returns:
        Dictionary with prediction results
    """
    try:
        # Convert features dictionary to DataFrame
        features = pd.DataFrame([features_dict])

        # Initialize the predictor
        predictor = ModelPredictor(model_path)

        # Make predictions
        prediction = predictor.predict(features)[0]

        # Get probabilities if available
        probabilities = {}
        try:
            proba = predictor.predict_proba(features)[0]
            for i, p in enumerate(proba):
                probabilities["class_{i}"] = float(p)
        except (AttributeError, ValueError):
            pass

        # Return the results
        result = {
            "prediction": (
                int(prediction)
                if isinstance(prediction, (np.integer, int))
                else float(prediction)
            ),
            "probabilities": probabilities,
        }

        return result

    except Exception as e:
        logger.error("Prediction error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Make predictions with a trained ML model"
    )
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument(
        "--feature-data", required=True, help="Path to the feature data CSV"
    )
    parser.add_argument("--output", required=True, help="Path to save the predictions")
    parser.add_argument(
        "--include-probabilities",
        action="store_true",
        help="Include prediction probabilities",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    batch_predict(
        model_path=args.model,
        feature_data_path=args.feature_data,
        output_predictions_path=args.output,
        include_probabilities=args.include_probabilities,
    )
