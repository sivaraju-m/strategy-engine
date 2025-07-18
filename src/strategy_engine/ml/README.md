# Machine Learning Module

This directory contains the machine learning components of the AI Trading Machine project.

## Directory Structure

- `train_model.py`: Model training pipeline that works with features and signals
- `predict.py`: Inference API and batch prediction functionality
- `features/`: Feature generation modules
- `model_registry/`: Model storage and versioning using GCS or Vertex AI

## Usage

### Training a Model

```python
from ai_trading_machine.ml.train_model import ModelTrainer

# Initialize the trainer
trainer = ModelTrainer(
    model_type="random_forest",
    model_params={"n_estimators": 100, "max_depth": 10}
)

# Train the model
model, metrics = trainer.train(features_df, labels_series)

# Save the model
model_path = trainer.save_model("models/my_model.joblib")
```

### Generating Features

```python
from ai_trading_machine.ml.features import create_default_pipeline

# Create a feature pipeline
pipeline = create_default_pipeline()

# Generate features
features_df = pipeline.generate_all(market_data_df)
```

### Making Predictions

```python
from ai_trading_machine.ml.predict import ModelPredictor

# Load a model
predictor = ModelPredictor("models/my_model.joblib")

# Make predictions
predictions = predictor.predict(features_df)
```

### Using the Model Registry

```python
from ai_trading_machine.ml.model_registry import get_model_registry

# Create a GCS-based model registry
registry = get_model_registry(
    registry_type="gcs",
    bucket_name="my-model-bucket"
)

# Save a model to the registry
model_id = registry.save_model(
    "models/my_model.joblib",
    metadata={"description": "My trading model"}
)

# List models in the registry
models = registry.list_models()

# Load a model from the registry
local_path = registry.load_model(
    model_id,
    destination_path="models/loaded_model.joblib"
)
```

## Installation

Make sure to install the required dependencies:

```bash
pip install -r requirements.txt
```

The core requirements include:
- scikit-learn
- pandas
- numpy
- joblib
- xgboost (optional)
- google-cloud-storage (for GCS model registry)
- google-cloud-aiplatform (for Vertex AI model registry)
