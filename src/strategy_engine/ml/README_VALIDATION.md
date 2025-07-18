# Model Validation Pipeline

This directory contains the validation pipeline for machine learning models in the AI Trading Machine. The validation pipeline evaluates model performance using classification metrics, financial metrics, and visualizations.

## Components

- `validate_model.py`: Core validation module with the `ModelValidator` class
- `batch_validate_models.py`: Script for batch validation of multiple models

## Features

- **Classification Metrics**: Accuracy, precision, recall, F1 score, ROC-AUC, confusion matrix
- **Financial Metrics**: Sharpe ratio, mean daily return, cumulative return
- **Visualizations**: Confusion matrix and feature importance charts
- **Results Logging**: Structured JSON outputs and optional BigQuery integration
- **Batch Processing**: Validate multiple models in one run

## Usage

For detailed usage instructions, see the [Model Validation Guide](../../docs/model_validation_guide.md).

### Basic Usage

```python
from src.ai_trading_machine.ml.validate_model import ModelValidator, load_test_data

# Load test data
features, signals, returns = load_test_data(
    'data/test/TICKER.csv',
    'TICKER',
    return_column='pct_return'
)

# Initialize validator
validator = ModelValidator(
    'models/TICKER_model.pkl',
    'TICKER',
    visualize=True
)

# Validate model
metrics = validator.validate(features, signals, returns)

# Save results
output_file = validator.save_results(metrics)
```

### CLI Usage

```bash
# Single model validation
python -m src.ai_trading_machine.ml.validate_model \
  --model-path models/TICKER_model.pkl \
  --data-path data/test/TICKER.csv \
  --ticker TICKER \
  --visualize

# Batch validation
python batch_validate_models.py --tickers TICKER1 TICKER2
```

## Output

The validation pipeline generates:

1. **JSON Metrics File**: `logs/model_metrics/TICKER_DATE.json`
2. **Confusion Matrix**: `logs/model_metrics/visualizations/TICKER_confusion_matrix.png`
3. **Feature Importance**: `logs/model_metrics/visualizations/TICKER_feature_importance.png`
4. **Log File**: `logs/batch_validation.log` (for batch validation)
