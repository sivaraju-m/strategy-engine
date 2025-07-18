#!/usr/bin/env python3
"""
Simple script to train a model on INFY.NS data
"""

import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Load the data
print("Loading data...")
data_path = "../../../data/cleaned/INFY.NS.csv"
data = pd.read_csv(data_path)

# Display data info
print("Data shape: {data.shape}")
print("Columns: {data.columns.tolist()}")
print("Signal distribution: {data['signal'].value_counts().to_dict()}")

# Split features and target
X = data.drop(columns=["signal", "date"])  # Exclude date and signal
y = data["signal"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set shape: {X_train.shape}")
print("Test set shape: {X_test.shape}")

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

model.fit(X_train, y_train)  # Random Forest can handle -1, 0, 1 directly

# Evaluate model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert predictions back to original range (no conversion needed for RandomForest)
# y_train_pred and y_test_pred are already in the correct range [-1, 0, 1]

# Calculate metrics
metrics = {
    "train_accuracy": accuracy_score(y_train, y_train_pred),
    "test_accuracy": accuracy_score(y_test, y_test_pred),
    "train_precision": precision_score(y_train, y_train_pred, average="weighted"),
    "test_precision": precision_score(y_test, y_test_pred, average="weighted"),
    "train_recall": recall_score(y_train, y_train_pred, average="weighted"),
    "test_recall": recall_score(y_test, y_test_pred, average="weighted"),
    "train_f1": f1_score(y_train, y_train_pred, average="weighted"),
    "test_f1": f1_score(y_test, y_test_pred, average="weighted"),
}

# Print metrics
print("\nModel performance metrics:")
for metric_name, metric_value in metrics.items():
    print("  {metric_name}: {metric_value:.4f}")

# Save model
import joblib

output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "INFY.NS_model.pkl")
joblib.dump(model, output_path)
print("\nModel saved to: {output_path}")

# Feature importance
if hasattr(model, "feature_importances_"):
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop features by importance:")
    print(feature_importance.head(10))

    # Save feature importance
    importance_path = os.path.join(output_dir, "INFY.NS_feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print("Feature importance saved to: {importance_path}")
