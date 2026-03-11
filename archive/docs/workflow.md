# End-to-End MLOps Project

> From Raw Data to Production with Automation, Versioning, and Monitoring

## Table of Contents

1. [Overview](#overview) - What makes MLOps different from ML
2. [Project Structure](#project-structure) - Repository layout and conventions
3. [Phase 1: Problem Definition](#phase-1-problem-definition) - Business framing and success metrics
4. [Phase 2: Environment Setup](#phase-2-environment-setup) - Tools and reproducibility
5. [Phase 3: Data Pipeline](#phase-3-data-pipeline) - Versioning and validation
6. [Phase 4: Feature Engineering](#phase-4-feature-engineering) - Feature store patterns
7. [Phase 5: Experimentation](#phase-5-experimentation) - Tracking and comparison
8. [Phase 6: Model Training](#phase-6-model-training) - Reproducible training pipelines
9. [Phase 7: Evaluation](#phase-7-evaluation) - Metrics and validation gates
10. [Phase 8: Model Registry](#phase-8-model-registry) - Versioning and staging
11. [Phase 9: Deployment](#phase-9-deployment) - Serving patterns
12. [Phase 10: Monitoring](#phase-10-monitoring) - Drift detection and alerting
13. [Phase 11: CI/CD Pipeline](#phase-11-cicd-pipeline) - Automation end-to-end
14. [Quick Reference](#quick-reference) - Commands and tools cheatsheet

---

## Overview

**ML vs MLOps**

| Aspect | Traditional ML | MLOps |
|--------|---------------|-------|
| Code | Notebooks only | Modular Python packages |
| Data | Local files | Versioned datasets (DVC) |
| Experiments | Manual notes | Tracked (MLflow) |
| Models | Saved to disk | Registered with metadata |
| Deployment | Manual copy | Automated CI/CD |
| Monitoring | None | Drift detection, alerts |

**Core MLOps Principles:**

1. **Reproducibility** - Any experiment can be recreated exactly
2. **Automation** - Manual steps become pipelines
3. **Versioning** - Code, data, and models are all versioned
4. **Testing** - Data quality and model performance are tested
5. **Monitoring** - Production models are continuously validated

**Example Project: Customer Churn Prediction**

- **Type:** Supervised Learning / Binary Classification
- **Target:** `churn ∈ {0, 1}`
- **Business Goal:** Identify at-risk customers for retention campaigns

---

## Project Structure

```
churn-prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint, test on PR
│       ├── train.yml           # Training pipeline
│       └── deploy.yml          # Model deployment
├── data/
│   ├── raw/                    # Original data (DVC tracked)
│   ├── processed/              # Cleaned data (DVC tracked)
│   └── features/               # Feature store output
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py             # Data loading functions
│   │   ├── validate.py         # Data quality checks
│   │   └── preprocess.py       # Cleaning pipeline
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # Training logic
│   │   ├── evaluate.py         # Evaluation metrics
│   │   └── predict.py          # Inference logic
│   └── monitoring/
│       ├── __init__.py
│       └── drift.py            # Drift detection
├── tests/
│   ├── test_data.py            # Data validation tests
│   ├── test_features.py        # Feature tests
│   └── test_model.py           # Model tests
├── notebooks/
│   ├── 01_eda.ipynb            # Exploration only
│   └── 02_prototyping.ipynb    # Quick experiments
├── configs/
│   ├── config.yaml             # Hyperparameters
│   └── logging.yaml            # Logging config
├── models/                     # Local model artifacts
├── mlruns/                     # MLflow tracking (local)
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC pipeline state
├── params.yaml                 # DVC parameters
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
├── Dockerfile                  # Container for serving
├── docker-compose.yml          # Local dev environment
└── README.md                   # Project documentation
```

---

## Phase 1: Problem Definition

### Business Context

```yaml
# configs/problem.yaml
project:
  name: customer-churn-prediction
  owner: data-science-team
  version: 1.0.0

business:
  problem: "High customer churn rate impacting revenue"
  goal: "Reduce churn by 15% through targeted retention"
  stakeholders:
    - marketing
    - customer-success
    - finance

ml_framing:
  type: supervised
  task: binary_classification
  target: churn
  positive_class: 1  # churned

success_metrics:
  business:
    - name: retention_rate
      target: "+15%"
    - name: campaign_roi
      target: ">3x"
  technical:
    - name: recall
      target: ">0.80"
      rationale: "Missing churners is expensive"
    - name: precision
      target: ">0.60"
      rationale: "Avoid wasting retention budget"
    - name: auc_roc
      target: ">0.85"

constraints:
  latency_ms: 100
  model_size_mb: 50
  interpretability: required
```

### Cost Matrix

```
                    Predicted
                 Churn    No Churn
Actual  Churn   [  0  ]  [ -500  ]  ← Missed churner (False Negative)
        Stay    [ -50 ]  [   0   ]  ← Unnecessary offer (False Positive)
```

**Key Insight:** False Negatives cost 10x more than False Positives → Optimize for **Recall**.

---

## Phase 2: Environment Setup

### Dependencies

```txt
# requirements.txt
# Core ML
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0

# MLOps
mlflow>=2.9.0
dvc>=3.30.0
dvc-s3>=3.0.0  # or dvc-gcs for GCP

# Data Quality
great-expectations>=0.18.0
pandera>=0.17.0

# Serving
fastapi>=0.104.0
uvicorn>=0.24.0

# Monitoring
evidently>=0.4.0
prometheus-client>=0.19.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Explainability
shap>=0.43.0
```

### Setup Script

```bash
#!/bin/bash
# scripts/setup.sh

set -e

echo "Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Initializing DVC..."
dvc init
dvc remote add -d storage s3://mlops-bucket/churn-project
# Or local: dvc remote add -d storage /data/dvc-storage

echo "Initializing MLflow..."
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 --port 5000 &

echo "Setup complete!"
echo "MLflow UI: http://localhost:5000"
```

### Docker Environment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/production/model.pkl

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mlflow:
    image: python:3.11-slim
    command: >
      bash -c "pip install mlflow &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri sqlite:///mlflow.db
               --default-artifact-root /mlruns"
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlruns
      - mlflow-db:/mlflow.db

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

volumes:
  mlflow-data:
  mlflow-db:
```

---

## Phase 3: Data Pipeline

### Data Versioning with DVC

```yaml
# dvc.yaml
stages:
  load_data:
    cmd: python src/data/load.py
    deps:
      - src/data/load.py
    outs:
      - data/raw/churn.csv

  validate_data:
    cmd: python src/data/validate.py
    deps:
      - src/data/validate.py
      - data/raw/churn.csv
    metrics:
      - reports/data_quality.json:
          cache: false

  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - src/data/preprocess.py
      - data/raw/churn.csv
    params:
      - preprocess.test_size
      - preprocess.random_state
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
```

### Data Loading

```python
# src/data/load.py
"""Data loading with source tracking."""
import pandas as pd
from pathlib import Path
import hashlib
import json
from datetime import datetime


def load_raw_data(source_path: str, output_path: str = "data/raw/churn.csv"):
    """Load raw data and create metadata."""
    df = pd.read_csv(source_path)

    # Create data fingerprint
    data_hash = hashlib.md5(df.to_json().encode()).hexdigest()

    metadata = {
        "source": source_path,
        "loaded_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "hash": data_hash
    }

    # Save data and metadata
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    with open(output_path.replace(".csv", "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Loaded {len(df)} rows, hash: {data_hash[:8]}")
    return df


if __name__ == "__main__":
    load_raw_data("data/source/telco_churn.csv")
```

### Data Validation

```python
# src/data/validate.py
"""Data quality validation using Pandera."""
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import json
from pathlib import Path


# Define expected schema
churn_schema = DataFrameSchema({
    "customerID": Column(str, Check.str_length(min_value=1)),
    "tenure": Column(int, Check.in_range(0, 100)),
    "MonthlyCharges": Column(float, Check.in_range(0, 500)),
    "TotalCharges": Column(float, Check.ge(0), nullable=True),
    "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
    "Churn": Column(str, Check.isin(["Yes", "No"])),
})


def validate_data(input_path: str = "data/raw/churn.csv"):
    """Validate data quality and generate report."""
    df = pd.read_csv(input_path)

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "schema_valid": False,
        "errors": []
    }

    try:
        churn_schema.validate(df, lazy=True)
        report["schema_valid"] = True
    except pa.errors.SchemaErrors as e:
        report["errors"] = e.failure_cases.to_dict("records")

    # Save report
    Path("reports").mkdir(exist_ok=True)
    with open("reports/data_quality.json", "w") as f:
        json.dump(report, f, indent=2)

    if not report["schema_valid"]:
        raise ValueError(f"Data validation failed: {len(report['errors'])} errors")

    print(f"Validation passed: {report['total_rows']} rows")
    return report


if __name__ == "__main__":
    validate_data()
```

### Data Preprocessing

```python
# src/data/preprocess.py
"""Data preprocessing pipeline."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import yaml
from pathlib import Path


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def preprocess_data(
    input_path: str = "data/raw/churn.csv",
    output_dir: str = "data/processed"
):
    """Clean and split data."""
    params = load_params()["preprocess"]

    df = pd.read_csv(input_path)

    # Handle missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode categoricals
    categorical_cols = df.select_dtypes(include=["object"]).columns
    categorical_cols = [c for c in categorical_cols if c != "customerID"]

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scale numericals
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Split data
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y
    )

    # Save processed data
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    # Save preprocessors
    Path("models/preprocessors").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, "models/preprocessors/scaler.pkl")
    joblib.dump(label_encoders, "models/preprocessors/encoders.pkl")

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


if __name__ == "__main__":
    preprocess_data()
```

### Parameters File

```yaml
# params.yaml
preprocess:
  test_size: 0.2
  random_state: 42

features:
  numerical:
    - tenure
    - MonthlyCharges
    - TotalCharges
  categorical:
    - Contract
    - PaymentMethod
    - InternetService

train:
  model_type: xgboost
  hyperparameters:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    min_child_weight: 1
    subsample: 0.8
    colsample_bytree: 0.8
  cross_validation:
    n_splits: 5
    shuffle: true

evaluate:
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc
  threshold: 0.5
```

---

## Phase 4: Feature Engineering

### Feature Building

```python
# src/features/build_features.py
"""Feature engineering pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path


def build_features(input_path: str, output_path: str):
    """Create derived features."""
    df = pd.read_csv(input_path)

    # Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6+yr"]
    )

    # Charge ratio
    df["charge_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Service intensity (count of services)
    service_cols = [c for c in df.columns if "Service" in c]
    df["service_count"] = df[service_cols].sum(axis=1)

    # Contract risk score
    contract_risk = {"Month-to-month": 3, "One year": 2, "Two year": 1}
    df["contract_risk"] = df["Contract"].map(contract_risk)

    # Engagement score
    df["engagement_score"] = (
        df["tenure"] * 0.4 +
        df["service_count"] * 0.3 +
        (3 - df["contract_risk"]) * 0.3
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Features built: {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    build_features("data/processed/train.csv", "data/features/train.csv")
    build_features("data/processed/test.csv", "data/features/test.csv")
```

### Feature Registry

```python
# src/features/registry.py
"""Feature documentation and lineage."""
from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class Feature:
    name: str
    dtype: str
    description: str
    source_columns: List[str]
    transformation: str
    version: str = "1.0"


FEATURE_REGISTRY = {
    "tenure_bucket": Feature(
        name="tenure_bucket",
        dtype="category",
        description="Customer tenure grouped into buckets",
        source_columns=["tenure"],
        transformation="pd.cut with bins [0,12,24,48,72,100]"
    ),
    "charge_ratio": Feature(
        name="charge_ratio",
        dtype="float",
        description="Monthly charges as ratio of total charges",
        source_columns=["MonthlyCharges", "TotalCharges"],
        transformation="MonthlyCharges / (TotalCharges + 1)"
    ),
    "engagement_score": Feature(
        name="engagement_score",
        dtype="float",
        description="Weighted score of customer engagement",
        source_columns=["tenure", "service_count", "contract_risk"],
        transformation="tenure*0.4 + service_count*0.3 + (3-contract_risk)*0.3"
    ),
}


def export_registry(output_path: str = "docs/feature_registry.json"):
    """Export feature registry to JSON."""
    registry_dict = {
        name: {
            "dtype": f.dtype,
            "description": f.description,
            "source_columns": f.source_columns,
            "transformation": f.transformation,
            "version": f.version
        }
        for name, f in FEATURE_REGISTRY.items()
    }

    with open(output_path, "w") as f:
        json.dump(registry_dict, f, indent=2)
```

---

## Phase 5: Experimentation

### Experiment Tracking with MLflow

```python
# src/models/train.py
"""Model training with MLflow tracking."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import yaml
import joblib
from pathlib import Path


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_model(model_type: str, params: dict):
    """Factory for model creation."""
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(**params),
        "xgboost": XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
    }
    return models[model_type]


def train_model(
    train_path: str = "data/features/train.csv",
    experiment_name: str = "churn-prediction"
):
    """Train model with full tracking."""
    params = load_params()["train"]

    # Load data
    df = pd.read_csv(train_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Setup MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params["hyperparameters"])
        mlflow.log_param("model_type", params["model_type"])
        mlflow.log_param("train_samples", len(X))

        # Create and train model
        model = get_model(params["model_type"], params["hyperparameters"])

        # Cross-validation
        cv_config = params["cross_validation"]
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv_config["n_splits"],
            scoring="roc_auc"
        )

        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_scores.std())

        # Final training
        model.fit(X, y)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="churn-classifier"
        )

        # Save locally
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        print(f"Run ID: {run.info.run_id}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return model, run.info.run_id


if __name__ == "__main__":
    train_model()
```

### Experiment Comparison

```python
# src/experiments/compare.py
"""Compare multiple experiments."""
import mlflow
from mlflow.tracking import MlflowClient


def compare_experiments(experiment_name: str, metric: str = "cv_auc_mean"):
    """Get best runs from experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=10
    )

    print(f"\nTop runs by {metric}:")
    print("-" * 60)

    for run in runs:
        model_type = run.data.params.get("model_type", "unknown")
        metric_value = run.data.metrics.get(metric, 0)
        print(f"Run: {run.info.run_id[:8]} | {model_type:15} | {metric}: {metric_value:.4f}")

    return runs


if __name__ == "__main__":
    compare_experiments("churn-prediction")
```

---

## Phase 6: Model Training

### Training Pipeline (DVC Stage)

```yaml
# Add to dvc.yaml
stages:
  # ... previous stages ...

  build_features:
    cmd: python src/features/build_features.py
    deps:
      - src/features/build_features.py
      - data/processed/train.csv
      - data/processed/test.csv
    outs:
      - data/features/train.csv
      - data/features/test.csv

  train:
    cmd: python src/models/train.py
    deps:
      - src/models/train.py
      - data/features/train.csv
    params:
      - train.model_type
      - train.hyperparameters
    outs:
      - models/model.pkl
    metrics:
      - reports/training_metrics.json:
          cache: false
```

### Hyperparameter Tuning

```python
# src/models/tune.py
"""Hyperparameter tuning with MLflow tracking."""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import mlflow
from scipy.stats import uniform, randint


def tune_hyperparameters(
    train_path: str = "data/features/train.csv",
    n_iter: int = 20
):
    """Random search with MLflow logging."""
    df = pd.read_csv(train_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    param_distributions = {
        "n_estimators": randint(50, 300),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    }

    mlflow.set_experiment("churn-tuning")

    with mlflow.start_run(run_name="hyperparameter_tuning"):
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )

        search.fit(X, y)

        # Log best parameters
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", search.best_score_)

        # Log all trials
        for i, (params, score) in enumerate(zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"]
        )):
            mlflow.log_metric(f"trial_{i}_score", score)

        print(f"Best score: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")

        return search.best_params_


if __name__ == "__main__":
    tune_hyperparameters()
```

---

## Phase 7: Evaluation

### Comprehensive Evaluation

```python
# src/models/evaluate.py
"""Model evaluation with detailed metrics."""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import json
import joblib
from pathlib import Path


def evaluate_model(
    model_path: str = "models/model.pkl",
    test_path: str = "data/features/test.csv",
    threshold: float = 0.5
):
    """Full evaluation with business metrics."""
    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    X = df.drop(columns=["Churn"])
    y_true = df["Churn"]

    # Predictions
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }

    # Business metrics
    cost_fn = 500  # Cost of missing a churner
    cost_fp = 50   # Cost of unnecessary retention offer

    total_cost = (cm[1, 0] * cost_fn) + (cm[0, 1] * cost_fp)
    churners_caught = cm[1, 1]

    metrics["business"] = {
        "total_cost": total_cost,
        "churners_caught": int(churners_caught),
        "churners_missed": int(cm[1, 0]),
        "potential_savings": churners_caught * cost_fn
    }

    # Save metrics
    Path("reports").mkdir(exist_ok=True)
    with open("reports/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    plot_evaluation(y_true, y_prob, y_pred)

    print(f"\nEvaluation Results (threshold={threshold}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nBusiness Impact:")
    print(f"  Churners caught: {churners_caught}")
    print(f"  Potential savings: ${metrics['business']['potential_savings']}")

    return metrics


def plot_evaluation(y_true, y_prob, y_pred):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    axes[2].imshow(cm, cmap="Blues")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["No Churn", "Churn"])
    axes[2].set_yticklabels(["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, cm[i, j], ha="center", va="center")
    axes[2].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("reports/evaluation_plots.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    evaluate_model()
```

### Quality Gates

```python
# src/models/gates.py
"""Quality gates for model promotion."""
import json
from pathlib import Path


def check_quality_gates(metrics_path: str = "reports/evaluation_metrics.json"):
    """Verify model meets minimum requirements."""
    with open(metrics_path) as f:
        metrics = json.load(f)

    gates = {
        "recall": {"min": 0.80, "actual": metrics["recall"]},
        "precision": {"min": 0.60, "actual": metrics["precision"]},
        "roc_auc": {"min": 0.85, "actual": metrics["roc_auc"]},
    }

    passed = True
    print("\nQuality Gates:")
    print("-" * 50)

    for metric, values in gates.items():
        status = "PASS" if values["actual"] >= values["min"] else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  {metric}: {values['actual']:.4f} (min: {values['min']}) [{status}]")

    print("-" * 50)
    print(f"Overall: {'PASSED' if passed else 'FAILED'}")

    return passed


if __name__ == "__main__":
    import sys
    if not check_quality_gates():
        sys.exit(1)
```

---

## Phase 8: Model Registry

### MLflow Model Registry

```python
# src/models/registry.py
"""Model registry operations."""
import mlflow
from mlflow.tracking import MlflowClient


def register_model(run_id: str, model_name: str = "churn-classifier"):
    """Register a trained model."""
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)

    print(f"Registered model version: {result.version}")
    return result


def transition_model(
    model_name: str,
    version: int,
    stage: str  # "Staging" or "Production"
):
    """Move model between stages."""
    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )

    print(f"Model {model_name} v{version} → {stage}")


def get_production_model(model_name: str = "churn-classifier"):
    """Load the current production model."""
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def list_model_versions(model_name: str = "churn-classifier"):
    """List all versions of a model."""
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")

    print(f"\nModel: {model_name}")
    print("-" * 50)
    for v in versions:
        print(f"  v{v.version}: {v.current_stage} (run: {v.run_id[:8]})")

    return versions


if __name__ == "__main__":
    list_model_versions()
```

### Model Card

```python
# src/models/model_card.py
"""Generate model documentation."""
import json
from datetime import datetime
from pathlib import Path


def generate_model_card(
    metrics_path: str = "reports/evaluation_metrics.json",
    params_path: str = "params.yaml",
    output_path: str = "docs/MODEL_CARD.md"
):
    """Create model card documentation."""
    import yaml

    with open(metrics_path) as f:
        metrics = json.load(f)

    with open(params_path) as f:
        params = yaml.safe_load(f)

    card = f"""# Model Card: Customer Churn Classifier

## Model Details

- **Name:** churn-classifier
- **Version:** 1.0.0
- **Type:** {params['train']['model_type']}
- **Task:** Binary Classification
- **Created:** {datetime.now().strftime('%Y-%m-%d')}

## Intended Use

- **Primary Use:** Predict customer churn for retention campaigns
- **Users:** Marketing team, Customer Success
- **Out of Scope:** Real-time fraud detection, credit scoring

## Training Data

- **Source:** Internal customer database
- **Size:** ~7,000 customers
- **Features:** 20 (demographics, services, account info)
- **Label Distribution:** ~27% churn rate

## Evaluation Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Accuracy | {metrics['accuracy']:.4f} | - |
| Precision | {metrics['precision']:.4f} | >0.60 |
| Recall | {metrics['recall']:.4f} | >0.80 |
| F1 Score | {metrics['f1']:.4f} | - |
| ROC AUC | {metrics['roc_auc']:.4f} | >0.85 |

## Limitations

- Trained on historical data; may not reflect future trends
- Performance may degrade if customer behavior changes
- Requires retraining if new products are introduced

## Ethical Considerations

- Model should not be used for discriminatory purposes
- Predictions should be reviewed before taking action
- Customer privacy must be maintained

## Monitoring

- Weekly performance monitoring
- Drift detection on feature distributions
- Quarterly retraining evaluation
"""

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write(card)

    print(f"Model card generated: {output_path}")


if __name__ == "__main__":
    generate_model_card()
```

---

## Phase 9: Deployment

### FastAPI Serving

```python
# src/api/main.py
"""Model serving API."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/preprocessors/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str  # "Month-to-month", "One year", "Two year"
    payment_method: str
    internet_service: str


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str


class BatchRequest(BaseModel):
    customers: List[CustomerData]


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Single prediction endpoint."""
    try:
        # Prepare features
        features = prepare_features(customer)

        # Predict
        prob = model.predict_proba(features)[0, 1]
        pred = prob >= 0.5

        # Risk level
        if prob >= 0.7:
            risk = "HIGH"
        elif prob >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=pred,
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """Batch prediction endpoint."""
    results = []
    for customer in request.customers:
        result = predict(customer)
        results.append(result)
    return {"predictions": results}


def prepare_features(customer: CustomerData) -> pd.DataFrame:
    """Transform input to model features."""
    data = {
        "tenure": [customer.tenure],
        "MonthlyCharges": [customer.monthly_charges],
        "TotalCharges": [customer.total_charges],
    }

    df = pd.DataFrame(data)
    df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    # Add encoded categoricals (simplified)
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["Contract"] = contract_map.get(customer.contract, 0)

    return df


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-api
  labels:
    app: churn-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: churn-api
  template:
    metadata:
      labels:
        app: churn-api
    spec:
      containers:
      - name: api
        image: registry.local/churn-api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/model.pkl"
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: churn-api
spec:
  selector:
    app: churn-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: churn-api
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: churn-api.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: churn-api
            port:
              number: 80
```

---

## Phase 10: Monitoring

### Drift Detection

```python
# src/monitoring/drift.py
"""Data and model drift detection."""
import pandas as pd
import numpy as np
from scipy import stats
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import json
from datetime import datetime
from pathlib import Path


def detect_data_drift(
    reference_path: str,
    current_path: str,
    output_path: str = "reports/drift_report.html"
):
    """Detect data drift using Evidently."""
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    column_mapping = ColumnMapping(
        target="Churn",
        numerical_features=["tenure", "MonthlyCharges", "TotalCharges"],
        categorical_features=["Contract", "PaymentMethod"]
    )

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    Path(output_path).parent.mkdir(exist_ok=True)
    report.save_html(output_path)

    # Extract drift status
    drift_results = report.as_dict()

    print(f"Drift report saved: {output_path}")
    return drift_results


def statistical_drift_test(
    reference: pd.Series,
    current: pd.Series,
    threshold: float = 0.05
) -> dict:
    """KS test for numerical drift."""
    statistic, p_value = stats.ks_2samp(reference, current)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < threshold
    }


def monitor_predictions(
    predictions: pd.DataFrame,
    window_days: int = 7
):
    """Monitor prediction distribution over time."""
    predictions["date"] = pd.to_datetime(predictions["timestamp"]).dt.date

    daily_stats = predictions.groupby("date").agg({
        "churn_probability": ["mean", "std", "count"],
        "churn_prediction": "mean"
    }).round(4)

    # Alert if prediction rate changes significantly
    recent_churn_rate = daily_stats["churn_prediction"]["mean"].tail(window_days).mean()
    baseline_churn_rate = 0.27  # Expected from training

    alert = abs(recent_churn_rate - baseline_churn_rate) > 0.05

    return {
        "daily_stats": daily_stats.to_dict(),
        "recent_churn_rate": recent_churn_rate,
        "baseline_churn_rate": baseline_churn_rate,
        "alert": alert
    }
```

### Prometheus Metrics

```python
# src/api/metrics.py
"""Prometheus metrics for monitoring."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time

# Request metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

# Model metrics
PREDICTION_SCORE = Histogram(
    "prediction_score",
    "Distribution of prediction scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

CHURN_PREDICTIONS = Counter(
    "churn_predictions_total",
    "Total predictions by class",
    ["prediction"]
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Current model version",
    ["version"]
)


def track_prediction(prob: float, pred: bool):
    """Track prediction metrics."""
    PREDICTION_SCORE.observe(prob)
    CHURN_PREDICTIONS.labels(prediction="churn" if pred else "no_churn").inc()


def track_request(endpoint: str):
    """Decorator to track request metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
                raise
            finally:
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(
                    time.time() - start_time
                )
        return wrapper
    return decorator
```

---

## Phase 11: CI/CD Pipeline

### GitHub Actions: CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install ruff black

      - name: Lint
        run: |
          ruff check src/
          black --check src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml
```

### GitHub Actions: Training Pipeline

```yaml
# .github/workflows/train.yml
name: Training Pipeline

on:
  workflow_dispatch:
    inputs:
      experiment_name:
        description: "MLflow experiment name"
        default: "churn-production"
  schedule:
    - cron: "0 0 * * 0"  # Weekly

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Pull data with DVC
        run: |
          dvc pull
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Run training pipeline
        run: |
          dvc repro
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

      - name: Check quality gates
        run: python src/models/gates.py

      - name: Push artifacts
        run: |
          dvc push
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

### GitHub Actions: Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy Model

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: "Model version to deploy"
        required: true
      environment:
        description: "Target environment"
        type: choice
        options:
          - staging
          - production

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ secrets.REGISTRY_URL }}
          username: ${{ secrets.REGISTRY_USER }}
          password: ${{ secrets.REGISTRY_PASSWORD }}

      - name: Download model from registry
        run: |
          python -c "
          import mlflow
          mlflow.set_tracking_uri('${{ secrets.MLFLOW_TRACKING_URI }}')
          model = mlflow.sklearn.load_model('models:/churn-classifier/${{ github.event.inputs.model_version }}')
          import joblib
          joblib.dump(model, 'models/model.pkl')
          "

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.REGISTRY_URL }}/churn-api:${{ github.event.inputs.model_version }}
            ${{ secrets.REGISTRY_URL }}/churn-api:latest

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/churn-api \
            api=${{ secrets.REGISTRY_URL }}/churn-api:${{ github.event.inputs.model_version }}
        env:
          KUBECONFIG: ${{ secrets.KUBECONFIG }}
```

---

## Quick Reference

### DVC Commands

```bash
# Initialize
dvc init
dvc remote add -d storage s3://bucket/path

# Track data
dvc add data/raw/churn.csv
git add data/raw/churn.csv.dvc .gitignore
git commit -m "Track raw data"

# Push/Pull data
dvc push
dvc pull

# Run pipeline
dvc repro                    # Run full pipeline
dvc repro train              # Run specific stage
dvc dag                      # View pipeline DAG

# Compare experiments
dvc params diff              # Parameter changes
dvc metrics diff             # Metric changes
```

### MLflow Commands

```bash
# Start server
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

# UI
mlflow ui --port 5000

# CLI
mlflow experiments list
mlflow runs list --experiment-id 1

# Model registry
mlflow models serve -m "models:/churn-classifier/Production" -p 5001
```

### Pipeline Execution

```bash
# Full pipeline
dvc repro

# Individual stages
python src/data/load.py
python src/data/validate.py
python src/data/preprocess.py
python src/features/build_features.py
python src/models/train.py
python src/models/evaluate.py

# Quality gates
python src/models/gates.py

# Start API
uvicorn src.api.main:app --reload --port 8000
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific tests
pytest tests/test_data.py -v
pytest tests/test_model.py::test_prediction -v
```

---

## Next Steps

1. **Initialize Project:** Run `scripts/setup.sh`
2. **Get Data:** Place raw data in `data/raw/`
3. **Run Pipeline:** Execute `dvc repro`
4. **Track Experiments:** View MLflow UI at `localhost:5000`
5. **Deploy:** Build Docker image and deploy to K8s
6. **Monitor:** Set up drift detection and alerting

---

**Related Docs:**
- [AI/ML Workbench Setup](./04-aiml-workbench.md)
- [K8s Platform](../k8s-platform/README.md)
