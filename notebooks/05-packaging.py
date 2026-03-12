# %% [markdown]
# # Packaging — Serialize Model for Deployment
# > Save the best model as a reusable artifact with pinned dependencies.
#
# **What this does:**
#
# ```
# Training (notebooks)          Artifact (portable)
# ───────────────────           ──────────────────
# X_train, y_train              model.joblib        ← trained model
# model.fit()                   feature_names.json  ← expected input schema
# model.predict()               requirements.txt    ← pinned dependencies
#                                predict.py          ← standalone prediction script
# ```
#
# **Why serialize?**
# - Don't retrain every time you need a prediction
# - Share the model without sharing training code/data
# - Version control: model v1, v2, v3...
# - Deploy anywhere Python runs

# %% [markdown]
# ## 1. Setup — Retrain Best Model

# %%
import pandas as pd
import numpy as np
import json
from pathlib import Path

X_train = pd.read_csv('../data/processed/X_train.csv')
X_test = pd.read_csv('../data/processed/X_test.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('../data/processed/y_test.csv').squeeze()

print(f"Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
print(f"Features: {list(X_train.columns)}")

# %%
# Train the winning model (XGBoost with Phase 6 defaults)
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Verify it matches Phase 7 results
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
print(f"\nVerification — RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: ${mae:,.0f}")

# %% [markdown]
# ## 2. Save Model Artifact
#
# **joblib vs pickle:**
# ```
# pickle:  Python standard, works for any object
# joblib:  Optimized for large numpy arrays (faster for sklearn/xgboost models)
# ```

# %%
import joblib

artifact_dir = Path('../models/v1')
artifact_dir.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(model, artifact_dir / 'model.joblib')
print(f"Model saved: {artifact_dir / 'model.joblib'}")
print(f"File size: {(artifact_dir / 'model.joblib').stat().st_size / 1024:.0f} KB")

# %% [markdown]
# ## 3. Save Feature Schema
# Anyone loading this model needs to know: what features, in what order.

# %%
schema = {
    'features': list(X_train.columns),
    'n_features': len(X_train.columns),
    'target': 'log_price (log1p transformed)',
    'inverse_transform': 'np.expm1(prediction) to get dollars',
    'model': 'XGBoost',
    'metrics': {
        'rmse_log': round(rmse, 4),
        'r2': round(r2, 4),
        'mae_dollars': round(mae, 0),
    },
    'training_rows': X_train.shape[0],
    'random_state': 42,
}

with open(artifact_dir / 'model_meta.json', 'w') as f:
    json.dump(schema, f, indent=2)

print(json.dumps(schema, indent=2))

# %% [markdown]
# ## 4. Pin Dependencies
# Exact versions so the model loads identically on any machine.

# %%
import sklearn
import xgboost

deps = {
    'xgboost': xgboost.__version__,
    'scikit-learn': sklearn.__version__,
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'joblib': joblib.__version__,
}

req_lines = [f"{pkg}=={ver}" for pkg, ver in deps.items()]
with open(artifact_dir / 'requirements.txt', 'w') as f:
    f.write('\n'.join(req_lines) + '\n')

print("requirements.txt:")
for line in req_lines:
    print(f"  {line}")

# %% [markdown]
# ## 5. Create Standalone Prediction Script
# A single file that loads the model and makes predictions — no notebooks needed.

# %%
predict_script = '''"""
Standalone prediction script — King County House Prices.

Usage:
    python predict.py                          # run example prediction
    python predict.py --csv input.csv          # predict from CSV file

Model: XGBoost (Phase 6 default)
Target: log1p(price) → expm1 to get dollars
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent


def load_model():
    model = joblib.load(MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "model_meta.json") as f:
        meta = json.load(f)
    return model, meta


def predict(df, model, meta):
    """Predict prices in dollars."""
    # Validate features
    expected = meta["features"]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[expected]
    log_pred = model.predict(X)
    return np.expm1(log_pred)


def main():
    parser = argparse.ArgumentParser(description="Predict house prices")
    parser.add_argument("--csv", help="Input CSV file path")
    args = parser.parse_args()

    model, meta = load_model()
    print(f"Model loaded: {meta['model']} ({meta['n_features']} features)")
    print(f"Training metrics — R²: {meta['metrics']['r2']}, MAE: ${meta['metrics']['mae_dollars']:,.0f}")

    if args.csv:
        df = pd.read_csv(args.csv)
        prices = predict(df, model, meta)
        df["predicted_price"] = prices
        df["predicted_price"] = df["predicted_price"].map("${:,.0f}".format)
        print(f"\\nPredictions for {len(df)} rows:\\n")
        print(df.to_string(index=False))
    else:
        # Example: median King County home
        example = pd.DataFrame([{
            "bedrooms": 3, "bathrooms": 2.25, "sqft_living": 2080,
            "sqft_lot": 7500, "floors": 1.0, "waterfront": 0,
            "view": 0, "condition": 3, "grade": 7,
            "sqft_basement": 0, "lat": 47.56, "long": -122.21,
            "sqft_living15": 1800, "sqft_lot15": 7500,
            "house_age": 25, "has_renovation": 0,
        }])
        price = predict(example, model, meta)[0]
        print(f"\\nExample prediction: ${price:,.0f}")
        print("(3br/2.25ba, 2080 sqft, grade 7, Seattle area)")


if __name__ == "__main__":
    main()
'''

with open(artifact_dir / 'predict.py', 'w') as f:
    f.write(predict_script)

print(f"Saved: {artifact_dir / 'predict.py'}")

# %% [markdown]
# ## 6. Verify — Load and Predict from Artifact
# Simulate what a consumer of this model would do.

# %%
# Fresh load (as if on a different machine)
loaded_model = joblib.load(artifact_dir / 'model.joblib')
with open(artifact_dir / 'model_meta.json') as f:
    meta = json.load(f)

# Predict on test set
y_pred_loaded = loaded_model.predict(X_test[meta['features']])
rmse_loaded = np.sqrt(mean_squared_error(y_test, y_pred_loaded))

print(f"Original RMSE:     {rmse:.4f}")
print(f"Loaded model RMSE: {rmse_loaded:.4f}")
print(f"Match: {np.allclose(y_pred, y_pred_loaded)}")

# %% [markdown]
# ## 7. Artifact Summary

# %%
print(f"\nArtifact directory: {artifact_dir.resolve()}")
print(f"{'─' * 50}")
for f in sorted(artifact_dir.iterdir()):
    size = f.stat().st_size
    unit = 'KB' if size > 1024 else 'B'
    val = size / 1024 if size > 1024 else size
    print(f"  {f.name:<25s} {val:>6.0f} {unit}")

# %% [markdown]
# ## Summary
#
# | Artifact | Purpose |
# |----------|---------|
# | `model.joblib` | Serialized XGBoost model |
# | `model_meta.json` | Feature names, metrics, schema |
# | `requirements.txt` | Pinned Python dependencies |
# | `predict.py` | Standalone prediction script |
#
# ```
# models/v1/
# ├── model.joblib          ← the trained model
# ├── model_meta.json       ← what it expects and how it performed
# ├── requirements.txt      ← exact dependency versions
# └── predict.py            ← load + predict in one script
# ```
#
# **To use on another machine:**
# ```bash
# pip install -r models/v1/requirements.txt
# python models/v1/predict.py
# python models/v1/predict.py --csv new_houses.csv
# ```
#
# -> Next: [09-deployment](../09-deployment/) to serve via API / container.
# -> Future: MLflow for experiment tracking, model registry, and versioning.
