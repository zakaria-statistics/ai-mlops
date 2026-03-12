"""
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
        print(f"\nPredictions for {len(df)} rows:\n")
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
        print(f"\nExample prediction: ${price:,.0f}")
        print("(3br/2.25ba, 2080 sqft, grade 7, Seattle area)")


if __name__ == "__main__":
    main()
