"""
House Price Prediction API — Native Deployment (Level 1)

Run:
    cd /root/claude/ai_mlops
    source .venv/bin/activate
    pip install fastapi uvicorn
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Test:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"bedrooms":3,"bathrooms":2.25,"sqft_living":2080,"sqft_lot":7500,
           "floors":1.0,"waterfront":0,"view":0,"condition":3,"grade":7,
           "sqft_basement":0,"lat":47.56,"long":-122.21,"sqft_living15":1800,
           "sqft_lot15":7500,"house_age":25,"has_renovation":0}'
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Load model at startup (once) ---
MODEL_DIR = Path(__file__).parent.parent / "models" / "v1"

model = joblib.load(MODEL_DIR / "model.joblib")
with open(MODEL_DIR / "model_meta.json") as f:
    meta = json.load(f)

FEATURES = meta["features"]

# --- Request/Response schemas ---

class HouseFeatures(BaseModel):
    """Input: house features for price prediction."""
    bedrooms: int = Field(..., ge=0, le=15, example=3)
    bathrooms: float = Field(..., ge=0, le=10, example=2.25)
    sqft_living: int = Field(..., ge=200, le=15000, example=2080)
    sqft_lot: int = Field(..., ge=0, example=7500)
    floors: float = Field(..., ge=1, le=4, example=1.0)
    waterfront: int = Field(..., ge=0, le=1, example=0)
    view: int = Field(..., ge=0, le=4, example=0)
    condition: int = Field(..., ge=1, le=5, example=3)
    grade: int = Field(..., ge=1, le=13, example=7)
    sqft_basement: int = Field(..., ge=0, example=0)
    lat: float = Field(..., ge=47.0, le=48.0, example=47.56)
    long: float = Field(..., ge=-123.0, le=-121.0, example=-122.21)
    sqft_living15: int = Field(..., ge=0, example=1800)
    sqft_lot15: int = Field(..., ge=0, example=7500)
    house_age: int = Field(..., ge=0, le=200, example=25)
    has_renovation: int = Field(..., ge=0, le=1, example=0)


class PredictionResponse(BaseModel):
    """Output: predicted price and model info."""
    predicted_price: float
    predicted_price_formatted: str
    model: str
    model_r2: float


class HealthResponse(BaseModel):
    status: str
    model: str
    features: int


# --- API ---

app = FastAPI(
    title="House Price Prediction API",
    description="Predict King County house prices using XGBoost.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Check if the API and model are loaded."""
    return HealthResponse(
        status="ok",
        model=meta["model"],
        features=meta["n_features"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    """Predict house price from features."""
    df = pd.DataFrame([house.model_dump()])
    X = df[FEATURES]

    log_pred = model.predict(X)[0]
    price = float(np.expm1(log_pred))

    return PredictionResponse(
        predicted_price=round(price, 2),
        predicted_price_formatted=f"${price:,.0f}",
        model=meta["model"],
        model_r2=meta["metrics"]["r2"],
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(houses: list[HouseFeatures]):
    """Predict prices for multiple houses."""
    if len(houses) > 100:
        raise HTTPException(status_code=400, detail="Max 100 houses per batch")

    df = pd.DataFrame([h.model_dump() for h in houses])
    X = df[FEATURES]

    log_preds = model.predict(X)
    prices = np.expm1(log_preds)

    return [
        PredictionResponse(
            predicted_price=round(float(p), 2),
            predicted_price_formatted=f"${p:,.0f}",
            model=meta["model"],
            model_r2=meta["metrics"]["r2"],
        )
        for p in prices
    ]
