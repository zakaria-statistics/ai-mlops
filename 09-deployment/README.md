# Phase 9: Deployment

> Serve the model via HTTP API — from native Python to production K8s.

## Table of Contents

1. [Goal](#goal)
2. [Level 1 — Native (FastAPI)](#level-1--native-fastapi)
3. [Endpoints](#endpoints)
4. [Next Levels](#next-levels)

---

## Goal

Make predictions available over HTTP so any client (web app, script, another service) can get a price prediction without loading the model themselves.

## Level 1 — Native (FastAPI)

### Install & Run

```bash
cd /root/claude/ai_mlops
source .venv/bin/activate
pip install fastapi uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Libraries

| Library | Purpose | Required |
|---------|---------|----------|
| fastapi | HTTP framework with auto-validation | Yes |
| uvicorn | ASGI server to run FastAPI | Yes |
| pydantic | Request/response schema validation (bundled with FastAPI) | Yes |
| joblib | Load serialized model | Yes |
| xgboost | Model framework | Yes |
| pandas / numpy | Data handling | Yes |

### Architecture

```
Client (curl, browser, app)
  ↓ HTTP POST /predict
FastAPI (src/api.py)
  ↓ loads at startup
model.joblib + model_meta.json (models/v1/)
  ↓ returns
JSON { predicted_price, model, r2 }
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check API and model status |
| POST | `/predict` | Predict single house price |
| POST | `/predict/batch` | Predict up to 100 houses |
| GET | `/docs` | Auto-generated Swagger UI |

### Example — Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3, "bathrooms": 2.25, "sqft_living": 2080,
    "sqft_lot": 7500, "floors": 1.0, "waterfront": 0,
    "view": 0, "condition": 3, "grade": 7,
    "sqft_basement": 0, "lat": 47.56, "long": -122.21,
    "sqft_living15": 1800, "sqft_lot15": 7500,
    "house_age": 25, "has_renovation": 0
  }'
```

Response:
```json
{
  "predicted_price": 546314.25,
  "predicted_price_formatted": "$546,314",
  "model": "XGBoost",
  "model_r2": 0.9082
}
```

### Example — Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"bedrooms":3,"bathrooms":2.25,"sqft_living":2080,"sqft_lot":7500,"floors":1.0,"waterfront":0,"view":0,"condition":3,"grade":7,"sqft_basement":0,"lat":47.56,"long":-122.21,"sqft_living15":1800,"sqft_lot15":7500,"house_age":25,"has_renovation":0},
    {"bedrooms":2,"bathrooms":1.0,"sqft_living":850,"sqft_lot":5000,"floors":1.0,"waterfront":0,"view":0,"condition":3,"grade":6,"sqft_basement":0,"lat":47.48,"long":-122.34,"sqft_living15":1100,"sqft_lot15":6000,"house_age":70,"has_renovation":0}
  ]'
```

### Interactive Docs

Once running, visit `http://localhost:8000/docs` for Swagger UI — test endpoints directly in the browser.

---

## Next Levels

| Level | Approach | What It Adds |
|-------|----------|-------------|
| 2 | Docker container | Portable, reproducible environment |
| 3 | OpenFaaS on K8s | Scale-to-zero, auto-scaling, Prometheus metrics |
| 4 | MLflow serving | Model registry integration, stage promotion |

See [docs/ops-approaches.md](../docs/ops-approaches.md) for full comparison.
