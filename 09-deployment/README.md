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

## Level 2 — Docker Container

### Build & Run

```bash
cd /root/claude/ai_mlops
docker build -t house-price-api:v1 .
docker run -d --name house-price-api -p 8000:8000 house-price-api:v1
```

### What's Inside

```
Dockerfile
  ├── FROM python:3.11-slim          ← minimal base image
  ├── COPY requirements.txt          ← pinned deps from model artifact
  ├── RUN pip install                ← deps + fastapi + uvicorn
  ├── COPY models/v1/               ← model artifact baked in
  ├── COPY src/api.py               ← API code
  └── CMD uvicorn                   ← starts server on :8000
```

### Architecture

```
Docker Container
┌──────────────────────────────────┐
│  python:3.11-slim                │
│  ├── FastAPI (src/api.py)        │
│  ├── model.joblib                │
│  ├── model_meta.json             │
│  └── uvicorn :8000               │
└──────────────────────────────────┘
  ↕ port 8000
Host / K8s / any Docker runtime
```

### Useful Commands

```bash
# Check logs
docker logs house-price-api

# Test
curl http://localhost:8000/health

# Stop & remove
docker stop house-price-api && docker rm house-price-api

# New model version → rebuild
docker build -t house-price-api:v2 .
```

### Why Docker

| Without Docker | With Docker |
|----------------|-------------|
| "Works on my machine" | Works everywhere |
| Manual pip install | Deps baked into image |
| Python version mismatch risk | Exact Python version |
| Model file path issues | Fixed paths inside container |

---

## Next Levels

| Level | Approach | What It Adds |
|-------|----------|-------------|
| 3 | OpenFaaS on K8s | Scale-to-zero, auto-scaling, Prometheus metrics |
| 4 | MLflow serving | Model registry integration, stage promotion |

See [docs/ops-approaches.md](../docs/ops-approaches.md) for full comparison.
