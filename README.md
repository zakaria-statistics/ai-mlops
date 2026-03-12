# Real Estate Price Prediction — End-to-End ML Project

> Predict King County house prices using XGBoost — from EDA to deployed API.

## Table of Contents

1. [Overview](#overview)
2. [Results](#results)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Pipeline Phases](#pipeline-phases)
6. [Tech Stack](#tech-stack)
7. [API Usage](#api-usage)

---

## Overview

End-to-end machine learning project built iteratively — each phase teaches specific ML/MLOps concepts through a real prediction task.

- **Problem:** Predict residential property sale prices for buyers and sellers
- **Dataset:** King County House Sales (~21K rows, 16 engineered features)
- **Best Model:** XGBoost (R² = 0.91, MAE = $64K, MAPE = 12%)

## Results

| Model | RMSE (log) | R² | MAE ($) | MAPE (%) |
|-------|-----------|-----|---------|----------|
| **XGBoost** | **0.162** | **0.91** | **$64,514** | **11.7%** |
| Gradient Boosting (HGB) | 0.164 | 0.91 | $66,282 | 12.0% |
| Random Forest | 0.185 | 0.88 | $79,005 | 13.5% |
| Decision Tree (pruned) | 0.218 | 0.83 | $94,058 | 16.2% |
| Linear Regression | 0.257 | 0.77 | $118,071 | 20.3% |

**Top features:** grade (46%), lat (15%), sqft_living (13%)

## Quick Start

### Option 1 — CLI Prediction

```bash
cd ai_mlops
pip install -r models/v1/requirements.txt
python models/v1/predict.py
python models/v1/predict.py --csv models/v1/sample_input.csv
```

### Option 2 — API Server

```bash
pip install -r models/v1/requirements.txt
pip install fastapi uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000
# Visit http://localhost:8000/docs for Swagger UI
```

### Option 3 — Docker

```bash
docker build -t house-price-api:v1 .
docker run -d -p 8000:8000 house-price-api:v1
curl http://localhost:8000/health
```

## Project Structure

```
ai_mlops/
├── README.md                   ← you are here
├── roadmap.md                  ← learning path and phase map
│
├── notebooks/                  ← jupytext notebooks (run with # %% cells)
│   ├── 01-eda-house-prices.py  ← exploratory data analysis
│   ├── 02-data-preparation.py  ← cleaning, transforms, train/test split
│   ├── 03-modeling.py          ← model progression (Linear → XGBoost)
│   ├── 04-evaluation.py        ← cross-validation, tuning, error analysis
│   └── 05-packaging.py         ← serialize model to artifact
│
├── data/
│   ├── raw/                    ← original dataset (never modified)
│   └── processed/              ← X_train, X_test, y_train, y_test
│
├── models/v1/                  ← model artifact
│   ├── model.joblib            ← serialized XGBoost model
│   ├── model_meta.json         ← feature schema + metrics
│   ├── requirements.txt        ← pinned dependencies
│   ├── predict.py              ← standalone prediction script
│   └── sample_input.csv        ← test data
│
├── src/
│   └── api.py                  ← FastAPI prediction API
│
├── Dockerfile                  ← containerized deployment
├── .dockerignore
│
├── 01-business-problem/        ← phase guides (README.md each)
├── 02-data-collection/
├── 03-data-exploration/
├── 04-data-preparation/
├── 05-feature-engineering/
├── 06-modeling/
├── 07-evaluation/
├── 08-packaging/
├── 09-deployment/
├── 10-monitoring/
│
├── docs/
│   └── ops-approaches.md       ← Native → MLflow → OpenFaaS migration path
│
├── maths-behind/               ← mathematical formulas reference
│   ├── 01-algorithms.md        ← all model formulas
│   ├── 02-metrics.md           ← RMSE, MAE, R², MAPE
│   └── 03-techniques.md        ← regularization, bagging, boosting, etc.
│
└── reports/figures/             ← EDA visualization outputs
```

## Pipeline Phases

```
UNDERSTAND                  BUILD                       VALIDATE & SHIP
───────────                 ─────                       ───────────────
01 Business Problem    →    04 Data Preparation    →    07 Evaluation
02 Data Collection     →    05 Feature Engineering →    08 Packaging
03 Data Exploration    →    06 Modeling            →    09 Deployment
                                                        10 Monitoring
```

| Phase | What | Key Output |
|-------|------|------------|
| 01 | Problem framing, success metrics | Problem statement |
| 02 | Dataset sourcing | Raw data in `data/raw/` |
| 03 | EDA, distributions, correlations | Exploration notebook + figures |
| 04 | Cleaning, encoding, train/test split | Processed data (16 features) |
| 05 | house_age, has_renovation features | Combined with Phase 04 |
| 06 | 6-level model progression | XGBoost best (R²=0.91) |
| 07 | Cross-validation, hyperparameter tuning | Confirmed XGBoost winner |
| 08 | Serialize model, pin dependencies | `models/v1/` artifact |
| 09 | FastAPI + Docker | HTTP API on port 8000 |
| 10 | Drift detection, metrics | *(planned)* |

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| API | FastAPI, uvicorn, pydantic |
| Container | Docker |
| Notebooks | Jupytext (`.py` with `# %%` cells) |

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Single Prediction

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
  "predicted_price": 546314.44,
  "predicted_price_formatted": "$546,314",
  "model": "XGBoost",
  "model_r2": 0.9082
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"bedrooms":3,...}, {"bedrooms":2,...}]'
```

---

*Built as a learning project — see [roadmap.md](roadmap.md) for the full learning path.*
