# MLOps Approaches — Native → Managed → Production

> Three levels of operational maturity for serving our model, each building on the last.

## Table of Contents

1. [Overview](#overview)
2. [Level 1 — Native (Current)](#level-1--native-current)
3. [Level 2 — MLflow](#level-2--mlflow)
4. [Level 3 — OpenFaaS on K8s](#level-3--openfaas-on-k8s)
5. [Comparison](#comparison)
6. [Migration Path](#migration-path)

---

## Overview

```
Level 1: Native               Level 2: MLflow              Level 3: OpenFaaS + MLflow
─────────────────              ──────────────               ──────────────────────────
joblib + script                Experiment tracking          K8s-native serving
Manual versioning (v1/)        Model registry (versions)    Scale-to-zero
requirements.txt               Artifact store               Prometheus metrics
predict.py                     mlflow models serve          Grafana dashboards
                               Stage promotion              Auto-scaling
```

## Level 1 — Native (Current)

**What:** Serialize model with joblib, pin deps, standalone predict script.

### Components

```
models/v1/
├── model.joblib          ← serialized model
├── model_meta.json       ← feature schema + metrics
├── requirements.txt      ← pinned versions
└── predict.py            ← CLI prediction script
```

### Serves

```bash
python models/v1/predict.py --csv input.csv
```

### Pros
- Zero infrastructure needed
- Easy to understand — just files
- Portable — copy the folder anywhere

### Cons
- No experiment tracking (which params gave which results?)
- Manual versioning (v1, v2, v3 dirs)
- No HTTP serving (batch only, no real-time)
- No monitoring

### You Learn
- Model serialization (joblib vs pickle)
- Dependency management
- Schema contracts (what features the model expects)

---

## Level 2 — MLflow

**What:** Experiment tracking, model registry, and optional model serving.

### Components

```
MLflow Server (can run locally or on K8s)
├── Experiment Tracking     ← log every training run
│   ├── parameters          ← n_estimators=500, lr=0.05...
│   ├── metrics             ← RMSE, R², MAE per run
│   └── artifacts           ← model files, plots
├── Model Registry          ← version + stage management
│   ├── v1 (Production)     ← currently serving
│   ├── v2 (Staging)        ← being validated
│   └── v3 (None)           ← just trained
└── Model Serving           ← HTTP endpoint
    └── mlflow models serve ← REST API
```

### Workflow

```
1. Train model
   ↓
2. mlflow.log_params({"n_estimators": 500, ...})
   mlflow.log_metrics({"rmse": 0.162, "r2": 0.908})
   mlflow.xgboost.log_model(model, "model")
   ↓
3. Compare runs in MLflow UI (http://localhost:5000)
   ↓
4. Register best model: mlflow.register_model("xgboost-house-price")
   ↓
5. Promote: None → Staging → Production
   ↓
6. Serve: mlflow models serve -m "models:/xgboost-house-price/Production" -p 5001
   ↓
7. Predict: curl -X POST localhost:5001/invocations -d '{"dataframe_split": {...}}'
```

### Pros
- Full experiment history (never lose a run)
- Model versioning with stage gates
- Built-in REST API serving
- UI for comparing runs
- Supports multiple ML frameworks

### Cons
- Extra infrastructure (MLflow server + backend store)
- Serving is basic — no auto-scaling, no scale-to-zero
- Single-process server (not production-grade for high traffic)

### You Learn
- Experiment tracking discipline
- Model registry workflow (staging → production)
- REST API model serving
- Artifact management

### Integration with Level 1

```
Level 1 artifacts     →    MLflow artifacts
────────────────           ────────────────
model.joblib          →    mlflow.xgboost.log_model()
model_meta.json       →    mlflow.log_params() + log_metrics()
requirements.txt      →    auto-generated conda.yaml
predict.py            →    mlflow models serve (replaces script)
```

---

## Level 3 — OpenFaaS on K8s

**What:** Serve the model as a serverless function on your K8s cluster.

### Components

```
K8s Cluster
├── OpenFaaS Gateway            ← HTTP routing
│   └── /function/predict-price ← model endpoint
├── Function Pod                ← runs prediction code
│   ├── Docker image            ← model + deps baked in
│   └── handler.py              ← load model, return prediction
├── Prometheus                  ← collect metrics
│   ├── request count
│   ├── latency (p50, p95, p99)
│   └── error rate
└── Grafana                     ← dashboards + alerts
    ├── prediction latency
    ├── requests per second
    └── model drift detection
```

### Workflow

```
1. Build Docker image with model artifact
   ↓
2. faas-cli deploy -f predict-price.yml
   ↓  creates
3. K8s: Deployment + Service + HPA
   ↓  inspect
4. kubectl get deployment,svc,hpa -n openfaas-fn
   ↓
5. curl http://gateway:8080/function/predict-price -d '{"sqft_living": 2080, ...}'
   ↓
6. Prometheus scrapes metrics → Grafana dashboards
```

### Pros
- Scale-to-zero (no cost when idle)
- Auto-scaling under load
- K8s-native — fits your existing infra
- Built-in Prometheus metrics
- Easy rollback (deploy previous image)

### Cons
- Cold start latency (scale from zero takes seconds)
- Need Docker image per model version
- More infrastructure to manage
- Overkill for low-traffic use cases

### You Learn
- Containerizing ML models
- Serverless function patterns
- K8s resource management (Deployments, Services, HPA)
- Production monitoring (Prometheus + Grafana)

### Integration with MLflow (Level 2 + 3)

```
MLflow (registry)          OpenFaaS (serving)
─────────────────          ──────────────────
Train → log → register     Build image → deploy → serve
        ↓                          ↑
  Promote to Production ──→ Trigger new deployment
        ↓                          ↓
  Track metrics ←───────── Prometheus feeds back
```

---

## Comparison

| Aspect | Level 1 (Native) | Level 2 (MLflow) | Level 3 (OpenFaaS) |
|--------|-------------------|-------------------|---------------------|
| **Serving** | CLI / script | REST API | HTTP function |
| **Scaling** | None | Single process | Auto-scale + scale-to-zero |
| **Versioning** | Manual (v1, v2 dirs) | Model registry | Docker image tags |
| **Tracking** | None | Full experiment history | N/A (use MLflow) |
| **Monitoring** | None | Basic (MLflow UI) | Prometheus + Grafana |
| **Infra needed** | Python | Python + MLflow server | K8s + OpenFaaS + Docker |
| **Best for** | Learning, batch | Development, small teams | Production, variable load |
| **Complexity** | Low | Medium | High |

---

## Migration Path

```
NOW                     NEXT                    LATER
───                     ────                    ─────
Level 1 (Native)   →   Level 2 (MLflow)    →   Level 3 (OpenFaaS)
                    │                       │
What changes:       │   What changes:       │
- Add mlflow.log()  │   - Dockerize model   │
  to training code  │   - Write handler.py  │
- Use registry      │   - Deploy to K8s     │
  instead of v1/v2  │   - Add monitoring    │
- Serve via mlflow  │   - Connect MLflow    │
                    │     registry to deploy │
                    │                       │
What stays:         │   What stays:         │
- Same model code   │   - MLflow tracking   │
- Same features     │   - Same model format │
- Same metrics      │   - Same predict code │
```

Each level adds capability without throwing away previous work.
