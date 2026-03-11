# Real Estate Price Prediction — ML Learning Roadmap

> End-to-end ML project: from business problem to deployed model, built iteratively.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Learning Path](#learning-path)
3. [Phase Map](#phase-map)
4. [Current Status](#current-status)

---

## Project Overview

**Goal:** Predict residential property sale prices — one model serving both buyers ("Am I overpaying?") and sellers ("What should I list at?").

**Approach:** Learn ML fundamentals by building a real project iteratively — each phase teaches specific concepts, and each model level shows WHY you need the next technique.

**Dataset:** King County House Sales (~21K rows, 19 features)

**After this project:** Apply the same workflow to other domains — see `future-ideas.md` for ideas (used cars, rentals, churn, etc.)

## Learning Path

```
                    YOU ARE HERE
                        ↓
┌─────────────────────────────────────────────────────────┐
│  PHASE 1: UNDERSTAND                                    │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │ Business  │→ │   Data    │→ │   Data    │          │
│  │ Problem   │  │Collection │  │Exploration│          │
│  │  01-*     │  │  02-*     │  │  03-*     │          │
│  └───────────┘  └───────────┘  └───────────┘          │
├─────────────────────────────────────────────────────────┤
│  PHASE 2: BUILD                                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │   Data    │→ │ Feature   │→ │ Modeling  │          │
│  │   Prep    │  │Engineering│  │ (train)   │          │
│  │  04-*     │  │  05-*     │  │  06-*     │          │
│  └───────────┘  └───────────┘  └───────────┘          │
├─────────────────────────────────────────────────────────┤
│  PHASE 3: VALIDATE                                      │
│  ┌───────────┐                                          │
│  │Evaluation │  Compare models, select best, validate  │
│  │  07-*     │                                          │
│  └───────────┘                                          │
├─────────────────────────────────────────────────────────┤
│  PHASE 4: SHIP (added iteratively)                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │ Package   │→ │  Deploy   │→ │ Monitor   │          │
│  │  08-*     │  │  09-*     │  │  10-*     │          │
│  └───────────┘  └───────────┘  └───────────┘          │
└─────────────────────────────────────────────────────────┘
```

## Phase Map

| # | Phase | What You Learn | Key Output |
|---|-------|---------------|------------|
| 01 | Business Problem | Problem framing, success metrics, ML type selection | Problem statement doc |
| 02 | Data Collection | Finding datasets, data sources, initial validation | Raw dataset in `data/raw/` |
| 03 | Data Exploration | EDA, distributions, correlations, outliers, visualization | Exploration notebook + insights |
| 04 | Data Preparation | Cleaning, missing values, encoding, scaling, train/test split | Clean dataset in `data/processed/` |
| 05 | Feature Engineering | Creating features, selection, domain knowledge application | Feature pipeline |
| 06 | Modeling | Algorithm progression (linear → ensemble), training, tuning | Trained models |
| 07 | Evaluation | Metrics, cross-validation, model comparison, error analysis | Best model selected |
| 08 | Packaging | Serialization, dependency management, reproducibility | Packaged model artifact |
| 09 | Deployment | API serving, containerization | Running prediction service |
| 10 | Monitoring | Drift detection, performance tracking | Monitoring dashboard |

## Model Progression (Phase 6 Detail)

```
Level 1: Linear Regression
   ↓ "Why not enough?" → Can't capture non-linear relationships
Level 2: Polynomial Regression + Regularization (Ridge/Lasso)
   ↓ "Why not enough?" → Still assumes global pattern, can overfit
Level 3: Decision Tree
   ↓ "Why not enough?" → High variance, overfits easily
Level 4: Random Forest
   ↓ "Why not enough?" → Good but slow, can improve accuracy
Level 5: Gradient Boosting (XGBoost/LightGBM)
   ↓ "Final comparison" → Usually best for tabular data
Level 6: Model Selection & Comparison
```

## Current Status

- [x] Phase 01 — Business Problem defined
- [x] Phase 02 — Data collection guide ready
- [ ] Phase 02 — Download dataset
- [ ] Phase 03 — Data exploration
- [ ] Phase 04 — Data preparation
- [ ] Phase 05 — Feature engineering
- [ ] Phase 06 — Modeling
- [ ] Phase 07 — Evaluation
- [ ] Phase 08 — Packaging
- [ ] Phase 09 — Deployment
- [ ] Phase 10 — Monitoring

---

## Directory Structure

```
ai_mlops/
├── roadmap.md              ← You are here
├── CLAUDE.md               ← Project instructions
├── data/
│   ├── raw/                ← Original downloaded data (never modify)
│   └── processed/          ← Cleaned, transformed data
├── notebooks/              ← Jupyter notebooks for exploration
├── src/                    ← Python source code (models, pipelines)
├── 01-business-problem/    ← Problem definition docs
├── 02-data-collection/     ← Dataset sourcing guides
├── 03-data-exploration/    ← EDA results and insights
├── 04-data-preparation/    ← Cleaning and preprocessing
├── 05-feature-engineering/ ← Feature creation and selection
├── 06-modeling/            ← Model training code and results
├── 07-evaluation/          ← Model comparison and validation
├── 08-packaging/           ← Model serialization and packaging
├── 09-deployment/          ← API and serving code
├── 10-monitoring/          ← Drift detection and metrics
├── future-ideas.md         ← Next project ideas (used cars, rent, churn, etc.)
└── archive/                ← Previous docs (reference)
```
