# Phase 8: Packaging

> Serialize the best model into a portable, reproducible artifact.

## Table of Contents

1. [Goal](#goal)
2. [How to Run](#how-to-run)
3. [Artifact Contents](#artifact-contents)
4. [Usage](#usage)

---

## Goal

Turn the trained model into a standalone artifact that can be loaded and used without retraining — on any machine with the right dependencies.

## How to Run

```bash
cd /root/claude/ai_mlops
source .venv/bin/activate
pip install scikit-learn xgboost pandas numpy joblib  # required
```

Open `notebooks/05-packaging.py` in VS Code and run cells with `# %%` markers.

### Libraries

| Library | Purpose | Required |
|---------|---------|----------|
| pandas | Data loading | Yes |
| numpy | Numerical operations | Yes |
| xgboost | Model training | Yes |
| scikit-learn | Metrics verification | Yes |
| joblib | Model serialization | Yes |

## Artifact Contents

Output saved to `models/v1/`:

| File | Purpose |
|------|---------|
| `model.joblib` | Serialized XGBoost model |
| `model_meta.json` | Feature names, metrics, schema |
| `requirements.txt` | Pinned dependency versions |
| `predict.py` | Standalone prediction script |

## Usage

```bash
# Install dependencies
pip install -r models/v1/requirements.txt

# Run example prediction
python models/v1/predict.py

# Predict from CSV
python models/v1/predict.py --csv new_houses.csv
```

## Next Step

-> Proceed to [09-deployment](../09-deployment/) for API serving and containerization.
