# 16 — Capstone: Methodology Guide
> How to approach any ML problem from scratch — checklists, decision frameworks, and common mistakes

## Table of Contents
1. [Problem Framing Checklist](#1-problem-framing)
2. [Model Selection Decision Tree](#2-model-selection)
3. [The Iterative Loop](#3-iterative-loop)
4. [Common Mistakes](#4-common-mistakes)
5. [Production Readiness](#5-production)
6. [Portfolio Presentation](#6-portfolio)

---

## 1. Problem Framing

Before ANY code, answer these:

```
□ What decision is being automated?
□ What does success look like? (metric + threshold)
□ What data is available?
□ What would a human do without ML? (this is your baseline)
□ What's the cost of being wrong? (false positive vs false negative)
□ Is this regression, classification, clustering, or something else?
```

---

## 2. Model Selection

```
Is your data TABULAR (rows/columns)?
├── YES
│   ├── < 1000 samples → Logistic Reg / Linear Reg (simple, won't overfit)
│   ├── Need interpretability → Decision Tree / Logistic Reg
│   └── Want best accuracy → Random Forest → XGBoost
│
├── Is it IMAGES?
│   └── YES → CNN (start with pretrained ResNet/VGG)
│
├── Is it TEXT?
│   ├── Simple task, small data → Naive Bayes + TF-IDF
│   └── Complex task → LSTM / Transformer
│
├── Is it TIME SERIES?
│   ├── Univariate → ARIMA / Prophet
│   └── Multivariate → LSTM / XGBoost with lag features
│
└── Is it UNLABELED?
    ├── Find groups → K-Means / DBSCAN
    └── Reduce dimensions → PCA
```

---

## 3. Iterative Loop

```
1. Baseline (simplest model) → establish floor
2. Feature engineering → biggest gains usually here
3. Try 3+ algorithms → compare with cross-validation
4. Tune best model → hyperparameter search
5. Error analysis → WHERE does it fail? Fix that
6. Repeat 2-5 until diminishing returns
```

**Always start simple.** If logistic regression gets 80%, ask if you really need 85% before spending days on XGBoost tuning.

---

## 4. Common Mistakes

| Mistake | Why it's bad | Fix |
|---------|-------------|-----|
| Data leakage | Inflated test score, fails in production | Fit preprocessors on train only |
| Not stratifying | Imbalanced eval, unreliable metrics | StratifiedKFold |
| Wrong metric | Accuracy on imbalanced data is meaningless | F1, ROC-AUC for imbalanced |
| Overfitting to test set | Tuning on test = no honest evaluation | Use validation set or nested CV |
| No baseline | Can't tell if your model is actually good | Always compare vs simple model |
| Feature selection after split | Uses test info for feature decisions | Select features on train only |

---

## 5. Production

```
□ Model serialized (joblib/pickle/ONNX)
□ Feature schema documented (what inputs, what types, what ranges)
□ Preprocessing pipeline saved (scaler, encoder — not just model)
□ API endpoint with input validation
□ Error handling for bad inputs
□ Monitoring: track prediction distribution over time (drift detection)
□ Performance: latency < threshold for your use case
```

---

## 6. Portfolio

For your GitHub project:
```
□ Clear README: problem, approach, results (with numbers)
□ Clean notebook: narrative flow, not just code dumps
□ Reproducible: requirements.txt, random seeds, data source documented
□ Honest: show what didn't work too, not just the best model
□ Visual: plots > tables > text
```

---

## What to Do in the Application Lab

This IS the application lab. Pick a real problem, apply everything you've learned:
- Problem framing → EDA → Preparation → Modeling → Evaluation → Deployment
- Use the checklists above at each step
- Document your decisions and trade-offs

Suggested projects: credit card fraud, customer churn, image classifier, text sentiment, housing prices.
