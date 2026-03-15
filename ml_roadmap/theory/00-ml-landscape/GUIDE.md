# The Machine Learning Landscape
> A math-first map of ML types, the universal pipeline, and the forces that govern every model.

## Table of Contents
1. [What Is Machine Learning?](#1-what-is-machine-learning) — Learning from data instead of explicit rules
2. [ML Types Taxonomy](#2-ml-types-taxonomy) — Supervised, unsupervised, reinforcement and their sub-types
3. [The Universal ML Pipeline](#3-the-universal-ml-pipeline) — Every project follows this skeleton
4. [Mapping Business Problems to ML Types](#4-mapping-business-problems-to-ml-types) — Decision framework
5. [Overfitting and Underfitting](#5-overfitting-and-underfitting) — The two failure modes
6. [The Bias-Variance Tradeoff](#6-the-bias-variance-tradeoff) — The fundamental tension in all of ML
7. [What to Look for in the Application Lab](#7-what-to-look-for-in-the-application-lab)

---

## 1. What Is Machine Learning?

Traditional programming:

```
  Input Data  ──────┐
                     ├──►  Program  ──►  Output
  Rules (human) ────┘
```

Machine learning flips this:

```
  Input Data  ──────┐
                     ├──►  Learning Algorithm  ──►  Rules (model)
  Expected Output ──┘
```

The core idea: instead of a human writing rules, an algorithm **discovers** rules
(parameters, weights, structure) by examining data.

A model is just a function with adjustable parameters:

```
  ŷ = f(X; θ)

  where:
    X = input features (data)
    θ = parameters the algorithm learns
    ŷ = prediction
```

"Training" means finding the θ that makes ŷ close to the true y.

---

## 2. ML Types Taxonomy

```
Machine Learning
├── Supervised Learning
│   ├── Regression        (predict a continuous number)
│   │   ├── Linear Regression
│   │   ├── Ridge / Lasso
│   │   ├── Decision Tree Regressor
│   │   └── Random Forest Regressor ...
│   └── Classification    (predict a discrete category)
│       ├── Logistic Regression
│       ├── Decision Tree Classifier
│       ├── SVM
│       └── Neural Networks ...
│
├── Unsupervised Learning
│   ├── Clustering        (find groups)
│   │   ├── K-Means
│   │   ├── DBSCAN
│   │   └── Hierarchical ...
│   ├── Dimensionality Reduction (compress features)
│   │   ├── PCA
│   │   └── t-SNE ...
│   └── Association Rules (find co-occurrences)
│       └── Apriori ...
│
├── Reinforcement Learning
│   └── Agent learns by trial-and-error in an environment
│       ├── Q-Learning
│       ├── Policy Gradient
│       └── Deep RL ...
│
└── Semi-Supervised / Self-Supervised
    └── Small labeled set + large unlabeled set
```

### Supervised Learning — The Key Distinction

You have **labeled** data: each input X comes with a known output y.

| Sub-type | Output y | Example |
|----------|----------|---------|
| Regression | Continuous number | House price = $342,000 |
| Classification | Discrete category | Email = spam / not-spam |

The model learns the mapping f: X -> y by minimizing a **loss function**:

```
  Training objective:  minimize L(y, ŷ)  over all training examples

  Regression loss (MSE):      L = (1/n) Σ (yi - ŷi)²
  Classification loss (log):  L = -(1/n) Σ [yi·log(ŷi) + (1-yi)·log(1-ŷi)]
```

### Unsupervised Learning

No labels. The algorithm finds **structure** in X alone.

```
  Clustering:  Assign each xi to a group k that minimizes
               within-cluster variance  Σ ||xi - μk||²

  PCA:         Find directions of maximum variance in X
               Project X onto top-k eigenvectors of covariance matrix
```

### Reinforcement Learning

An agent takes actions in an environment, receives rewards, learns a policy.

```
  Agent ──action──► Environment
    ▲                    │
    └──reward + state────┘

  Goal: maximize cumulative reward  Σ γᵗ · rₜ
  where γ = discount factor (0 < γ ≤ 1)
```

---

## 3. The Universal ML Pipeline

Every ML project, regardless of algorithm, follows this skeleton:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐
│  1. Define  │───►│  2. Collect  │───►│  3. Explore  │───►│ 4. Prep  │
│   Problem   │    │    Data      │    │   (EDA)      │    │  Data    │
└─────────────┘    └─────────────┘    └─────────────┘    └────┬─────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐
│  8. Deploy  │◄───│  7. Tune    │◄───│  6. Evaluate │◄───│ 5. Train │
│  & Monitor  │    │  Hyperparam │    │   Model      │    │  Model   │
└─────────────┘    └─────────────┘    └─────────────┘    └──────────┘
```

| Step | What happens | Key output |
|------|-------------|------------|
| 1. Define | Frame as regression/classification/clustering | Problem type + metric |
| 2. Collect | Gather raw data | Dataset |
| 3. Explore | Statistics, plots, correlations | Feature understanding |
| 4. Prep | Clean, scale, encode, split | Train/test sets |
| 5. Train | Fit model to training data | Learned parameters θ |
| 6. Evaluate | Measure on test data | Performance metric |
| 7. Tune | Adjust hyperparameters | Best configuration |
| 8. Deploy | Serve predictions in production | API / pipeline |

---

## 4. Mapping Business Problems to ML Types

### Decision Framework

```
"What do I want to predict or discover?"
         │
         ├── I have labeled examples
         │       │
         │       ├── Target is a number? ──────► REGRESSION
         │       │     "How much / how many?"
         │       │
         │       └── Target is a category? ────► CLASSIFICATION
         │             "Which class / yes-no?"
         │
         ├── I have NO labels
         │       │
         │       ├── Find groups? ─────────────► CLUSTERING
         │       │
         │       ├── Reduce dimensions? ───────► DIMENSIONALITY REDUCTION
         │       │
         │       └── Find patterns? ───────────► ASSOCIATION / ANOMALY
         │
         └── Agent interacts with environment ─► REINFORCEMENT LEARNING
```

### By-Hand Example: Mapping 5 Business Problems

| Business Problem | Target | Type |
|------------------|--------|------|
| "Predict next month's revenue" | $amount (continuous) | Regression |
| "Will this customer churn?" | yes/no (binary) | Classification |
| "Segment our customers" | no label, find groups | Clustering |
| "Which products are bought together?" | no label, find co-occurrence | Association |
| "Optimize ad bidding in real-time" | maximize clicks via actions | Reinforcement |

> **Key Intuition:** The nature of your **target variable** (or absence of one) determines
> the ML type. Everything else — algorithm choice, metrics, evaluation — follows from this
> first decision.

---

## 5. Overfitting and Underfitting

### The Two Failure Modes

```
  Model Complexity ──────────────────────────────►

  Error
   ▲
   │ \                              ·  ·
   │   \    Underfitting          ·      Training error
   │    \   zone               ·         (keeps decreasing)
   │     \                   ·
   │      ·  ·  ·  ·  ·  ·
   │                  ▲
   │                  │ Sweet spot
   │            · · ·
   │          ·                   Overfitting
   │        ·                     zone
   │       · Test error
   │      ·  (starts increasing)
   └──────────────────────────────────────────► Complexity
```

**Underfitting** (high bias):
- Model is too simple to capture patterns
- Poor performance on BOTH training and test data
- Example: fitting a straight line to curved data

**Overfitting** (high variance):
- Model memorizes training data, including noise
- Great on training data, poor on test data
- Example: fitting a degree-20 polynomial to 10 data points

### Visual: Fitting a Curve

```
  True relationship: y = x²

  Underfitting              Good fit              Overfitting
  (linear)                  (quadratic)           (degree 10)

  y│     ·                  y│     ·              y│   ··
   │   · ·     /            │   · ·  ·             │  ·  ··
   │  ·    · /              │  ·    ·               │ · ·   ··
   │ ·   · /                │ ·   ·                 │·  · ·  ·
   │·  · /                  │·  ·                   │· · ·  ·
   │· · /                   │· ·                    │·· ·  ·
   └──────── x              └──────── x             └──────── x
   Line ignores curve       Captures the shape      Wiggles through
                                                    every point
```

### Detecting It

| Symptom | Training error | Test error | Diagnosis |
|---------|---------------|------------|-----------|
| Both high | High | High | Underfitting |
| Train low, test high | Low | High | Overfitting |
| Both low, close together | Low | Low | Good fit |

### Remedies

| Underfitting | Overfitting |
|-------------|-------------|
| More complex model | Simpler model |
| Add features | Remove features |
| Less regularization | More regularization |
| Train longer | Early stopping |
| — | More training data |

---

## 6. The Bias-Variance Tradeoff

This is the mathematical foundation behind overfitting and underfitting.

### Decomposing Prediction Error

For any model, the expected prediction error on a new data point can be decomposed:

```
  E[(y - ŷ)²]  =  Bias²  +  Variance  +  Irreducible Noise

  where:
    Bias²     = [E[ŷ] - f(x)]²        how far the average prediction
                                        is from the true value

    Variance  = E[(ŷ - E[ŷ])²]        how much predictions vary
                                        across different training sets

    Noise     = σ²                      randomness in the data itself
                                        (can never be removed)
```

### Intuition with a Dartboard Analogy

```
  Low Bias, Low Variance    Low Bias, High Variance
  (ideal)                   (overfitting)

      ┌───────┐                 ┌───────┐
      │  ·    │                 │ ·     │
      │ ··●·  │                 │  ●· · │
      │  ·    │                 │·    · │
      └───────┘                 └───────┘
   Clustered on bullseye     Centered but scattered

  High Bias, Low Variance   High Bias, High Variance
  (underfitting)             (worst case)

      ┌───────┐                 ┌───────┐
      │   ●   │                 │   ●   │
      │       │                 │     · │
      │  ···  │                 │·      │
      │  ··   │                 │    ·  │
      └───────┘                 └───────┘
   Clustered but off-center  Scattered and off-center

  ● = bullseye (true value)
  · = individual predictions from models trained on different samples
```

### The Tradeoff

```
  Error
   ▲
   │\                          /
   │ \   Bias²              /  Variance
   │  \                   /
   │   \               /
   │    ──── · · · ────
   │         ▲
   │         │
   │    Optimal complexity
   │    (minimizes total error)
   └──────────────────────────► Model Complexity
```

- **Simple models**: High bias, low variance (consistently wrong in the same way)
- **Complex models**: Low bias, high variance (right on average but unstable)
- **Goal**: Find the complexity sweet spot that minimizes **total** error

> **Key Intuition:** You can never eliminate both bias and variance simultaneously.
> Every modeling decision is a negotiation between these two forces.
> Regularization, cross-validation, and ensemble methods are all strategies
> for managing this tradeoff.

### By-Hand Example: Bias-Variance with 3 Training Sets

Suppose the true function is f(x) = 2x, and we have noise σ = 0.5.

Three different training samples give us three fitted models:

```
  Training Set 1: ŷ = 1.8x + 0.3   →  prediction at x=2:  3.9
  Training Set 2: ŷ = 2.1x - 0.1   →  prediction at x=2:  4.1
  Training Set 3: ŷ = 1.9x + 0.2   →  prediction at x=2:  4.0

  True value at x=2:  f(2) = 4.0

  Average prediction:  E[ŷ] = (3.9 + 4.1 + 4.0) / 3 = 4.0

  Bias² = (E[ŷ] - f(2))² = (4.0 - 4.0)² = 0.0    ← unbiased

  Variance = E[(ŷ - E[ŷ])²]
           = [(3.9-4.0)² + (4.1-4.0)² + (4.0-4.0)²] / 3
           = [0.01 + 0.01 + 0.00] / 3
           = 0.0067                                   ← low variance

  Total error ≈ 0.0 + 0.0067 + 0.25 = 0.257
                bias²  var     noise(σ²=0.5²)
```

Now imagine a very complex model that overfits:

```
  Training Set 1: prediction at x=2:  4.8
  Training Set 2: prediction at x=2:  3.2
  Training Set 3: prediction at x=2:  4.0

  E[ŷ] = (4.8 + 3.2 + 4.0) / 3 = 4.0

  Bias² = (4.0 - 4.0)² = 0.0      ← still unbiased on average!

  Variance = [(4.8-4.0)² + (3.2-4.0)² + (4.0-4.0)²] / 3
           = [0.64 + 0.64 + 0.00] / 3
           = 0.427                   ← high variance!

  Total error ≈ 0.0 + 0.427 + 0.25 = 0.677   ← worse because of variance
```

The complex model is unbiased but its high variance makes it worse overall.

---

## 7. What to Look for in the Application Lab

When you move to code, watch for these connections:

| Theory concept | What you will see in code |
|---|---|
| ML type selection | `from sklearn.linear_model import LinearRegression` vs `LogisticRegression` |
| Loss function | `model.score()`, `mean_squared_error()`, `log_loss()` |
| Train/test split | `train_test_split(X, y, test_size=0.2)` |
| Overfitting detection | Training score >> test score |
| Bias-variance | Comparing simple vs complex models on same data |
| Pipeline | The order of your code cells: load, explore, prep, train, evaluate |

**Questions to ask yourself during the lab:**
1. What type of problem is this? (regression / classification / clustering)
2. After training, is my training error much lower than test error? (overfitting check)
3. Are both errors high? (underfitting check)
4. What would happen if I used a simpler or more complex model?
