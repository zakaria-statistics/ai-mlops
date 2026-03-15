# ML Essentials Roadmap
> Complete machine learning curriculum: math-first, then code

## Table of Contents
1. [Structure](#structure) - How theory and application dirs work together
2. [Learning Path](#learning-path) - Sequential order with dependencies
3. [Progress Tracker](#progress-tracker) - Checklist

## Structure

```
ml_roadmap/
├── theory/       ← Math, derivations, formulas, intuition (read FIRST)
├── application/  ← Notebooks, datasets, code (do SECOND)
└── REFERENCE.md  ← Cheat sheet: algo taxonomy, metrics, data prep
```

Each topic follows: **Math → Numpy scratch → Library → Compare**

## Learning Path

### Phase 1: Foundations (Labs 00-02)
> "Before any algorithm, understand the data and the pipeline"

```
00-ml-landscape ──→ 01-data-and-eda ──→ 02-data-preparation
(what is ML?)       (explore data)       (prepare for modeling)
```

### Phase 2: Supervised — Regression (Lab 03)
> "Predict a number: the math behind fitting a line"

```
03-linear-regression
(OLS → Ridge → Lasso → weight update by hand)
```

### Phase 3: Supervised — Classification (Labs 04-08)
> "Predict a category: from sigmoid to decision boundaries"

```
04-logistic-regression ──→ 05-knn-and-svm ──→ 06-decision-trees
(sigmoid, log-loss)        (distance, kernels)  (gini, entropy)
        │
        ↓
07-ensemble-methods ──→ 08-naive-bayes-and-text
(bagging, boosting)     (probability, TF-IDF)
```

### Phase 4: Unsupervised (Labs 09-10)
> "No labels: find structure in the data"

```
09-unsupervised-learning ──→ 10-dimensionality-reduction
(K-Means, DBSCAN)            (PCA, t-SNE)
```

### Phase 5: Evaluation Mastery (Lab 11)
> "Ties it all together: how to properly judge and compare models"

```
11-evaluation-deep-dive
(cross-val, learning curves, bias-variance, model selection)
```

### Phase 6: Deep Learning (Labs 12-13)
> "Neural networks: from a single neuron to image recognition"

```
12-neural-networks ──→ 13-cnn-basics
(perceptron, MLP,      (convolutions, pooling,
 backprop by hand,      transfer learning)
 weight adjustment)
```

### Phase 7: Sequential & Time Data (Labs 14-15)
> "Data with order matters: time series and sequences"

```
14-time-series ──→ 15-rnn-sequence
(ARIMA, trends)    (RNN, LSTM, weight gates)
```

### Phase 8: Capstone (Lab 16)
> "Prove it: full pipeline on a real problem"

```
16-capstone
(business problem → EDA → model → eval → deploy)
```

## Progress Tracker

- [ ] 00 — ML Landscape
- [ ] 01 — Data & EDA
- [ ] 02 — Data Preparation
- [ ] 03 — Linear Regression
- [ ] 04 — Logistic Regression
- [ ] 05 — KNN & SVM
- [ ] 06 — Decision Trees
- [ ] 07 — Ensemble Methods
- [ ] 08 — Naive Bayes & Text
- [ ] 09 — Unsupervised Learning
- [ ] 10 — Dimensionality Reduction
- [ ] 11 — Evaluation Deep Dive
- [ ] 12 — Neural Networks
- [ ] 13 — CNN Basics
- [ ] 14 — Time Series
- [ ] 15 — RNN & Sequences
- [ ] 16 — Capstone
