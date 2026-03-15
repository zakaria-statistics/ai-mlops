# ML Essentials — Reference Cheat Sheet
> Quick-lookup for algorithms, metrics, data prep, and problem mapping

## Table of Contents
1. [ML Problem Types](#ml-problem-types) - Taxonomy tree
2. [Problem → Algorithm → Data → Business Map](#algorithm-map) - What to use when
3. [ML Pipeline](#ml-pipeline) - Universal 9-step process
4. [Evaluation Metrics](#evaluation-metrics) - Which metric for which problem
5. [Data Preparation Toolkit](#data-preparation-toolkit) - Scaling, encoding, splitting
6. [Algorithm Complexity Cheat Sheet](#algorithm-complexity) - When each algo shines or fails
7. [Math Symbols Quick Reference](#math-symbols) - Common notation

---

## ML Problem Types

```
ML
├── Supervised Learning (labeled data: X → y)
│   ├── Regression           → predict continuous NUMBER
│   │   "How much? How many?"
│   └── Classification       → predict discrete CATEGORY
│       ├── Binary           → 2 classes (yes/no, spam/ham)
│       └── Multi-class      → 3+ classes (cat/dog/bird)
│
├── Unsupervised Learning (no labels: just X)
│   ├── Clustering           → find GROUPS
│   │   "Which data points belong together?"
│   ├── Dimensionality Reduction → compress FEATURES
│   │   "Can I represent 100 features with 3?"
│   └── Anomaly Detection    → find OUTLIERS
│       "Which data points are abnormal?"
│
└── Reinforcement Learning (agent + environment + reward)
    "Learn by trial and error" → [not covered in essentials]
```

---

## Algorithm Map

### Regression Algorithms

| Algorithm | Math Core | Best Data For | Business Example | Complexity |
|-----------|-----------|---------------|------------------|------------|
| Linear Regression | y = Xw + b, minimize MSE | Continuous, linear relationships | House price, salary prediction | Simple |
| Ridge (L2) | + α·Σwⱼ² penalty | Many features, multicollinearity | Sales prediction with 50+ features | Simple |
| Lasso (L1) | + α·Σ\|wⱼ\| penalty | Feature selection needed | Identify top 5 predictors from 100 | Simple |
| ElasticNet | L1 + L2 combined | Best of both worlds | Genomics, many correlated features | Simple |
| Decision Tree | Variance reduction splits | Non-linear, mixed types | Insurance claim amount | Medium |
| Random Forest | Bagging + feature sampling | Tabular, non-linear | Demand forecasting | Medium |
| XGBoost | Sequential boosting + regularization | Tabular, competitions | Revenue prediction | Complex |

### Classification Algorithms

| Algorithm | Math Core | Best Data For | Business Example | Complexity |
|-----------|-----------|---------------|------------------|------------|
| Logistic Regression | σ(z) = 1/(1+e⁻ᶻ), log-loss | Linear boundary, interpretable | Churn yes/no, spam detection | Simple |
| KNN | Distance voting (Euclidean/Manhattan) | Small data, local patterns | Customer similarity matching | Simple |
| SVM | Maximize margin, kernel trick | High-dimensional, clear margin | Cancer malignant/benign | Medium |
| Decision Tree | Gini/Entropy split criterion | Interpretable rules needed | Loan approval yes/no | Medium |
| Naive Bayes | P(y\|X) ∝ P(X\|y)·P(y), Bayes theorem | Text, categorical, small data | Email spam, sentiment | Simple |
| Random Forest | Ensemble of trees, majority vote | Tabular, many classes | Product category prediction | Medium |
| XGBoost | Boosted trees, softmax for multi-class | Tabular, competitions | Customer segment A/B/C/D | Complex |
| Neural Network | Layers of σ(Wx+b), backprop | Complex boundaries, large data | Digit recognition, image class | Complex |

### Unsupervised Algorithms

| Algorithm | Math Core | Best Data For | Business Example |
|-----------|-----------|---------------|------------------|
| K-Means | Minimize within-cluster distance | Spherical clusters, known K | Customer segmentation |
| DBSCAN | Density-based, ε-neighborhood | Arbitrary shapes, noise | Geographic hotspot detection |
| Hierarchical | Agglomerative/Divisive linkage | Need dendrogram, small data | Gene expression grouping |
| PCA | Eigendecomposition of covariance | High-dimensional numerical | Compress 50 features to 2-3 |
| t-SNE | Probability distribution matching | Visualization only (2D/3D) | Visualize cluster structure |
| Isolation Forest | Random splits, short path = anomaly | Tabular anomaly detection | Fraud detection |

---

## ML Pipeline

```
Step 1: Business Problem
  │  "What decision are we automating?"
  │  → Define success metric, stakeholders, constraints
  ↓
Step 2: Data Collection
  │  "What data exists? What's needed?"
  │  → Sources, volume, quality, labeling
  ↓
Step 3: EDA (Exploratory Data Analysis)
  │  "What does the data look like?"
  │  → Distributions, correlations, missing values, outliers
  ↓
Step 4: Data Preparation
  │  "Make data algorithm-ready"
  │  → Clean, encode, scale, split (train/val/test)
  ↓
Step 5: Feature Engineering
  │  "Create better inputs"
  │  → Derive features, select important ones, transform
  ↓
Step 6: Modeling
  │  "Train algorithm(s)"
  │  → Start simple (baseline), increase complexity
  ↓
Step 7: Evaluation
  │  "How good is the model?"
  │  → Metrics, cross-validation, error analysis
  ↓
Step 8: Tuning
  │  "Can we do better?"
  │  → Hyperparameters, GridSearch/RandomSearch, regularization
  ↓
Step 9: Deployment
     "Ship it"
     → Package, API, container, monitor
```

---

## Evaluation Metrics

### Regression Metrics

| Metric | Formula | Intuition | When to use |
|--------|---------|-----------|-------------|
| MAE | (1/n)·Σ\|yᵢ - ŷᵢ\| | Average error in original units | Interpretable, robust to outliers |
| RMSE | √((1/n)·Σ(yᵢ - ŷᵢ)²) | Penalizes large errors more | When big errors are costly |
| R² | 1 - SS_res/SS_tot | % of variance explained (0→1) | Compare models, overall fit |
| MAPE | (100/n)·Σ\|yᵢ - ŷᵢ\|/\|yᵢ\| | Percentage error | Business reporting |

### Classification Metrics

| Metric | Formula | Intuition | When to use |
|--------|---------|-----------|-------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | % correct overall | ONLY when classes are balanced |
| Precision | TP/(TP+FP) | "Of predicted positive, how many correct?" | Cost of false positive is high (spam → inbox) |
| Recall | TP/(TP+FN) | "Of actual positive, how many found?" | Cost of false negative is high (miss cancer) |
| F1 | 2·(P·R)/(P+R) | Harmonic mean of precision & recall | Balance both errors |
| ROC-AUC | Area under ROC curve | Overall ranking quality across thresholds | Compare models, probability output |
| Confusion Matrix | 2x2 (or NxN) table | See all error types | Always — visual debugging |

### Clustering Metrics

| Metric | Intuition | When to use |
|--------|-----------|-------------|
| Silhouette Score | How similar to own cluster vs nearest (-1→1) | Compare K values, cluster quality |
| Inertia | Sum of distances to centroid | Elbow method for K-Means |
| Davies-Bouldin | Ratio of within/between cluster distances | Lower = better separation |

---

## Data Preparation Toolkit

### Splitting Strategy

| Method | When | Why |
|--------|------|-----|
| Train/Test (80/20) | Simple evaluation | Honest performance estimate |
| Train/Val/Test (60/20/20) | Hyperparameter tuning | Val for tuning, test untouched |
| K-Fold Cross-validation | Small data, model comparison | More robust, uses all data |
| Stratified split | Imbalanced classes | Preserves class ratios in each split |
| Time-based split | Time series | Future data cannot leak into training |

### Scaling

| Method | Formula | When | Algorithms that need it |
|--------|---------|------|------------------------|
| StandardScaler | (x - μ) / σ | Default for most | SVM, KNN, Linear/Logistic, PCA |
| MinMaxScaler | (x - min) / (max - min) | Bounded 0-1 needed | Neural networks |
| RobustScaler | (x - median) / IQR | Outliers present | Same as Standard but robust |
| No scaling needed | — | Tree-based models | Decision Tree, Random Forest, XGBoost |

### Encoding Categorical Features

| Method | When | Example |
|--------|------|---------|
| One-Hot Encoding | Nominal (no order): color, city | red → [1,0,0], blue → [0,1,0] |
| Label/Ordinal Encoding | Ordinal (has order): low/med/high | low→0, med→1, high→2 |
| Target Encoding | High cardinality (1000 cities) | Replace category with mean of target |

### Handling Imbalanced Data

| Method | How | When |
|--------|-----|------|
| Class weights | Algorithm penalizes minority errors more | First try, simplest |
| SMOTE | Generate synthetic minority samples | Moderate imbalance |
| Undersampling | Remove majority samples | Large dataset, extreme imbalance |
| Threshold tuning | Adjust decision boundary | When you need specific precision/recall |

### Handling Missing Values

| Method | When |
|--------|------|
| Drop rows | Very few missing, large dataset |
| Drop column | >50% missing in that feature |
| Mean/Median impute | Numerical, random missingness |
| Mode impute | Categorical |
| KNN impute | Pattern in missingness |
| Flag + impute | Missingness itself is informative |

---

## Algorithm Complexity

### When Each Algorithm Shines vs Fails

| Algorithm | Shines when | Fails when |
|-----------|-------------|------------|
| Linear/Logistic Regression | Linear relationship, interpretability needed | Non-linear data, interactions |
| KNN | Small data, local patterns | High dimensions (curse of dimensionality), large data (slow) |
| SVM | Clear margin, high-dimensional | Large datasets (slow), need probability output |
| Decision Tree | Interpretability, mixed feature types | Overfits easily, unstable (small data change → different tree) |
| Random Forest | General tabular data, robust | Very high dimensional, need interpretability |
| XGBoost | Tabular competitions, best accuracy | Small data (overfits), slow to tune |
| Naive Bayes | Text, small data, fast baseline | Feature independence assumption violated |
| Neural Network | Large data, complex patterns | Small data, interpretability needed, slow |
| K-Means | Spherical clusters, known K | Non-spherical, unknown K, varying density |
| PCA | Reduce dimensions, decorrelate | Non-linear relationships (use t-SNE for viz) |

---

## Math Symbols

| Symbol | Meaning |
|--------|---------|
| X | Feature matrix (n samples × m features) |
| y | Target vector (what we predict) |
| ŷ (y-hat) | Predicted value |
| w, θ | Weights / parameters |
| b | Bias term |
| α, λ | Regularization strength |
| η | Learning rate |
| σ(z) | Sigmoid function: 1/(1+e⁻ᶻ) |
| ∇ | Gradient (vector of partial derivatives) |
| J(w) | Cost/loss function |
| Σ | Summation |
| ∂ | Partial derivative |
| \|\|w\|\|₂ | L2 norm (Euclidean length of weight vector) |
| argmin | "The value that minimizes..." |
