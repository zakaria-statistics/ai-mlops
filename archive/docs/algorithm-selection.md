# ML Algorithm Selection Guide

> Choosing the right algorithm for your problem

## Table of Contents

1. [Quick Decision Tree](#quick-decision-tree) - Start here
2. [Classification](#classification) - Predict categories
3. [Regression](#regression) - Predict continuous values
4. [Clustering](#clustering) - Group similar data
5. [Dimensionality Reduction](#dimensionality-reduction) - Reduce features
6. [Time Series](#time-series) - Temporal predictions
7. [Anomaly Detection](#anomaly-detection) - Find outliers
8. [NLP](#natural-language-processing) - Text analysis
9. [Recommendation](#recommendation-systems) - Suggest items
10. [Deep Learning](#deep-learning) - When to use neural networks

---

## Quick Decision Tree

```
What's your goal?
│
├─► Predict a category? ──────────────► CLASSIFICATION
│   └─ (spam/not spam, churn/stay)
│
├─► Predict a number? ────────────────► REGRESSION
│   └─ (price, temperature, sales)
│
├─► Group similar items? ─────────────► CLUSTERING
│   └─ (customer segments, topics)
│
├─► Find unusual patterns? ───────────► ANOMALY DETECTION
│   └─ (fraud, defects, intrusions)
│
├─► Predict future values? ───────────► TIME SERIES
│   └─ (stock prices, demand)
│
├─► Reduce data complexity? ──────────► DIMENSIONALITY REDUCTION
│   └─ (visualization, preprocessing)
│
├─► Analyze text? ────────────────────► NLP
│   └─ (sentiment, topics, entities)
│
└─► Suggest items to users? ──────────► RECOMMENDATION
    └─ (products, content, matches)
```

---

## Classification

### Binary Classification (2 classes)

| Algorithm | Best When | Avoid When | Interpretable |
|-----------|-----------|------------|---------------|
| **Logistic Regression** | Linear relationships, need explainability, baseline | Non-linear data, complex interactions | Yes |
| **Decision Tree** | Need rules, mixed feature types | High variance needed to avoid | Yes |
| **Random Forest** | Robust performance, feature importance | Real-time inference, memory limited | Moderate |
| **XGBoost/LightGBM** | Best accuracy needed, tabular data | Very small datasets, need simplicity | Moderate |
| **SVM** | High-dimensional data, clear margin | Large datasets (slow), probability needed | No |
| **Naive Bayes** | Text classification, fast inference | Feature dependencies matter | Yes |
| **Neural Network** | Large data, complex patterns | Small data, need interpretability | No |

### Multi-class Classification (3+ classes)

| Algorithm | Best When | Avoid When |
|-----------|-----------|------------|
| **Logistic Regression (OvR)** | Baseline, linear separable | Many classes, complex boundaries |
| **Random Forest** | General purpose, robust | Hundreds of classes |
| **XGBoost** | Best performance on tabular | Simple problems |
| **SVM (OvO/OvR)** | Few classes, high dimensions | Many classes (slow) |
| **Neural Network** | Many classes, complex patterns | Small datasets |

### Multi-label Classification (multiple labels per sample)

| Algorithm | Best When |
|-----------|-----------|
| **Binary Relevance** | Labels are independent |
| **Classifier Chains** | Label dependencies exist |
| **Multi-output NN** | Complex label relationships |

### Algorithm Selection by Data Size

```
Samples        Recommended Algorithms
─────────────────────────────────────────────────
< 100          Logistic Regression, Naive Bayes
100 - 1K       Decision Tree, SVM, Logistic Regression
1K - 10K       Random Forest, XGBoost, SVM
10K - 100K     XGBoost, LightGBM, Neural Networks
100K - 1M      LightGBM, Neural Networks
> 1M           Neural Networks, LightGBM (sampled)
```

### Classification Code Templates

```python
# Baseline: Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Best for tabular: XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=ratio  # for imbalanced
)

# Fast and robust: LightGBM
from lightgbm import LGBMClassifier
model = LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    class_weight='balanced'
)

# Ensemble for best results
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier([
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier())
], voting='soft')
```

---

## Regression

### Continuous Value Prediction

| Algorithm | Best When | Avoid When | Handles Non-linearity |
|-----------|-----------|------------|----------------------|
| **Linear Regression** | Linear relationships, baseline, interpretability | Non-linear data, outliers | No |
| **Ridge/Lasso** | Multicollinearity, feature selection (Lasso) | Non-linear relationships | No |
| **Elastic Net** | Many features, some correlated | Non-linear data | No |
| **Decision Tree** | Non-linear, interpretable rules | Need smooth predictions | Yes |
| **Random Forest** | Robust, non-linear, feature importance | Extrapolation needed | Yes |
| **XGBoost/LightGBM** | Best accuracy, tabular data | Simple linear problems | Yes |
| **SVR** | Non-linear, small-medium data | Large datasets | Yes |
| **Neural Network** | Complex patterns, large data | Small data, interpretability | Yes |

### Special Regression Cases

| Problem | Algorithm |
|---------|-----------|
| **Count data** (0, 1, 2...) | Poisson Regression, Negative Binomial |
| **Bounded output** (0-1) | Beta Regression, Logistic (scaled) |
| **Censored data** | Survival models (Cox, AFT) |
| **Quantile prediction** | Quantile Regression, Gradient Boosting |
| **Robust to outliers** | Huber Regression, RANSAC |

### Regression Code Templates

```python
# Baseline: Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# With regularization: Ridge
from sklearn.linear_model import Ridge, Lasso, ElasticNet
model = Ridge(alpha=1.0)  # L2
model = Lasso(alpha=0.1)  # L1 (feature selection)
model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Both

# Best for tabular: XGBoost
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror'
)

# Robust to outliers
from sklearn.linear_model import HuberRegressor
model = HuberRegressor(epsilon=1.35)

# Quantile prediction
from lightgbm import LGBMRegressor
model = LGBMRegressor(objective='quantile', alpha=0.5)  # median
```

---

## Clustering

### Unsupervised Grouping

| Algorithm | Best When | Cluster Shape | Scalability |
|-----------|-----------|---------------|-------------|
| **K-Means** | Spherical clusters, known K, large data | Spherical | Excellent |
| **K-Means++** | Better initialization than K-Means | Spherical | Excellent |
| **Mini-Batch K-Means** | Very large datasets | Spherical | Excellent |
| **DBSCAN** | Unknown K, arbitrary shapes, outliers | Arbitrary | Good |
| **HDBSCAN** | Varying density clusters | Arbitrary | Good |
| **Hierarchical** | Need dendrogram, small data | Any | Poor |
| **Gaussian Mixture** | Soft assignments, elliptical clusters | Elliptical | Good |
| **Spectral** | Non-convex clusters, graph data | Arbitrary | Poor |
| **Mean Shift** | Unknown K, find modes | Arbitrary | Poor |

### Choosing Number of Clusters

| Method | Use Case |
|--------|----------|
| **Elbow Method** | K-Means, quick heuristic |
| **Silhouette Score** | Any algorithm, cluster quality |
| **Gap Statistic** | Statistical approach |
| **Dendrogram** | Hierarchical clustering |
| **Domain Knowledge** | Business requirements |

### Clustering Code Templates

```python
# K-Means (most common)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
labels = model.fit_predict(X)

# DBSCAN (no need to specify K)
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

# HDBSCAN (robust DBSCAN)
import hdbscan
model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels = model.fit_predict(X)

# Gaussian Mixture (soft clustering)
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=5, covariance_type='full')
labels = model.fit_predict(X)
probs = model.predict_proba(X)  # soft assignments

# Finding optimal K
from sklearn.metrics import silhouette_score
scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    scores.append(silhouette_score(X, labels))
```

---

## Dimensionality Reduction

### Feature Reduction Techniques

| Algorithm | Best When | Preserves | Linear |
|-----------|-----------|-----------|--------|
| **PCA** | Variance matters, visualization, preprocessing | Global variance | Yes |
| **t-SNE** | 2D/3D visualization, cluster exploration | Local structure | No |
| **UMAP** | Visualization, faster than t-SNE, preserves global | Local + global | No |
| **LDA** | Supervised, maximize class separation | Class separability | Yes |
| **Truncated SVD** | Sparse data, text (TF-IDF) | Variance | Yes |
| **ICA** | Find independent components | Independence | Yes |
| **Autoencoders** | Non-linear, reconstruction | Learned features | No |
| **Factor Analysis** | Latent factors, covariance | Latent structure | Yes |

### When to Use What

```
Goal                          Algorithm
────────────────────────────────────────────────
Preprocessing for ML      →   PCA (keep 95% variance)
Visualization (explore)   →   t-SNE or UMAP
Visualization (preserve)  →   UMAP
Text data                 →   Truncated SVD / LSA
Supervised reduction      →   LDA
Feature extraction        →   Autoencoders
```

### Dimensionality Reduction Code

```python
# PCA (linear, fast)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X)
print(f"Components: {pca.n_components_}")

# t-SNE (visualization)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

# UMAP (faster, preserves structure)
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_2d = reducer.fit_transform(X)

# For sparse data (text)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X_sparse)

# LDA (supervised)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

---

## Time Series

### Forecasting Algorithms

| Algorithm | Best When | Handles | Complexity |
|-----------|-----------|---------|------------|
| **ARIMA** | Univariate, stationary after differencing | Trend, seasonality | Medium |
| **SARIMA** | Seasonal patterns | Trend, seasonality | Medium |
| **Exponential Smoothing** | Simple trends, fast | Trend, seasonality | Low |
| **Prophet** | Multiple seasonality, holidays, missing data | Trend, multiple seasonality | Low |
| **LSTM** | Complex patterns, multivariate, long sequences | Any pattern | High |
| **Transformer** | Very long sequences, attention needed | Any pattern | High |
| **XGBoost** | Feature-rich, tabular with time features | Any (with engineering) | Medium |
| **VAR** | Multiple related time series | Cross-correlations | Medium |

### Time Series Selection Guide

```
Data Characteristics              Recommended
──────────────────────────────────────────────────────
Simple trend                  →   Exponential Smoothing
Trend + seasonality           →   SARIMA, Prophet
Multiple seasonalities        →   Prophet, TBATS
Multiple related series       →   VAR, LSTM
Lots of external features     →   XGBoost, LightGBM
Very long sequences           →   Transformer
Real-time streaming           →   Online learning variants
```

### Time Series Code Templates

```python
# Prophet (easy, robust)
from prophet import Prophet
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(df[['ds', 'y']])
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(series, order=(p, d, q))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

# SARIMA (with seasonality)
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit()

# XGBoost with time features
def create_time_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['lag_1'] = df['value'].shift(1)
    df['lag_7'] = df['value'].shift(7)
    df['rolling_mean_7'] = df['value'].rolling(7).mean()
    return df

from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
```

---

## Anomaly Detection

### Outlier Detection Algorithms

| Algorithm | Best When | Assumes | Scalability |
|-----------|-----------|---------|-------------|
| **Isolation Forest** | General purpose, high-dim | Anomalies are few and different | Excellent |
| **One-Class SVM** | Known normal class, small data | Normal data is compact | Poor |
| **LOF** | Local density variations | Anomalies in sparse regions | Moderate |
| **DBSCAN** | Cluster-based outliers | Anomalies don't cluster | Good |
| **Autoencoder** | Complex patterns, large data | Normal data is reconstructible | Good |
| **Statistical** | Known distribution | Data follows distribution | Excellent |
| **Elliptic Envelope** | Gaussian data | Data is Gaussian | Good |

### Anomaly Detection by Domain

| Domain | Recommended Algorithms |
|--------|----------------------|
| **Network intrusion** | Isolation Forest, Autoencoder |
| **Fraud detection** | Isolation Forest, XGBoost (supervised) |
| **Manufacturing defects** | One-Class SVM, Autoencoder |
| **Time series anomalies** | Prophet, LSTM Autoencoder |
| **Medical diagnosis** | One-Class SVM, Isolation Forest |

### Anomaly Detection Code

```python
# Isolation Forest (most versatile)
from sklearn.ensemble import IsolationForest
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # expected anomaly ratio
    random_state=42
)
predictions = model.fit_predict(X)  # -1 = anomaly, 1 = normal
scores = model.decision_function(X)  # lower = more anomalous

# Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor
model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
predictions = model.fit_predict(X)

# One-Class SVM
from sklearn.svm import OneClassSVM
model = OneClassSVM(kernel='rbf', nu=0.05)
model.fit(X_normal)
predictions = model.predict(X_test)

# Statistical (Z-score)
from scipy import stats
z_scores = stats.zscore(X)
anomalies = (abs(z_scores) > 3).any(axis=1)

# Autoencoder-based
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = X.shape[1]
encoding_dim = 8

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_normal, X_normal, epochs=50, batch_size=32)

# Anomaly score = reconstruction error
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold
```

---

## Natural Language Processing

### NLP Task to Algorithm Mapping

| Task | Traditional | Deep Learning |
|------|-------------|---------------|
| **Text Classification** | Naive Bayes, SVM + TF-IDF | BERT, RoBERTa |
| **Sentiment Analysis** | Logistic Regression + TF-IDF | BERT, DistilBERT |
| **Named Entity Recognition** | CRF, BiLSTM-CRF | BERT-NER, spaCy |
| **Topic Modeling** | LDA, NMF | BERTopic |
| **Text Similarity** | TF-IDF + Cosine | Sentence-BERT |
| **Question Answering** | - | BERT, T5, GPT |
| **Text Generation** | - | GPT, T5, LLaMA |
| **Summarization** | - | BART, T5 |
| **Translation** | - | MarianMT, mBART |

### NLP Algorithm Selection

```
Data Size / Compute    Recommended Approach
─────────────────────────────────────────────────────
Small + Limited       →  TF-IDF + Traditional ML
Medium + GPU          →  DistilBERT (fast), Sentence-BERT
Large + GPU           →  BERT, RoBERTa
Very Large + GPU      →  Fine-tuned LLM
```

### NLP Code Templates

```python
# Traditional: TF-IDF + Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)

# Topic Modeling: LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)

# Modern: Sentence Transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

# Similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings)

# Classification with Transformers
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this product!")

# Zero-shot Classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a tutorial about ML algorithms",
    candidate_labels=["education", "politics", "business"]
)
```

---

## Recommendation Systems

### Recommendation Algorithms

| Algorithm | Best When | Cold Start | Scalability |
|-----------|-----------|------------|-------------|
| **Collaborative Filtering (User-based)** | Dense interactions | Poor | Poor |
| **Collaborative Filtering (Item-based)** | Stable item catalog | Poor | Good |
| **Matrix Factorization (SVD, ALS)** | Sparse data, latent factors | Poor | Good |
| **Content-Based** | Rich item features | Handles | Good |
| **Hybrid** | Best of both worlds | Handles | Moderate |
| **Deep Learning (NCF, DLRM)** | Large data, complex patterns | Moderate | Good |
| **Graph Neural Networks** | Social/network data | Moderate | Moderate |

### Recommendation Selection Guide

```
Scenario                              Recommended
───────────────────────────────────────────────────────
New platform, few users           →   Content-Based
Lots of interactions              →   Matrix Factorization (ALS)
Need explanations                 →   Content-Based, Item-Based CF
Real-time personalization         →   Pre-computed + Online updates
Social network data               →   Graph-based
E-commerce (many items)           →   Two-stage: Retrieval + Ranking
```

### Recommendation Code Templates

```python
# Collaborative Filtering with Surprise
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
cross_validate(model, data, cv=5)

trainset = data.build_full_trainset()
model.fit(trainset)
prediction = model.predict(user_id, item_id)

# Content-Based with Embeddings
from sklearn.metrics.pairwise import cosine_similarity

# Item features → embeddings
item_embeddings = model.encode(item_descriptions)

# Find similar items
def get_similar_items(item_idx, top_k=10):
    similarities = cosine_similarity([item_embeddings[item_idx]], item_embeddings)[0]
    similar_indices = similarities.argsort()[-top_k-1:-1][::-1]
    return similar_indices

# Implicit Feedback with ALS
import implicit
from scipy.sparse import csr_matrix

# Create sparse user-item matrix
sparse_matrix = csr_matrix((ratings, (user_ids, item_ids)))

model = implicit.als.AlternatingLeastSquares(factors=50, iterations=20)
model.fit(sparse_matrix)

# Get recommendations for user
recommendations = model.recommend(user_id, sparse_matrix[user_id], N=10)
```

---

## Deep Learning

### When to Use Deep Learning

| Use Deep Learning | Use Traditional ML |
|-------------------|-------------------|
| Large dataset (>100K samples) | Small dataset |
| Unstructured data (images, text, audio) | Structured/tabular data |
| Complex patterns | Linear/simple relationships |
| GPU available | Limited compute |
| State-of-the-art needed | Interpretability needed |
| Transfer learning possible | Domain-specific features work |

### Architecture Selection

| Data Type | Architecture | Examples |
|-----------|--------------|----------|
| **Tabular** | MLP, TabNet, FT-Transformer | Structured business data |
| **Images** | CNN (ResNet, EfficientNet) | Classification, detection |
| **Text** | Transformers (BERT, GPT) | NLP tasks |
| **Sequences** | LSTM, GRU, Transformer | Time series, logs |
| **Graphs** | GNN (GCN, GAT) | Social networks, molecules |
| **Audio** | CNN, Wav2Vec | Speech, music |

### Pre-trained Models by Task

```
Task                    Model                   Library
─────────────────────────────────────────────────────────────
Image Classification    ResNet, EfficientNet    torchvision, timm
Object Detection        YOLO, Faster-RCNN       ultralytics, detectron2
Image Segmentation      U-Net, Mask R-CNN       segmentation_models
Text Classification     BERT, RoBERTa           transformers
Text Generation         GPT, LLaMA              transformers, ollama
Question Answering      BERT, T5                transformers
Speech Recognition      Whisper, Wav2Vec        transformers
Embeddings             CLIP, Sentence-BERT      sentence-transformers
```

### Deep Learning Code Templates

```python
# Tabular: Simple MLP
import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Image: Transfer Learning
from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=True)
# Freeze base
for param in model.parameters():
    param.requires_grad = False
# Replace head
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Text: Fine-tune BERT
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_classes
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
```

---

## Algorithm Comparison Summary

### By Problem Type

| Problem | First Try | Best Performance | Interpretable |
|---------|-----------|------------------|---------------|
| Binary Classification | Logistic Regression | XGBoost | Decision Tree |
| Multi-class | Random Forest | XGBoost | Decision Tree |
| Regression | Linear Regression | XGBoost | Linear/Ridge |
| Clustering | K-Means | HDBSCAN | K-Means |
| Anomaly Detection | Isolation Forest | Autoencoder | Statistical |
| Time Series | Prophet | LSTM | ARIMA |
| Text Classification | TF-IDF + LR | BERT | Naive Bayes |
| Recommendations | ALS | Neural CF | Item-Based CF |

### By Data Characteristics

| Characteristic | Recommended |
|----------------|-------------|
| Small data (<1K) | Simple models, regularization |
| Large data (>100K) | Gradient boosting, neural networks |
| High dimensional | PCA preprocessing, regularized models |
| Imbalanced classes | SMOTE, class weights, focal loss |
| Missing values | XGBoost, imputation + any |
| Noisy labels | Robust loss, label smoothing |
| Sparse features | Linear models, embeddings |

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    ALGORITHM QUICK PICK                      │
├─────────────────────────────────────────────────────────────┤
│ CLASSIFICATION                                               │
│   Baseline      → Logistic Regression                        │
│   Best Tabular  → XGBoost / LightGBM                        │
│   Interpretable → Decision Tree                              │
│   Text          → BERT / TF-IDF + LR                        │
├─────────────────────────────────────────────────────────────┤
│ REGRESSION                                                   │
│   Baseline      → Linear Regression                          │
│   Best Tabular  → XGBoost / LightGBM                        │
│   Regularized   → Ridge / Lasso                              │
├─────────────────────────────────────────────────────────────┤
│ CLUSTERING                                                   │
│   Known K       → K-Means                                    │
│   Unknown K     → DBSCAN / HDBSCAN                          │
│   Soft clusters → Gaussian Mixture                           │
├─────────────────────────────────────────────────────────────┤
│ ANOMALY                                                      │
│   General       → Isolation Forest                           │
│   One class     → One-Class SVM                              │
│   Complex       → Autoencoder                                │
├─────────────────────────────────────────────────────────────┤
│ TIME SERIES                                                  │
│   Quick & Easy  → Prophet                                    │
│   Statistical   → ARIMA / SARIMA                            │
│   Complex       → LSTM / Transformer                         │
├─────────────────────────────────────────────────────────────┤
│ DIM REDUCTION                                                │
│   Preprocessing → PCA                                        │
│   Visualization → UMAP / t-SNE                              │
│   Text          → Truncated SVD                              │
└─────────────────────────────────────────────────────────────┘
```

---

**Related Docs:**
- [MLOps Workflow](./workflow.md)
- [AI/ML Workbench Setup](./04-aiml-workbench.md)
