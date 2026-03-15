# Lab 09: Unsupervised Learning
> Cluster customers without labels — K-Means from scratch, DBSCAN, hierarchical clustering

## Table of Contents
1. [Setup](#setup) - Dataset and libraries
2. [EDA](#step-1-eda) - Explore customer segments visually
3. [K-Means from Scratch](#step-2-k-means-from-scratch) - Implement the algorithm with numpy
4. [K-Means with sklearn](#step-3-k-means-with-sklearn) - Compare with library
5. [Elbow Method](#step-4-elbow-method) - Find optimal K
6. [Silhouette Analysis](#step-5-silhouette-analysis) - Validate cluster quality
7. [Visualize Clusters](#step-6-visualize-clusters) - 2D scatter plots
8. [DBSCAN](#step-7-dbscan) - Density-based clustering
9. [K-Means vs DBSCAN](#step-8-compare-k-means-vs-dbscan) - Side-by-side comparison
10. [Hierarchical Clustering](#step-9-hierarchical-clustering) - Dendrogram approach
11. [Anomaly Detection Bonus](#step-10-anomaly-detection-bonus) - Isolation Forest

## Prerequisites
- Read `theory/09-unsupervised-learning/GUIDE.md` first
- Completed Labs 01-02 (EDA and data prep fundamentals)

## Dataset
**Mall Customers** — `Mall_Customers.csv`
- Source: Kaggle (`kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python`)
- 200 rows, 5 columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
- Simple enough to visualize in 2D, clear natural clusters

## Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import dendrogram, linkage
```

---

## Step 1: EDA

Load the dataset and explore it.

1. `df.head()`, `df.info()`, `df.describe()`
2. Check for nulls — there shouldn't be any
3. Scatter plot: Annual Income (x) vs Spending Score (y)
4. Look at the scatter — do you see natural groups by eye? How many?
5. Distribution plots for Age, Annual Income, Spending Score
6. Pairplot colored by Gender — any gender-based separation?

**Expected output:** The income vs spending scatter should show ~5 visible clusters. Gender does not separate cleanly.

---

## Step 2: K-Means from Scratch

Use only `Annual Income` and `Spending Score` as features (2D for easy visualization).

```
Algorithm:
1. Pick K random points as initial centroids
2. Assign each data point to the nearest centroid (Euclidean distance)
3. Recompute centroids as the mean of all assigned points
4. Repeat steps 2-3 until centroids stop moving (or max iterations)
```

Implement these functions:

1. `initialize_centroids(X, K)` — randomly pick K data points as starting centroids
2. `assign_clusters(X, centroids)` — for each point, find nearest centroid, return labels array
3. `update_centroids(X, labels, K)` — compute mean of points in each cluster
4. `compute_inertia(X, labels, centroids)` — sum of squared distances from each point to its centroid
5. `kmeans_scratch(X, K, max_iter=100)` — full loop, track inertia per iteration

Run with K=5. Plot inertia vs iteration to confirm convergence.

**Expected output:** Inertia should drop rapidly in first 5-10 iterations then plateau. Final centroids should be near the center of each visible cluster.

> **Checkpoint 1:** Your scratch K-Means should converge in < 20 iterations and produce 5 distinct clusters.

---

## Step 3: K-Means with sklearn

1. Fit `KMeans(n_clusters=5, random_state=42)` on same 2D data
2. Print `kmeans.cluster_centers_` — compare with your scratch centroids
3. Print `kmeans.inertia_` — compare with your scratch inertia
4. Compare labels: are clusters assigned the same way? (Note: label indices may differ, compare visually)

**Expected output:** sklearn centroids should be very close to your scratch centroids. Inertia should be similar or slightly better (sklearn uses kmeans++ initialization).

---

## Step 4: Elbow Method

1. Run KMeans for K = 2, 3, 4, ..., 10
2. Record inertia for each K
3. Plot K (x-axis) vs Inertia (y-axis)
4. Find the "elbow" — where the curve bends, diminishing returns start

```python
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
```

**Expected output:** Elbow at K=5. Inertia drops sharply from K=2 to K=5, then flattens.

---

## Step 5: Silhouette Analysis

1. Compute silhouette score for each K (2-10):
   ```python
   silhouette_score(X, labels)
   ```
2. Plot K vs silhouette score — higher is better
3. For best K, plot silhouette diagram (per-sample silhouette values grouped by cluster):
   ```python
   silhouette_vals = silhouette_samples(X, labels)
   ```
4. In the silhouette plot, each cluster should have roughly equal width and no values below 0

**Expected output:** K=5 should have the highest (or near-highest) silhouette score. Silhouette plot should show 5 roughly even clusters.

> **Checkpoint 2:** Both elbow and silhouette methods should agree on K=5.

---

## Step 6: Visualize Clusters

1. Scatter plot with K=5 clusters, each colored differently
2. Plot centroids as large markers (star or X)
3. Add legend with cluster labels
4. Title: "Mall Customer Segments"
5. Try to interpret each cluster: "high income, high spending" = premium customers, etc.

**Expected output:** Five clearly separated groups in income-spending space. Name each segment based on its position.

---

## Step 7: DBSCAN

1. **Scale the data first** — DBSCAN is distance-sensitive
2. Fit `DBSCAN(eps=0.5, min_samples=5)` on scaled data
3. Check how many clusters it found: `len(set(labels)) - (1 if -1 in labels else 0)`
4. Check noise points: count labels == -1
5. Experiment with `eps` values (0.3, 0.5, 0.8, 1.0) — how does it change results?
6. Scatter plot colored by DBSCAN labels, noise points in black

**Expected output:** DBSCAN will likely find a different number of clusters depending on eps. Some points will be labeled as noise (-1). The clusters may not match K-Means exactly.

---

## Step 8: Compare K-Means vs DBSCAN

1. Side-by-side scatter plots: K-Means clusters (left) vs DBSCAN clusters (right)
2. Which algorithm handles the edges/boundaries better?
3. Add some artificial outlier points (e.g., income=200, spending=1) and rerun both
4. K-Means will force outliers into a cluster. DBSCAN will label them as noise.
5. Create a comparison table:

```
| Feature            | K-Means           | DBSCAN             |
|--------------------|--------------------|---------------------|
| Needs K specified  | Yes                | No                  |
| Handles noise      | No (forces assign) | Yes (labels as -1)  |
| Cluster shapes     | Spherical          | Arbitrary           |
| Speed              | Fast               | Varies              |
```

**Expected output:** K-Means gives cleaner clusters on this data. DBSCAN is better at identifying noise/outliers.

> **Checkpoint 3:** You should understand when to use K-Means vs DBSCAN based on data shape and noise.

---

## Step 9: Hierarchical Clustering

1. Compute linkage matrix:
   ```python
   Z = linkage(X_scaled, method='ward')
   ```
2. Plot dendrogram — set `color_threshold` to cut at a level that gives ~5 clusters
3. Fit `AgglomerativeClustering(n_clusters=5)` on scaled data
4. Scatter plot with hierarchical cluster labels
5. Compare with K-Means visually — should be similar for this data

**Expected output:** Dendrogram shows natural merge points. Cutting at the right height gives 5 clusters that closely match K-Means.

---

## Step 10: Anomaly Detection Bonus

1. Fit `IsolationForest(contamination=0.05, random_state=42)` on the original 2D data
2. Predict: +1 = normal, -1 = anomaly
3. Scatter plot: normal points in blue, anomalies in red
4. Which customers are flagged? Check their income/spending values
5. Business interpretation: are these customers worth investigating?

**Expected output:** ~10 points flagged as anomalies, likely at the extreme edges of the scatter plot (very high/low combinations).

> **Checkpoint 4:** You've implemented clustering from scratch, used 3 clustering algorithms, and applied anomaly detection. You understand the tradeoffs between each approach.

---

## Summary Deliverables
- [ ] Scratch K-Means implementation that converges
- [ ] Elbow plot confirming K=5
- [ ] Silhouette analysis confirming K=5
- [ ] K-Means vs DBSCAN comparison
- [ ] Dendrogram + hierarchical clustering
- [ ] Anomaly detection with Isolation Forest
- [ ] Business interpretation of customer segments
