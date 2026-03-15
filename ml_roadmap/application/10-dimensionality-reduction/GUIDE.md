# Lab 10: Dimensionality Reduction
> Reduce 64 dimensions to 2 — PCA from scratch, t-SNE, and the accuracy-speed tradeoff

## Table of Contents
1. [Setup](#setup) - Dataset and libraries
2. [Visualize Digits](#step-1-visualize-digits) - Understand the 64D space
3. [PCA from Scratch](#step-2-pca-from-scratch) - Eigendecomposition with numpy
4. [PCA with sklearn](#step-3-pca-with-sklearn) - Compare with library
5. [Explained Variance](#step-4-cumulative-explained-variance) - How many components?
6. [2D Visualization](#step-5-visualize-in-2d) - PCA projection colored by digit
7. [t-SNE](#step-6-t-sne) - Non-linear dimensionality reduction
8. [PCA as Preprocessing](#step-7-pca-as-preprocessing) - Accuracy vs speed tradeoff
9. [Feature Selection](#step-8-feature-selection-alternative) - SelectKBest comparison

## Prerequisites
- Read `theory/10-dimensionality-reduction/GUIDE.md` first
- Completed Lab 09 (clustering concepts used in visualization)
- Familiarity with eigenvalues/eigenvectors (covered in theory)

## Dataset
**MNIST Digits (8x8)** — `sklearn.datasets.load_digits()`
- 1797 samples, 64 features (8x8 pixel images), 10 classes (digits 0-9)
- Small enough for fast iteration, large enough to show real effects
- No download needed — built into sklearn

## Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
import time
```

---

## Step 1: Visualize Digits

1. Load the dataset:
   ```python
   digits = load_digits()
   X, y = digits.data, digits.target
   print(X.shape)  # (1797, 64)
   ```
2. Show a grid of 10 sample digits (one per class):
   ```python
   fig, axes = plt.subplots(2, 5, figsize=(10, 4))
   for i, ax in enumerate(axes.flat):
       idx = np.where(y == i)[0][0]
       ax.imshow(X[idx].reshape(8, 8), cmap='gray')
       ax.set_title(f'Digit: {i}')
       ax.axis('off')
   ```
3. Think about: each image is a point in 64-dimensional space. Can you separate 10 classes in 64D?
4. Check class balance: `np.bincount(y)` — roughly balanced?

**Expected output:** Clear 8x8 digit images. Each class has ~180 samples. The 64 features are pixel intensities.

---

## Step 2: PCA from Scratch

Implement PCA step by step:

1. **Center the data** — subtract mean of each feature:
   ```python
   X_centered = X - np.mean(X, axis=0)
   ```

2. **Compute covariance matrix** — 64x64 matrix:
   ```python
   cov_matrix = np.cov(X_centered, rowvar=False)  # or (X_centered.T @ X_centered) / (n-1)
   ```

3. **Eigendecomposition** — find eigenvalues and eigenvectors:
   ```python
   eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
   ```

4. **Sort by eigenvalue** — largest first:
   ```python
   sorted_idx = np.argsort(eigenvalues)[::-1]
   eigenvalues = eigenvalues[sorted_idx]
   eigenvectors = eigenvectors[:, sorted_idx]
   ```

5. **Project onto top K components** — matrix multiply:
   ```python
   K = 2
   W = eigenvectors[:, :K]  # 64 x K
   X_pca_scratch = X_centered @ W  # n x K
   ```

6. **Compute explained variance ratio**:
   ```python
   explained_var = eigenvalues / np.sum(eigenvalues)
   print(f"Top 2 components explain: {sum(explained_var[:2]):.2%}")
   ```

**Expected output:** Top 2 components explain roughly 28-30% of total variance. X_pca_scratch shape is (1797, 2).

> **Checkpoint 1:** Your scratch PCA should produce a (1797, 2) matrix. The eigenvalues should be positive and decreasing.

---

## Step 3: PCA with sklearn

1. Fit sklearn PCA:
   ```python
   pca = PCA(n_components=2)
   X_pca_sklearn = pca.fit_transform(X)
   ```
2. Compare with scratch:
   - `pca.explained_variance_ratio_` vs your computed ratios — should match
   - Scatter plot both projections side-by-side — should look identical (or mirrored, since eigenvector sign is arbitrary)
3. Note: if signs are flipped, multiply a component by -1 to match

**Expected output:** Values should be nearly identical. Any sign differences are expected — PCA directions are unique up to a sign flip.

---

## Step 4: Cumulative Explained Variance

1. Fit PCA with all components:
   ```python
   pca_full = PCA().fit(X)
   cumvar = np.cumsum(pca_full.explained_variance_ratio_)
   ```
2. Plot cumulative explained variance vs number of components
3. Draw a horizontal line at 95%
4. Find the number of components for 95%:
   ```python
   n_95 = np.argmax(cumvar >= 0.95) + 1
   print(f"Components for 95% variance: {n_95}")
   ```
5. Also check: how many for 90%? 99%?

**Expected output:** ~28-30 components for 95% variance. That's a reduction from 64 to ~30 features — about 50% compression while keeping 95% of the information.

> **Checkpoint 2:** You should see that a large portion of variance is captured by relatively few components. The curve rises steeply then flattens.

---

## Step 5: Visualize in 2D

1. Scatter plot of first 2 PCs, color by digit label:
   ```python
   plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
   plt.colorbar(label='Digit')
   plt.xlabel('PC1')
   plt.ylabel('PC2')
   ```
2. Which digits cluster together? Which overlap?
3. Are 0s and 1s well-separated? What about 4s and 9s?

**Expected output:** Some digits (0, 6) form tight clusters. Others (3, 5, 8) overlap significantly. Two components aren't enough for perfect separation — expected since they only capture ~30% of variance.

---

## Step 6: t-SNE

1. Apply t-SNE:
   ```python
   tsne = TSNE(n_components=2, random_state=42, perplexity=30)
   X_tsne = tsne.fit_transform(X)
   ```
2. Scatter plot colored by digit — compare with PCA 2D plot
3. Try different perplexity values (5, 30, 50, 100) — how does it change?
4. Side-by-side: PCA 2D vs t-SNE 2D

**Key differences to note:**
- t-SNE creates tighter, more separated clusters
- t-SNE is non-linear — it can unfold curved manifolds
- t-SNE distances between clusters are NOT meaningful (only within-cluster structure is)
- t-SNE is slow and cannot transform new data (no `.transform()` method)

**Expected output:** t-SNE should show 10 clearly separated clusters, much better than PCA 2D. Each digit class should form its own blob.

> **Checkpoint 3:** t-SNE visualization should clearly separate all 10 digit classes. PCA 2D should show some overlap.

---

## Step 7: PCA as Preprocessing

Test whether PCA reduction helps or hurts classification accuracy and speed.

1. Train/test split (80/20, stratified)
2. Run KNN (k=5) and SVM (rbf kernel) on:
   - Full 64 features
   - PCA with 10 components
   - PCA with 20 components
   - PCA with 30 components
3. For each, record accuracy AND training time:
   ```python
   start = time.time()
   model.fit(X_train_pca, y_train)
   train_time = time.time() - start
   acc = accuracy_score(y_test, model.predict(X_test_pca))
   ```
4. Create a results table:

```
| Features | KNN Accuracy | KNN Time | SVM Accuracy | SVM Time |
|----------|-------------|----------|-------------|----------|
| 64 (all) |             |          |             |          |
| 30 (PCA) |             |          |             |          |
| 20 (PCA) |             |          |             |          |
| 10 (PCA) |             |          |             |          |
```

**Expected output:** Accuracy stays nearly the same (or even improves) with 30 components vs 64. Training time decreases. At 10 components, accuracy may start to drop. The sweet spot is around 20-30 components — nearly full accuracy at half the features.

---

## Step 8: Feature Selection Alternative

Compare PCA (transformation) with feature selection (choosing original features).

1. Use `SelectKBest` with `mutual_info_classif`:
   ```python
   selector = SelectKBest(mutual_info_classif, k=20)
   X_selected = selector.fit_transform(X, y)
   ```
2. Which 20 pixels were selected? Visualize on an 8x8 grid:
   ```python
   mask = selector.get_support()
   plt.imshow(mask.reshape(8, 8), cmap='gray')
   ```
3. Run KNN on the 20 selected features — compare accuracy with PCA at 20 components
4. Key difference: selected features are interpretable (actual pixels), PCA components are not

**Expected output:** Selected pixels should be in the center of the 8x8 grid (edges are always dark/uninformative). Accuracy should be comparable to PCA-20 but slightly lower — PCA captures more info per component because it creates new combined features.

> **Checkpoint 4:** You understand PCA (transforms features) vs feature selection (picks features). PCA gives better compression; selection gives interpretability.

---

## Summary Deliverables
- [ ] PCA from scratch matching sklearn output
- [ ] Cumulative variance plot with 95% threshold marked
- [ ] PCA 2D vs t-SNE 2D comparison
- [ ] Accuracy vs speed tradeoff table (PCA preprocessing)
- [ ] Feature selection vs PCA comparison
- [ ] Understanding of when to use PCA vs t-SNE vs feature selection
