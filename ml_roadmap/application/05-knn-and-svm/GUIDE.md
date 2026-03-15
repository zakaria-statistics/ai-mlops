# Lab 05 — KNN and SVM
> Implement KNN from scratch, explore SVM kernels, visualize decision boundaries

**Prerequisites:** Read `theory/05-knn-and-svm/GUIDE.md`
**Dataset:** Breast Cancer Wisconsin (`from sklearn.datasets import load_breast_cancer`)
**Libraries:** numpy, matplotlib, sklearn

---

## Steps

### Step 1: FROM SCRATCH — Euclidean Distance
```python
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

### Step 2: FROM SCRATCH — KNN Classifier
```python
def knn_predict(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    k_nearest = np.argsort(distances)[:k]
    k_labels = y_train[k_nearest]
    return np.bincount(k_labels).argmax()  # majority vote
```

### Step 3: Test Different K Values
```
- K = 1, 3, 5, 7, 9, 15, 25
- Plot: accuracy vs K
- K=1 overfits (memorizes), large K underfits (too smooth)
```
**Expected:** Best K around 5-9

### Step 4: sklearn KNeighborsClassifier
```
- Compare accuracy with your scratch implementation
```

### Checkpoint: Scratch KNN ≈ sklearn KNN

### Step 5: Show WHY Scaling Matters
```
- Run KNN WITHOUT scaling → note accuracy
- Run KNN WITH StandardScaler → note accuracy
- The unscaled version is much worse because features with large ranges dominate distance
```

### Step 6: SVM with sklearn
```
- LinearSVC → linear decision boundary
- SVC(kernel='rbf') → non-linear boundary
- SVC(kernel='poly', degree=3)
```

### Step 7: Decision Boundary Visualization
```
- Reduce to 2D with PCA (2 components)
- Plot decision boundaries for KNN and SVM (each kernel)
- Use meshgrid + contourf
```

### Step 8: Compare SVM Kernels
```
| Kernel  | Accuracy | Precision | Recall | F1   | Time |
|---------|----------|-----------|--------|------|------|
| Linear  |          |           |        |      |      |
| RBF     |          |           |        |      |      |
| Poly(3) |          |           |        |      |      |
```

### Step 9: Grid Search for SVM
```
- GridSearchCV over: C=[0.1, 1, 10, 100], gamma=['scale', 'auto', 0.01, 0.1]
- Best params? Best score?
```

### Step 10: Grand Comparison
```
| Model        | Accuracy | Precision | Recall | F1   | Train Time |
|--------------|----------|-----------|--------|------|------------|
| KNN (best K) |          |           |        |      |            |
| SVM Linear   |          |           |        |      |            |
| SVM RBF      |          |           |        |      |            |
```

---

## Checkpoint

- [ ] KNN from scratch works and matches sklearn
- [ ] Understand why scaling is critical for distance-based methods
- [ ] Can visualize and compare decision boundaries
- [ ] Can tune SVM hyperparameters with GridSearch
