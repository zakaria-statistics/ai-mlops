# Lab 11: Evaluation Deep Dive
> Master model selection — learning curves, cross-validation from scratch, hyperparameter tuning, statistical comparison

## Table of Contents
1. [Setup](#setup) - Datasets and libraries
2. [Learning Curves](#step-1-learning-curves) - Diagnose bias vs variance
3. [Validation Curves](#step-2-validation-curves) - Tune a single hyperparameter
4. [K-Fold from Scratch](#step-3-k-fold-cross-validation-from-scratch) - Implement the loop
5. [Stratified K-Fold](#step-4-stratified-k-fold) - Handle imbalanced data
6. [GridSearchCV](#step-5-gridsearchcv) - Exhaustive hyperparameter search
7. [RandomizedSearchCV](#step-6-randomizedsearchcv) - Faster alternative
8. [Nested CV](#step-7-nested-cross-validation) - Unbiased evaluation
9. [Model Comparison](#step-8-model-comparison-dashboard) - 6 models head-to-head
10. [Statistical Tests](#step-9-statistical-significance) - Are differences real?
11. [Final Selection](#step-10-final-model-selection) - The proper workflow

## Prerequisites
- Read `theory/11-evaluation-deep-dive/GUIDE.md` first
- Completed Labs 03-08 (familiar with multiple model types)
- Labs 01-02 (data handling)

## Datasets
- **Titanic** — classification (reuse from earlier labs or `sns.load_dataset('titanic')`)
- **California Housing** — regression (`sklearn.datasets.fetch_california_housing()`)

## Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    learning_curve, validation_curve, KFold, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
import time
```

---

## Step 1: Learning Curves

Use Titanic data (preprocessed — handle nulls, encode categoricals, scale numerics).

1. Fit a Decision Tree (max_depth=3) and plot learning curve:
   ```python
   train_sizes, train_scores, val_scores = learning_curve(
       DecisionTreeClassifier(max_depth=3), X, y,
       train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
   )
   ```
2. Plot: training set size (x) vs mean train/val score (y), with std shading
3. Diagnose:
   - **High bias (underfitting):** both curves plateau at low score, close together
   - **High variance (overfitting):** train score high, val score much lower, gap persists
4. Repeat with an unrestricted Decision Tree (no max_depth limit) — expect high variance
5. Compare the two plots side by side

**Expected output:** Shallow tree (max_depth=3) shows slight underfitting. Unrestricted tree shows clear overfitting — train accuracy ~100%, val accuracy much lower.

---

## Step 2: Validation Curves

1. Plot validation curve for Decision Tree varying `max_depth` from 1 to 20:
   ```python
   param_range = range(1, 21)
   train_scores, val_scores = validation_curve(
       DecisionTreeClassifier(), X, y,
       param_name='max_depth', param_range=param_range,
       cv=5, scoring='accuracy'
   )
   ```
2. Plot: max_depth (x) vs train/val accuracy (y)
3. Find the sweet spot: where val score peaks before dropping
4. Train score keeps climbing — that's memorization, not learning

**Expected output:** Val accuracy peaks around max_depth 3-6, then stays flat or drops. Train accuracy approaches 100% as depth increases. The gap is the overfitting zone.

> **Checkpoint 1:** You can diagnose underfitting and overfitting using learning and validation curves.

---

## Step 3: K-Fold Cross-Validation from Scratch

Implement K-Fold without sklearn:

1. Shuffle indices, split into K equal-ish folds:
   ```python
   def kfold_split(n_samples, k, shuffle=True, random_state=42):
       indices = np.arange(n_samples)
       if shuffle:
           np.random.seed(random_state)
           np.random.shuffle(indices)
       fold_size = n_samples // k
       folds = []
       for i in range(k):
           start = i * fold_size
           end = start + fold_size if i < k - 1 else n_samples
           val_idx = indices[start:end]
           train_idx = np.concatenate([indices[:start], indices[end:]])
           folds.append((train_idx, val_idx))
       return folds
   ```
2. Loop over folds, train a LogisticRegression, record accuracy:
   ```python
   scores = []
   for train_idx, val_idx in kfold_split(len(X), k=5):
       model = LogisticRegression(max_iter=1000)
       model.fit(X[train_idx], y[train_idx])
       scores.append(accuracy_score(y[val_idx], model.predict(X[val_idx])))
   print(f"Mean: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
   ```
3. Compare with sklearn:
   ```python
   sklearn_scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
   ```
4. Results should be very similar (not identical due to different fold assignments)

**Expected output:** Mean accuracy around 0.78-0.82 for Titanic. Std around 0.02-0.04. Your scratch version should be close to sklearn.

---

## Step 4: Stratified K-Fold

1. Check class balance in Titanic: `y.value_counts()` — likely ~60/40 split
2. Show the problem: in regular K-Fold, some folds might have 70% positive, others 50%
3. Use `StratifiedKFold` — preserves class ratio in each fold:
   ```python
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
       print(f"Fold {fold}: train class ratio = {y[train_idx].mean():.3f}, "
             f"val class ratio = {y[val_idx].mean():.3f}")
   ```
4. Compare CV scores: regular KFold vs StratifiedKFold — stratified usually has lower variance

**Expected output:** StratifiedKFold maintains consistent class ratios across all folds (~38% survival in each). Regular KFold may have 5-10% variation between folds.

> **Checkpoint 2:** You understand why stratified CV matters for imbalanced datasets.

---

## Step 5: GridSearchCV

Exhaustive search over a hyperparameter grid for Random Forest:

1. Define the grid:
   ```python
   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [3, 5, 10, None],
       'min_samples_split': [2, 5, 10]
   }
   ```
2. That's 3 x 4 x 3 = 36 combinations, each with 5-fold CV = 180 fits
3. Run GridSearchCV:
   ```python
   grid = GridSearchCV(
       RandomForestClassifier(random_state=42), param_grid,
       cv=5, scoring='accuracy', n_jobs=-1, verbose=1
   )
   grid.fit(X, y)
   ```
4. Print best params and best score:
   ```python
   print(f"Best params: {grid.best_params_}")
   print(f"Best CV score: {grid.best_score_:.4f}")
   ```
5. Record the total time

**Expected output:** Best accuracy around 0.82-0.84. Total time depends on machine but note it for comparison with RandomizedSearchCV.

---

## Step 6: RandomizedSearchCV

Same search space but random sampling:

1. Define distributions:
   ```python
   from scipy.stats import randint
   param_dist = {
       'n_estimators': randint(50, 300),
       'max_depth': [3, 5, 10, 20, None],
       'min_samples_split': randint(2, 20)
   }
   ```
2. Run RandomizedSearchCV with n_iter=20 (20 random combinations instead of 36 exhaustive):
   ```python
   random_search = RandomizedSearchCV(
       RandomForestClassifier(random_state=42), param_dist,
       n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
   )
   random_search.fit(X, y)
   ```
3. Compare: best score vs GridSearch, time vs GridSearch
4. Key insight: RandomizedSearch often finds equally good results in less time, especially with large grids

**Expected output:** Best score within 0.5% of GridSearch, but faster. With larger grids the time savings become dramatic.

> **Checkpoint 3:** You understand the tradeoff between exhaustive and random hyperparameter search.

---

## Step 7: Nested Cross-Validation

Avoid the optimistic bias of reporting GridSearchCV's best score as the true performance.

1. **Inner loop:** hyperparameter tuning (GridSearchCV)
2. **Outer loop:** performance estimation (KFold)

```python
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid,
        cv=inner_cv, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    nested_scores.append(grid.score(X_test, y_test))

print(f"Nested CV: {np.mean(nested_scores):.4f} +/- {np.std(nested_scores):.4f}")
```

3. Compare nested CV score with non-nested GridSearchCV best_score_
4. The nested score is usually slightly lower — that's the honest estimate

**Expected output:** Nested CV gives 0.5-1.5% lower accuracy than non-nested. The non-nested score is optimistically biased because the same data was used for tuning and evaluation.

---

## Step 8: Model Comparison Dashboard

Train 6 models on Titanic, compare with cross-validation:

1. Define models:
   ```python
   models = {
       'LogReg': LogisticRegression(max_iter=1000),
       'KNN': KNeighborsClassifier(),
       'SVM': SVC(),
       'Tree': DecisionTreeClassifier(max_depth=5),
       'RF': RandomForestClassifier(n_estimators=100, random_state=42),
       'GBM': GradientBoostingClassifier(random_state=42)
   }
   ```
2. 5-fold stratified CV for each:
   ```python
   results = {}
   for name, model in models.items():
       scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring='accuracy')
       results[name] = scores
       print(f"{name}: {scores.mean():.4f} +/- {scores.std():.4f}")
   ```
3. Bar chart with error bars showing mean +/- std for each model
4. Box plot of CV fold scores for each model (shows distribution, not just mean)

**Expected output:** Ensemble methods (RF, GBM) should be near the top. LogReg is a competitive baseline. KNN and Tree may trail slightly. All should be in the 0.78-0.84 range.

> **Checkpoint 4:** You have a visual comparison of 6 models with proper CV evaluation.

---

## Step 9: Statistical Significance

Are the differences between models actually meaningful?

1. Take the per-fold scores from Step 8
2. For the top 2 models, run a paired t-test:
   ```python
   t_stat, p_value = ttest_rel(results['RF'], results['GBM'])
   print(f"RF vs GBM: t={t_stat:.3f}, p={p_value:.3f}")
   ```
3. If p < 0.05, the difference is statistically significant
4. With only 5 folds, you have low power — differences need to be large to be significant
5. Try with 10-fold CV to get more samples for the test

**Expected output:** For models within 1% accuracy of each other, the p-value is likely > 0.05 (not significant). This means the difference could be due to random fold variation.

---

## Step 10: Final Model Selection

The proper end-to-end workflow:

1. **Hold out a test set first** (20%), don't touch it:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         stratify=y, random_state=42)
   ```
2. **Select and tune on training set only** — use CV / GridSearchCV on X_train
3. **Pick the best model** based on CV results
4. **Retrain on full training set:**
   ```python
   best_model.fit(X_train, y_train)
   ```
5. **Evaluate on test set ONCE:**
   ```python
   test_acc = accuracy_score(y_test, best_model.predict(X_test))
   print(f"Final test accuracy: {test_acc:.4f}")
   ```
6. This is the number you report. No going back to try other models on the test set.

**Expected output:** Test accuracy should be close to the CV estimate (within 1-2%). If it's much worse, something went wrong (data leakage, overfitting to CV folds).

> **Checkpoint 5:** You understand the complete model evaluation pipeline: hold out test set -> CV for model selection -> tune on train -> evaluate on test ONCE.

---

## Summary Deliverables
- [ ] Learning curve plots diagnosing bias vs variance
- [ ] Validation curve with optimal hyperparameter identified
- [ ] K-Fold CV from scratch matching sklearn
- [ ] GridSearchCV vs RandomizedSearchCV comparison
- [ ] Nested CV demonstrating honest evaluation
- [ ] 6-model comparison dashboard (bar chart + box plot)
- [ ] Statistical test showing whether top models truly differ
- [ ] Final model selected and evaluated on held-out test set
