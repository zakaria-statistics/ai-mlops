# Lab 07 — Ensemble Methods
> Bagging, boosting, stacking — from scratch bagging to XGBoost comparison

**Prerequisites:** Read `theory/07-ensemble-methods/GUIDE.md`
**Dataset:** Bank Marketing UCI (binary: did client subscribe to term deposit?)
**Libraries:** numpy, matplotlib, sklearn, xgboost

---

## Steps

### Step 1: Baseline — Single Decision Tree
```
- Load and prepare Bank Marketing data
- Handle class imbalance (note: ~88% "no", ~12% "yes")
- Fit DecisionTreeClassifier → note accuracy, F1, ROC-AUC
```

### Step 2: FROM SCRATCH — Bagging
```python
def bagging_predict(X_train, y_train, X_test, n_trees=10):
    predictions = []
    for i in range(n_trees):
        # Bootstrap sample
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot, y_boot = X_train[indices], y_train[indices]
        # Train tree
        tree = DecisionTreeClassifier()
        tree.fit(X_boot, y_boot)
        predictions.append(tree.predict(X_test))
    # Majority vote
    return np.round(np.mean(predictions, axis=0)).astype(int)
```

### Step 3: sklearn RandomForestClassifier
```
- Compare with scratch bagging (should be similar or better due to feature randomization)
- n_estimators=200
```

### Step 4: Feature Importance from Random Forest
```
- Plot top 15 features by importance
- Which features drive subscription decisions?
```

### Checkpoint: Random Forest > single tree

### Step 5: AdaBoostClassifier
```
- Fit with n_estimators=100
- Compare with Random Forest
```

### Step 6: GradientBoostingClassifier
```
- Fit with n_estimators=200, learning_rate=0.1
- Compare: sequential boosting vs parallel bagging
```

### Step 7: XGBoost
```
- import xgboost as xgb
- XGBClassifier(scale_pos_weight=ratio for imbalance)
- Compare with all above
```

### Step 8: VotingClassifier
```
- Combine: LogisticRegression + RandomForest + SVC
- Hard voting (majority vote) and soft voting (average probabilities)
```

### Step 9: StackingClassifier
```
- Base: RandomForest + XGBoost + SVM
- Meta-learner: LogisticRegression
- Does stacking beat the best individual model?
```

### Step 10: Grand Comparison
```
| Model               | Accuracy | F1    | ROC-AUC | Train Time |
|---------------------|----------|-------|---------|------------|
| Single Tree         |          |       |         |            |
| Bagging (scratch)   |          |       |         |            |
| Random Forest       |          |       |         |            |
| AdaBoost            |          |       |         |            |
| Gradient Boosting   |          |       |         |            |
| XGBoost             |          |       |         |            |
| Voting (soft)       |          |       |         |            |
| Stacking            |          |       |         |            |
```

---

## Checkpoint

- [ ] Bagging from scratch produces similar results to sklearn
- [ ] Understand: bagging (parallel, reduces variance) vs boosting (sequential, reduces bias)
- [ ] XGBoost typically wins on tabular data
- [ ] Stacking/voting can squeeze out extra performance
