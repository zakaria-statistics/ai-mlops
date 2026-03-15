# Lab 02 — Data Preparation
> Clean, encode, scale, and split the Titanic data — and demonstrate data leakage

**Prerequisites:** Read `theory/02-data-preparation/GUIDE.md`, complete Lab 01
**Dataset:** Titanic (continue from Lab 01)
**Libraries:** pandas, numpy, sklearn (preprocessing, model_selection)

---

## Steps

### Step 1: Handle Missing Values
```
- Age: impute with MEDIAN (robust to outliers and skew)
- Embarked: impute with MODE (most frequent port)
- Cabin: too much missing → drop, OR extract deck letter as feature
```

### Step 2: Feature Engineering
```
- Extract Title from Name: "Mr.", "Mrs.", "Miss.", "Master.", "Rare"
- family_size = sibsp + parch + 1
- is_alone = 1 if family_size == 1
- age_bin: child (<12), teen (12-18), adult (18-60), senior (60+)
- Drop: Name, Ticket, PassengerId (non-predictive)
```

### Step 3: Encode Categoricals
```
- Sex: LabelEncoder or map {'male':0, 'female':1}
- Embarked: OneHotEncoder (nominal — S, C, Q have no order)
- Pclass: keep as-is (ordinal) or one-hot (debate both)
- Title: OneHotEncoder
```
**Expected:** All features are now numerical

### Step 4: Scale Numerical Features
```
- Apply StandardScaler to: Age, Fare, family_size
- Plot before/after distributions to see the effect
- Important: fit on train ONLY (see step 5)
```

### Checkpoint: All features numerical, no missing values

### Step 5: Train/Test Split
```
- 80/20 split, STRATIFIED on 'survived'
- Verify: class distribution is preserved in both sets
- X_train, X_test, y_train, y_test
```

### Step 6: Demonstrate Data Leakage
```
WRONG way:
  scaler.fit(X_all)  → transform both
  → train accuracy is slightly inflated

RIGHT way:
  scaler.fit(X_train) → transform X_train and X_test separately

Show both approaches, compare test scores → the difference is real
```

### Step 7: Handle Class Imbalance
```
- Check: y_train.value_counts() — is it balanced?
- Titanic is roughly 60/40, not extreme
- Show: class_weight='balanced' option for models
- Mention: SMOTE for more extreme cases
```

---

## Output

Save these for use in labs 03+:
```
X_train, X_test, y_train, y_test  (scaled, encoded, clean)
```

## Checkpoint

- [ ] No missing values remain
- [ ] All features are numerical
- [ ] Scaling done correctly (fit on train only)
- [ ] Stratified split verified
- [ ] Understand data leakage and how to avoid it
