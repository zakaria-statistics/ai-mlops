# Lab 01 — Exploratory Data Analysis
> Explore the Titanic dataset — distributions, correlations, outliers, hypotheses

**Prerequisites:** Read `theory/01-data-and-eda/GUIDE.md`
**Dataset:** Titanic (from seaborn: `sns.load_dataset('titanic')` or sklearn)
**Libraries:** pandas, matplotlib, seaborn, numpy

---

## Steps

### Step 1: Load and Inspect
```
- Load dataset, check shape, dtypes
- df.head(), df.info(), df.describe()
- Identify: which columns are numerical? categorical? what's the target?
```
**Expected:** Understanding of 891 rows, ~12 columns, mix of types

### Step 2: Missing Value Analysis
```
- Count missing per column (df.isnull().sum())
- Visualize: seaborn heatmap of missing values
- Which columns have significant missing data?
```
**Expected:** Age ~20% missing, Cabin ~77% missing, Embarked ~0.2%

### Step 3: Distributions
```
- Histogram for every numerical feature (age, fare, sibsp, parch)
- Countplot for every categorical feature (sex, pclass, embarked, survived)
- Note: which are skewed? which are balanced?
```
**Expected:** Fare is right-skewed, Age roughly normal, Pclass imbalanced

### Step 4: Correlation
```
- df.corr() for numerical features
- Heatmap with seaborn (annotated)
- Which features correlate most with survival?
```
**Expected:** Fare has positive correlation with survival, Pclass negative

### Checkpoint: Do you see patterns in the data?

### Step 5: Outlier Detection
```
- Boxplots for Age and Fare
- Apply IQR rule: identify specific outlier values
- Z-score method: compare results with IQR
```
**Expected:** Fare has extreme outliers (>$500)

### Step 6: Bivariate Analysis
```
- Survival rate by sex (bar chart) → women survived more
- Survival rate by class (bar chart) → 1st class survived more
- Age distribution by survival (overlapping histograms or boxplot)
- Fare by survival status
- Create age bins (child/teen/adult/senior), check survival by bin
```
**Expected:** Clear patterns: female > male, 1st > 2nd > 3rd, children survived more

### Step 7: Write Hypotheses
```
From your EDA, write 5 testable hypotheses, e.g.:
1. "Women had significantly higher survival rates than men"
2. "First class passengers survived at higher rates"
3. ...
```

---

## Checkpoint

- [ ] Know the shape, types, and missing values
- [ ] Identified skewed features and outliers
- [ ] Found which features correlate with survival
- [ ] Have 5 hypotheses ready to test with models
