# Phase 3: Data Exploration (EDA)

> Understand the data before touching any model. What does it look like? What's broken? What drives price?

## Table of Contents

1. [Goal](#goal)
2. [How to Run](#how-to-run)
3. [EDA Checklist](#eda-checklist)
4. [Key Questions to Answer](#key-questions-to-answer)
5. [What to Look For](#what-to-look-for)

---

## Goal

Before building any model, you need to **know your data**. EDA answers:
- What shape is it? (rows, columns, types)
- Is anything missing or broken?
- What does the target (price) look like?
- Which features seem to drive price?
- Are there outliers that could hurt the model?

## How to Run

```bash
cd /root/claude/ai_mlops
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib seaborn numpy jupyter jupytext
```

### Generate the notebook

The source of truth is `notebooks/01-eda-house-prices.py` (percent format, git-friendly).
To create a runnable `.ipynb` paired with it:

```bash
jupytext --set-formats py:percent,ipynb notebooks/01-eda-house-prices.py
```

This creates `notebooks/01-eda-house-prices.ipynb` and links both files — saving either one syncs the other.

### Launch Jupyter

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Open in browser: `http://<proxmox-ip>:8888/?token=...`

Navigate to `notebooks/01-eda-house-prices.ipynb` → run cells with **Shift+Enter**.

### Git Strategy

Track only the `.py` file — add `*.ipynb` to `.gitignore`. Clean diffs, no JSON bloat.

### Notebook Sections

| # | Section | What You'll See |
|---|---------|----------------|
| 1 | Setup & Load | Load dataset, confirm shape |
| 2 | First Look | Data types, sample rows, basic stats |
| 3 | Missing Values | Any gaps in the data? |
| 4 | Price Distribution | Skew, median vs mean, log transform comparison |
| 5 | Correlations | Which features predict price most? |
| 6 | Scatter Plots | Top features vs price visually |
| 7 | Categorical Analysis | Grade, condition, waterfront effects |
| 8 | Outlier Detection | Suspicious values (33 bedrooms, extreme prices) |
| 9 | Geographic Analysis | Price heatmap, best/worst zipcodes |
| 10 | Feature Correlations | Redundant features to drop |
| 11 | Summary | Fill in findings → feeds Phase 4 |

### Output

Plots save automatically to this directory:
- `price_distribution.png`
- `correlations.png`
- `top_features_scatter.png`
- `categorical_vs_price.png`
- `geographic_analysis.png`
- `correlation_heatmap.png`

## EDA Checklist

- [ ] Confirm dataset shape (~21,613 rows × 21 columns)
- [ ] Check for missing values
- [ ] Examine price distribution and skewness
- [ ] Identify top 5 features correlated with price
- [ ] Spot outliers (bedrooms, price extremes)
- [ ] Analyze location effect (zipcode, lat/long)
- [ ] Flag redundant features (high inter-correlation)
- [ ] Fill in summary table in notebook section 11

## Key Questions to Answer

After running the notebook, you should know:

| Question | Expected Finding |
|----------|-----------------|
| How many records? | ~21,613 — enough for training |
| Missing data? | None or minimal |
| Price distribution? | Right-skewed, consider log transform |
| Top predictors? | sqft_living, grade, sqft_above, bathrooms |
| Suspicious values? | 33-bedroom house, $7.7M outliers |
| Location effect? | Zipcode creates 10x+ price difference |
| Features to drop? | id, date; sqft_above redundant with sqft_living |

## What to Look For

### Target Variable (price)

```
Healthy:  Slight right skew, no zeros/negatives
Red flag: Extreme skew, huge outliers ($7M+)
Action:   Log transform likely needed for modeling
```

### Feature-Price Relationships

```
Strong linear:  sqft_living vs price (expect r > 0.7)
Step patterns:  grade vs price (discrete jumps)
Weak/none:      id, date (drop these)
Non-linear:     yr_built vs price (needs feature engineering → "age")
```

### Outliers to Watch

```
Bedrooms:  33-bedroom house (data entry error?)
Price:     $7.7M max vs $540K median — extreme right tail
Sqft:      Very large lots that may be empty land
```

---

## Next Step

→ After completing the checklist, take findings to [04-data-preparation](../04-data-preparation/) for cleaning and preprocessing.
