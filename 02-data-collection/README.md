# Phase 2: Data Collection

> Find a real statistical dataset that represents the problem.

## Table of Contents

1. [Dataset Requirements](#dataset-requirements)
2. [Selected Dataset](#selected-dataset)
3. [Download Instructions](#download-instructions)
4. [Data Dictionary](#data-dictionary)
5. [Initial Sanity Check](#initial-sanity-check)

---

## Dataset Requirements

| Requirement | Why |
|-------------|-----|
| **Residential property sales** | Matches our business problem |
| **Price column** (sale price) | This is our target variable |
| **Property features** (size, rooms, etc.) | Input features for prediction |
| **Location info** (zipcode, coordinates) | Key price driver |
| **1,000+ rows** | Enough to train meaningful models |
| **Public/open source** | Reproducible, no licensing issues |

## Selected Dataset

### King County House Sales

- **What:** ~21,600 house sales in King County (Seattle area), 2014-2015
- **Source:** Kaggle / UCI Machine Learning Repository
- **Features:** 19 features — sqft, bedrooms, bathrooms, floors, waterfront, view, condition, grade, yr_built, zipcode, lat/long
- **Target:** `price` (sale price in USD)
- **Size:** ~2.5 MB
- **Why this one:**
  - Clean enough to learn, messy enough to need preprocessing
  - Rich feature set (numeric, categorical, geographic)
  - Well-documented, widely used — easy to validate your results
  - Supports both buyer and seller views from one model

## Download Instructions

```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d harlfoxem/housesalesprediction -p ../data/raw/
unzip ../data/raw/housesalesprediction.zip -d ../data/raw/

# Option B: Manual download
# https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
# Place in: data/raw/kc_house_data.csv
```

## Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique ID (drop this) |
| `date` | string | Date of sale |
| `price` | float | **TARGET — Sale price in USD** |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms |
| `sqft_living` | int | Living area in sqft |
| `sqft_lot` | int | Lot size in sqft |
| `floors` | float | Number of floors |
| `waterfront` | int | 0/1 — waterfront property? |
| `view` | int | 0-4 — view quality rating |
| `condition` | int | 1-5 — property condition |
| `grade` | int | 1-13 — construction quality |
| `sqft_above` | int | Sqft above ground |
| `sqft_basement` | int | Sqft basement |
| `yr_built` | int | Year built |
| `yr_renovated` | int | Year renovated (0 = never) |
| `zipcode` | int | ZIP code |
| `lat` | float | Latitude |
| `long` | float | Longitude |
| `sqft_living15` | int | Avg sqft of 15 nearest neighbors (living) |
| `sqft_lot15` | int | Avg sqft of 15 nearest neighbors (lot) |

## Initial Sanity Check

After downloading, verify:

```python
import pandas as pd

df = pd.read_csv('data/raw/kc_house_data.csv')

print(f"Rows: {len(df):,}")           # Expect ~21,613
print(f"Columns: {len(df.columns)}")   # Expect 21
print(f"Target range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
```

Expected:
```
Rows: 21,613
Columns: 21
Target range: $75,000 - $7,700,000
Missing values: (none or minimal)
```

---

## Next Step

→ Go to [03-data-exploration](../03-data-exploration/) to understand the data visually and statistically.
