# %% [markdown]
# # Data Preparation — King County House Prices
# > Clean, transform, and split data based on EDA findings.
#
# **EDA findings driving this notebook:**
#
# | Finding | Action |
# |---------|--------|
# | Price skew = 4.02 | Log transform target |
# | sqft_above ≈ sqft_living (r=0.88) | Drop sqft_above |
# | 33-bedroom outlier | Cap bedrooms at 10 |
# | yr_built weak (r=0.05) | Engineer house_age feature |
# | Location 8.1x price range | Keep lat/long, drop zipcode (numeric misleading) |
# | id, date not predictive | Drop both |

# %% [markdown]
# ## 1. Setup & Load

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/raw/kc_house_data.csv')
print(f"Raw: {df.shape[0]:,} rows x {df.shape[1]} columns")

# %% [markdown]
# ## 2. Drop Unnecessary Columns

# %%
drop_cols = ['id', 'date', 'sqft_above', 'zipcode']
df = df.drop(columns=drop_cols)
print(f"Dropped: {drop_cols}")
print(f"Remaining: {df.shape[1]} columns")

# %% [markdown]
# ## 3. Handle Outliers

# %%
# Cap bedrooms at 10 (33-bedroom entry is a data error)
before = len(df[df['bedrooms'] > 10])
df['bedrooms'] = df['bedrooms'].clip(upper=10)
print(f"Capped {before} rows with bedrooms > 10")

# %% [markdown]
# ## 4. Feature Engineering

# %%
# House age (more useful than yr_built)
df['house_age'] = 2015 - df['yr_built']  # dataset is from 2014-2015

# Has been renovated (binary)
df['has_renovation'] = (df['yr_renovated'] > 0).astype(int)

# Drop original columns
df = df.drop(columns=['yr_built', 'yr_renovated'])

print("New features: house_age, has_renovation")
print(f"house_age range: {df['house_age'].min()} - {df['house_age'].max()} years")

# %% [markdown]
# ## 5. Log Transform Target

# %%
df['log_price'] = np.log1p(df['price'])
print(f"Price skew:     {df['price'].skew():.2f}")
print(f"Log_price skew: {df['log_price'].skew():.2f}")

# %% [markdown]
# ## 6. Train/Test Split

# %%
X = df.drop(columns=['price', 'log_price'])
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]:,} rows")
print(f"Test:  {X_test.shape[0]:,} rows")
print(f"Features: {X_train.shape[1]}")
print(f"\nFeature list: {list(X_train.columns)}")

# %% [markdown]
# ## 7. Save Processed Data

# %%
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)

print("Saved to data/processed/:")
print("  X_train.csv, X_test.csv, y_train.csv, y_test.csv")

# %% [markdown]
# ## Summary
#
# | Step | Was | Now |
# |------|-----|-----|
# | Columns | 21 | 16 features |
# | Outliers | 33-bedroom row | Capped at 10 |
# | Target | price (skew 4.02) | log_price (near-normal) |
# | yr_built | Weak predictor (r=0.05) | house_age feature |
# | yr_renovated | Mostly zeros | has_renovation binary |
# | sqft_above | Redundant (r=0.88) | Dropped |
#
# -> Next: baseline model training
