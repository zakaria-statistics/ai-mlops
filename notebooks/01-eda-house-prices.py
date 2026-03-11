# %% [markdown]
# # House Price EDA — King County Dataset
# > Explore the data before building any model.
#
# Run cell-by-cell in Jupyter/VSCode, or as a full script.

# %% [markdown]
# ## 1. Setup & Load Data

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Display settings
pd.set_option('display.max_columns', 25)
pd.set_option('display.float_format', '{:,.2f}'.format)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = pd.read_csv('../data/raw/kc_house_data.csv')
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# %% [markdown]
# ## 2. First Look — Shape, Types, Sample

# %%
print("=== DATA TYPES ===")
print(df.dtypes)
print(f"\n=== FIRST 3 ROWS ===")
df.head(3)

# %%
print("=== BASIC STATISTICS ===")
df.describe()

# %% [markdown]
# ## 3. Missing Values

# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})
missing_df = missing_df[missing_df['count'] > 0]

if len(missing_df) == 0:
    print("No missing values — clean dataset.")
else:
    print("=== MISSING VALUES ===")
    print(missing_df.sort_values('count', ascending=False))

# %% [markdown]
# ## 4. Target Variable — Price Distribution
#
# **Key question:** Is price normally distributed or skewed?

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Raw distribution
axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Price Distribution (Raw)')
axes[0].set_xlabel('Price ($)')
axes[0].axvline(df['price'].median(), color='red', linestyle='--', label=f"Median: ${df['price'].median():,.0f}")
axes[0].axvline(df['price'].mean(), color='orange', linestyle='--', label=f"Mean: ${df['price'].mean():,.0f}")
axes[0].legend()

# Log-transformed
axes[1].hist(np.log1p(df['price']), bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title('Price Distribution (Log Transformed)')
axes[1].set_xlabel('log(Price)')

# Box plot
axes[2].boxplot(df['price'], vert=True)
axes[2].set_title('Price Box Plot')
axes[2].set_ylabel('Price ($)')

plt.tight_layout()
plt.savefig('03-data-exploration/price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPrice Summary:")
print(f"  Min:    ${df['price'].min():>12,.0f}")
print(f"  25th:   ${df['price'].quantile(0.25):>12,.0f}")
print(f"  Median: ${df['price'].median():>12,.0f}")
print(f"  75th:   ${df['price'].quantile(0.75):>12,.0f}")
print(f"  Max:    ${df['price'].max():>12,.0f}")
print(f"  Skew:   {df['price'].skew():>12.2f}  {'← Right-skewed (consider log transform)' if df['price'].skew() > 1 else ''}")

# %% [markdown]
# ## 5. Feature Correlations with Price
#
# **Key question:** Which features have the strongest relationship with price?

# %%
# Correlation with price (numeric columns only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['price'].drop('price').sort_values(ascending=False)

print("=== CORRELATIONS WITH PRICE ===")
print(correlations.to_string())

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
correlations.plot(kind='barh', ax=ax, color=['green' if x > 0 else 'red' for x in correlations])
ax.set_title('Feature Correlations with Price')
ax.set_xlabel('Correlation Coefficient')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('03-data-exploration/correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Top Features vs Price — Scatter Plots
#
# Visualize the strongest predictors.

# %%
top_features = ['sqft_living', 'grade', 'sqft_above', 'bathrooms', 'bedrooms']

fig, axes = plt.subplots(1, len(top_features), figsize=(20, 4))
for i, feat in enumerate(top_features):
    axes[i].scatter(df[feat], df['price'], alpha=0.1, s=5)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Price ($)' if i == 0 else '')
    axes[i].set_title(f'{feat} vs Price\n(r={df[feat].corr(df["price"]):.2f})')

plt.tight_layout()
plt.savefig('03-data-exploration/top_features_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Categorical/Discrete Features vs Price
#
# How do discrete groups (grade, condition, floors, waterfront) affect price?

# %%
cat_features = ['grade', 'condition', 'floors', 'waterfront', 'view']

fig, axes = plt.subplots(1, len(cat_features), figsize=(20, 5))
for i, feat in enumerate(cat_features):
    df.boxplot(column='price', by=feat, ax=axes[i])
    axes[i].set_title(feat)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Price ($)' if i == 0 else '')

plt.suptitle('Price by Categorical Features', y=1.02)
plt.tight_layout()
plt.savefig('03-data-exploration/categorical_vs_price.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Outlier Detection
#
# **Key question:** Are there suspicious or extreme values?

# %%
print("=== POTENTIAL OUTLIERS ===\n")

# Bedrooms
print(f"Bedrooms distribution:")
print(df['bedrooms'].value_counts().sort_index().to_string())
suspicious_beds = df[df['bedrooms'] > 10]
if len(suspicious_beds) > 0:
    print(f"\n⚠ {len(suspicious_beds)} houses with >10 bedrooms:")
    print(suspicious_beds[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'grade']].to_string())

# Price extremes
print(f"\n\nTop 10 most expensive:")
print(df.nlargest(10, 'price')[['price', 'bedrooms', 'sqft_living', 'grade', 'zipcode']].to_string())

print(f"\n\nBottom 10 cheapest:")
print(df.nsmallest(10, 'price')[['price', 'bedrooms', 'sqft_living', 'grade', 'zipcode']].to_string())

# %% [markdown]
# ## 9. Geographic Analysis
#
# **Key question:** Does location drive price?

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Price by location (scatter on lat/long)
scatter = axes[0].scatter(df['long'], df['lat'], c=df['price'], cmap='YlOrRd',
                          alpha=0.3, s=3, vmax=df['price'].quantile(0.95))
axes[0].set_title('Price by Location')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
plt.colorbar(scatter, ax=axes[0], label='Price ($)')

# Median price by zipcode (top 15)
zip_prices = df.groupby('zipcode')['price'].median().sort_values(ascending=False)
zip_prices.head(15).plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Top 15 Zipcodes by Median Price')
axes[1].set_xlabel('Median Price ($)')

plt.tight_layout()
plt.savefig('03-data-exploration/geographic_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nZipcode price range:")
print(f"  Most expensive:  {zip_prices.index[0]} — median ${zip_prices.iloc[0]:,.0f}")
print(f"  Least expensive: {zip_prices.index[-1]} — median ${zip_prices.iloc[-1]:,.0f}")
print(f"  Ratio: {zip_prices.iloc[0] / zip_prices.iloc[-1]:.1f}x difference")

# %% [markdown]
# ## 10. Feature-to-Feature Relationships
#
# Are any features redundant (highly correlated with each other)?

# %%
# Correlation heatmap (numeric features only, exclude id)
cols_for_heatmap = [c for c in numeric_cols if c not in ['id', 'price']]
corr_matrix = df[cols_for_heatmap].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('03-data-exploration/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Flag highly correlated pairs
print("=== HIGHLY CORRELATED PAIRS (|r| > 0.7) ===")
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            print(f"  {corr_matrix.index[i]:20s} ↔ {corr_matrix.columns[j]:20s}  r={r:.2f}")

# %% [markdown]
# ## 11. Summary — Key Findings
#
# Fill this in after running all cells above:
#
# | Finding | Detail | Action for Modeling |
# |---------|--------|-------------------|
# | Price is right-skewed | Long tail of expensive homes | Consider log transform |
# | Top predictor: sqft_living | r ≈ 0.70 | Must include |
# | Grade strongly predicts | r ≈ 0.67 | Must include |
# | 33-bedroom outlier | Likely data entry error | Drop or fix |
# | Location matters a lot | Zipcode creates 10x+ price difference | Encode location properly |
# | sqft_above ≈ sqft_living | Highly correlated, redundant | May drop one |
# | yr_built weak alone | Non-linear relationship | May need feature engineering (age) |
#
# → Take these findings into [04-data-preparation](../04-data-preparation/)
