# Phase 1: Business Problem Definition

> Predict residential property sale prices to help buyers and sellers make informed decisions.

## Table of Contents

1. [The Problem](#the-problem)
2. [Who Cares?](#who-cares)
3. [Success Criteria](#success-criteria)
4. [Scope & Constraints](#scope--constraints)
5. [ML Framing](#ml-framing)

---

## The Problem

**Business Question:** *"What is the fair market price of this property?"*

One model, two business views:

```
┌─────────────────────────────────────────────────────────┐
│              HOUSE PRICE PREDICTION                      │
│                                                         │
│              ┌─────────────────┐                        │
│              │   ONE MODEL     │                        │
│              │   predicts      │                        │
│              │   market value  │                        │
│              └────────┬────────┘                        │
│                       │                                 │
│            ┌──────────┴──────────┐                      │
│            ▼                     ▼                      │
│     BUYER'S VIEW          SELLER'S VIEW                 │
│     "Am I overpaying?"    "What should I list at?"      │
│                                                         │
│     Listed: $550K         Model says: $520K             │
│     Model:  $510K         → List at $520-540K           │
│     → Overpriced by 8%   → Room to negotiate up         │
└─────────────────────────────────────────────────────────┘
```

### The Cost of Getting It Wrong

```
Buyer overpays by 10% on a $400K house  = $40,000 lost equity on day one
Seller underprices by 10%               = $40,000 left on the table
Agent misprices a listing               = Weeks on market or lost commission
```

## Who Cares?

| Stakeholder | How They Use the Model |
|-------------|----------------------|
| **Home Buyer** | Compare listing price vs predicted value → negotiate |
| **Home Seller** | Get data-backed listing price suggestion |
| **Real Estate Agent** | Price listings accurately, advise clients |
| **Bank/Lender** | Automated property valuation for mortgages |
| **Investor** | Screen properties for undervalued deals |

### MVP Scenario

**Buyer asks:**
> "I found a 3-bed, 1,800 sqft house in zipcode 98115, built 1985, good condition. Listed at $550K."

**Model answers:** Predicted fair value: **$510K** → listing is 8% above market, negotiate down.

**Seller asks:**
> "I want to sell my 4-bed, 2,500 sqft waterfront house, renovated 2010, grade 9."

**Model answers:** Suggested list price: **$780K** — similar homes sold $750-820K.

## Success Criteria

### Business Metrics

| Metric | Target | Why |
|--------|--------|-----|
| **Median Absolute Error** | < 10% of median price | Predictions useful if within 10% |
| **R² Score** | > 0.80 | Model explains 80%+ of price variance |
| **Outlier rate** | < 5% predictions off by >25% | Few catastrophically wrong predictions |

### Learning Metrics (for this project)

- [ ] Understand regression problem framing
- [ ] Collect and explore a real dataset
- [ ] Build baseline model (linear regression)
- [ ] Iterate through increasingly complex models
- [ ] Evaluate and compare model performance
- [ ] Package and serve the best model

## Scope & Constraints

### In Scope

- Residential properties (houses) in King County, WA (Seattle area)
- One target: **sale price** (used for both buy and sell views)
- One dataset: King County House Sales (~21,600 transactions)
- Tabular features: size, rooms, location, age, condition, grade, amenities

### Out of Scope (for now)

- Rental price prediction (see `future-ideas.md` for this as a separate project)
- Commercial real estate
- Time-series price trends
- Image-based features (photos, floor plans)
- Real-time market data integration

## ML Framing

```
Type:       Supervised Learning → Regression
Target:     price (continuous, in USD)
Input:      Property features (sqft, bedrooms, location, condition, etc.)
Output:     Predicted sale price (single number)
Evaluation: MAE, RMSE, R², MAPE
Baseline:   Predict median price for all → beat this first
```

### Model Progression (Learning Roadmap)

```
Level 1: Linear Regression          ← Understand the basics
   ↓ "Why not enough?" → Can't capture non-linear relationships

Level 2: Polynomial + Regularization ← Handle non-linearity, overfitting
   ↓ "Why not enough?" → Still assumes global pattern

Level 3: Decision Tree               ← Non-parametric, interpretable
   ↓ "Why not enough?" → High variance, overfits easily

Level 4: Random Forest               ← Ensemble, reduce variance
   ↓ "Why not enough?" → Good but can improve accuracy

Level 5: Gradient Boosting (XGBoost) ← State of the art for tabular
   ↓ "Final step"

Level 6: Compare all + select best   ← Model selection skills
```

Each level builds on the previous — you learn WHY you need the next model by seeing the limitations of the current one.

---

## Next Step

→ Go to [02-data-collection](../02-data-collection/) to download the King County dataset.
