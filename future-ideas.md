# Future ML Project Ideas

> Business ideas to tackle after completing the house price project. Each follows the same workflow — only the domain, dataset, and algorithm emphasis change.

## Table of Contents

1. [Used Car Pricing](#1-used-car-pricing)
2. [Rental Price Prediction](#2-rental-price-prediction)
3. [Salary Prediction](#3-salary-prediction)
4. [Insurance Cost Prediction](#4-insurance-cost-prediction)
5. [Customer Churn](#5-customer-churn)
6. [Product Demand Forecasting](#6-product-demand-forecasting)
7. [Credit Risk Scoring](#7-credit-risk-scoring)
8. [Energy Consumption Prediction](#8-energy-consumption-prediction)

---

## 1. Used Car Pricing

**Question:** "What is this used car worth?"

| Aspect | Detail |
|--------|--------|
| **ML Type** | Regression |
| **Target** | Sale price ($) |
| **Key Features** | Brand, model, year, mileage, fuel type, transmission, engine size, owner count |
| **Dataset** | Vehicle Dataset (Kaggle, ~300K rows) or Craigslist Cars (Kaggle, ~400K rows) |
| **Who cares** | Buyers (avoid overpaying), sellers (maximize price), dealers (fair pricing) |
| **Learning focus** | High-cardinality categoricals (brand/model), feature interaction (age × mileage) |

**Why interesting:** Similar to house pricing but with different feature dynamics — cars depreciate (houses appreciate), brand matters more than location.

---

## 2. Rental Price Prediction

**Question:** "What should I charge/pay for rent?"

| Aspect | Detail |
|--------|--------|
| **ML Type** | Regression |
| **Target** | Monthly rent ($) |
| **Key Features** | City, area (m²/sqft), rooms, furnishing, floor, parking, pets allowed |
| **Dataset** | Brazil Houses to Rent (Kaggle, ~10K) or USA Apartment Listings |
| **Who cares** | Tenants, landlords, property managers, investors (yield calculation) |
| **Learning focus** | Categorical encoding (city, furnishing), comparing different markets |

**Why interesting:** Complements the buy/sell house project — together they form a complete real estate pricing system.

---

## 3. Salary Prediction

**Question:** "What salary should I expect for this role?"

| Aspect | Detail |
|--------|--------|
| **ML Type** | Regression |
| **Target** | Annual salary ($) |
| **Key Features** | Job title, experience years, education, location, company size, skills, industry |
| **Dataset** | Stack Overflow Survey (annual, ~70K), Glassdoor data, or DS/ML Salaries (Kaggle) |
| **Who cares** | Job seekers, recruiters, HR departments, career planners |
| **Learning focus** | Text features (job title), ordinal encoding (education levels), geographic effects |

**Why interesting:** Personal relevance — everyone wants to know their market value.

---

## 4. Insurance Cost Prediction

**Question:** "What will my health insurance premium be?"

| Aspect | Detail |
|--------|--------|
| **ML Type** | Regression |
| **Target** | Annual premium ($) |
| **Key Features** | Age, sex, BMI, children, smoker status, region |
| **Dataset** | Medical Cost Personal (Kaggle, ~1.3K rows — small but clean) |
| **Who cares** | Individuals, insurance companies, policy makers |
| **Learning focus** | Small dataset techniques, feature importance (smoker is dominant), interaction effects |

**Why interesting:** Small dataset = fast iteration. Clear feature importance story (smoking effect is dramatic).

---

## 5. Customer Churn

**Question:** "Will this customer leave?"

| Aspect | Detail |
|--------|--------|
| **ML Type** | **Classification** (binary) |
| **Target** | Churn: yes/no |
| **Key Features** | Tenure, monthly charges, contract type, payment method, services used |
| **Dataset** | Telco Customer Churn (Kaggle, ~7K rows) |
| **Who cares** | Subscription businesses, telecom, SaaS companies |
| **Learning focus** | Classification metrics (precision, recall, F1), class imbalance, business cost matrix |

**Why interesting:** First classification project. Introduces precision/recall trade-offs — missing a churner costs more than false alarm.

---

## 6. Product Demand Forecasting

**Question:** "How many units will we sell next week/month?"

| Aspect | Detail |
|--------|--------|
| **ML Type** | Regression / **Time Series** |
| **Target** | Units sold (count) |
| **Key Features** | Date, product category, store location, promotions, holidays, seasonality |
| **Dataset** | Store Sales (Kaggle, ~3M rows) or Walmart Sales (Kaggle) |
| **Who cares** | Retail managers, supply chain, warehouse operations |
| **Learning focus** | Time features (day of week, month), lag features, seasonality, large dataset handling |

**Why interesting:** Introduces time dimension. Real operational impact — overstock vs stockout costs.

---

## 7. Credit Risk Scoring

**Question:** "Will this borrower default on their loan?"

| Aspect | Detail |
|--------|--------|
| **ML Type** | **Classification** (binary) |
| **Target** | Default: yes/no |
| **Key Features** | Income, loan amount, credit history, employment length, home ownership, DTI ratio |
| **Dataset** | Lending Club (Kaggle, ~2M rows) or German Credit (UCI, ~1K rows) |
| **Who cares** | Banks, fintech lenders, regulators |
| **Learning focus** | Severe class imbalance, model explainability (regulatory requirement), threshold tuning |

**Why interesting:** Real business consequence — approve a bad loan = lose money, reject a good borrower = lose revenue. Explainability is legally required.

---

## 8. Energy Consumption Prediction

**Question:** "How much energy will this building use?"

| Aspect | Detail |
|--------|--------|
| **ML Type** | Regression / Time Series |
| **Target** | Energy consumption (kWh) |
| **Key Features** | Building type, size, temperature, humidity, occupancy, time of day, season |
| **Dataset** | ASHRAE Energy Prediction (Kaggle, ~20M rows) or UCI Household Power |
| **Who cares** | Building managers, utilities, sustainability teams |
| **Learning focus** | Large-scale data, weather features, time patterns, environmental impact |

**Why interesting:** Green tech relevance. Complex feature interactions (weather × building type × time).

---

## Suggested Learning Order

```
DONE:  House Price Prediction (regression basics)
  │
  ├─► Used Car Pricing          ← Similar structure, different domain
  │     or Rental Prediction      Reinforces regression skills
  │
  ├─► Customer Churn            ← First classification project
  │     or Credit Risk            Learn precision/recall/F1
  │
  ├─► Demand Forecasting        ← Introduces time dimension
  │     or Energy Consumption     Learn time series features
  │
  └─► Pick based on interest    ← Apply skills to chosen domain
```

Each project reuses the same 10-phase workflow from this repo — just swap the dataset and tune the approach.
