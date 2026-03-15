# Lab 00 — ML Landscape Worksheet
> No code — classify business problems, pick metrics, sketch pipelines

**Prerequisites:** Read `theory/00-ml-landscape/GUIDE.md`

---

## Exercise 1: Classify These Business Problems

For each scenario, identify: **ML type** (regression/classification/clustering/anomaly) and **target variable**.

1. A bank wants to predict if a loan applicant will default
2. An e-commerce site wants to estimate next quarter's revenue
3. A streaming service wants to group users by viewing habits
4. A factory wants to detect defective products on the assembly line
5. A hospital wants to predict patient length of stay (days)
6. A telecom company wants to identify which customers will churn
7. A retailer wants to find which products are frequently bought together
8. A credit card company wants to flag suspicious transactions in real-time
9. A real estate company wants to estimate home prices
10. An HR department wants to segment employees by engagement patterns

**Expected output:** Table with: Scenario | ML Type | Target (y) | Supervised/Unsupervised

---

## Exercise 2: Pick the Right Metric

For each scenario above, pick the PRIMARY evaluation metric and explain why.

Consider:
- Is accuracy enough? (balanced classes?)
- What's the cost of false positives vs false negatives?
- Is interpretability important?

**Expected output:** Table with: Scenario | Metric | Why this metric

---

## Exercise 3: Sketch the Pipeline

Pick ONE scenario from above. Write out the 9 pipeline steps with specifics:

```
1. Business Problem: [what decision, who cares, what's success]
2. Data Collection: [what data sources, what features would you need]
3. EDA: [what would you look for]
4. Data Prep: [encoding, scaling, split strategy]
5. Feature Engineering: [what derived features might help]
6. Modeling: [which algorithm and why]
7. Evaluation: [which metric, cross-validation strategy]
8. Tuning: [what hyperparameters to search]
9. Deployment: [how would this be served]
```

---

## Checkpoint

After completing all 3 exercises, you should be able to:
- [ ] Map any business question to an ML problem type
- [ ] Choose appropriate evaluation metrics based on the problem
- [ ] Sketch a complete ML pipeline for a real scenario
