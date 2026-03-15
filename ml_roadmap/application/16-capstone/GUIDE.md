# Lab 16: Capstone Project
> End-to-end ML project — pick a problem, build the full pipeline, deploy and present

## Table of Contents
1. [Overview](#overview) - What this lab is
2. [Phase 1: Problem Definition](#phase-1-problem-definition) - Frame the question
3. [Phase 2: Data](#phase-2-data) - Acquire and inspect
4. [Phase 3: EDA](#phase-3-eda) - Explore thoroughly
5. [Phase 4: Preparation](#phase-4-preparation) - Clean and transform
6. [Phase 5: Baseline](#phase-5-baseline) - Simplest model first
7. [Phase 6: Iterate](#phase-6-iterate) - Try multiple algorithms
8. [Phase 7: Tune](#phase-7-tune) - Optimize the best model
9. [Phase 8: Evaluate](#phase-8-evaluate) - Honest assessment
10. [Phase 9: Deploy](#phase-9-deploy) - Package and serve
11. [Phase 10: Present](#phase-10-present) - Document and share
12. [Suggested Projects](#suggested-projects) - Five starter ideas
13. [Grading Rubric](#grading-rubric) - Self-assessment criteria

## Overview

This is not a step-by-step tutorial. It is a **checklist and template** for your own end-to-end ML project. You choose the dataset, the problem, and the approach. The phases below are your guide — follow them in order, checking off each item before moving on.

**Goal:** Demonstrate that you can take a real problem from raw data to a deployed model with honest evaluation.

## Prerequisites
- All previous labs (01-15)
- Theory guides for the algorithm types you plan to use
- Reference: `ml_roadmap/REFERENCE.md` for algorithm selection

---

## Phase 1: Problem Definition

Before writing any code, answer these questions in writing:

- [ ] **What is the problem?** One sentence. (e.g., "Predict whether a credit card transaction is fraudulent")
- [ ] **What type of ML problem is this?** Binary classification, multi-class classification, regression, clustering, time series forecasting?
- [ ] **What is the target variable?** Name it, describe its distribution
- [ ] **What is the success metric?** Accuracy? F1? RMSE? Why this metric? (e.g., for fraud detection, recall matters more than accuracy)
- [ ] **What is the baseline to beat?** Majority class accuracy? Mean prediction? Naive forecast?
- [ ] **Who are the stakeholders?** Who would use this model and how?
- [ ] **What are the ethical considerations?** Bias in data? Impact of false positives/negatives?

Write a 1-paragraph problem statement summarizing the above.

> **Checkpoint 1:** You have a written problem statement with a clear target variable and success metric. Do not proceed without this.

---

## Phase 2: Data

- [ ] **Find or collect the dataset** — Kaggle, UCI ML Repository, government open data, APIs, web scraping
- [ ] **Initial quality check:**
  - How many rows and columns?
  - Any missing values? What percentage?
  - Any duplicates?
  - Data types correct?
- [ ] **Document a data dictionary:** for every column, write: name, type, description, example values
- [ ] **Check for data leakage:** does any feature contain information from the future or from the target?
- [ ] **Train/test split NOW** — before any exploration or preprocessing. Hold out 20% and don't touch it until Phase 8.

---

## Phase 3: EDA

Follow the Lab 01 pattern on your training set only:

- [ ] Univariate analysis: distribution of every feature and the target
- [ ] Bivariate analysis: relationship between each feature and the target
- [ ] Correlation matrix / heatmap
- [ ] Identify outliers and decide how to handle them (with justification)
- [ ] Check class balance (classification) or target distribution (regression)
- [ ] At least 5 visualizations that tell a story about the data
- [ ] Write 3-5 key findings from EDA that will inform modeling decisions

> **Checkpoint 2:** You have a thorough EDA with documented findings. Your modeling strategy should be informed by what you found here.

---

## Phase 4: Preparation

Follow the Lab 02 pattern. For every transformation, **justify why**:

- [ ] Handle missing values (drop, impute — which strategy and why?)
- [ ] Encode categorical variables (one-hot, label, target encoding — why?)
- [ ] Scale/normalize numerical features (StandardScaler, MinMaxScaler — why?)
- [ ] Feature engineering: create new features from existing ones (if applicable)
- [ ] Handle class imbalance (if applicable): SMOTE, undersampling, class weights
- [ ] Build a preprocessing pipeline (`sklearn.pipeline.Pipeline`) for reproducibility
- [ ] Apply pipeline to training data only. Test data gets transformed, never fit.

---

## Phase 5: Baseline

The simplest reasonable model. This sets the floor.

- [ ] **Classification:** LogisticRegression with default parameters
- [ ] **Regression:** LinearRegression or Ridge with default parameters
- [ ] **Time series:** Naive forecast (predict last value) or simple moving average
- [ ] Evaluate on training set with cross-validation
- [ ] Record the metric — this is the number every other model must beat
- [ ] If baseline already meets your success metric, the problem may be too easy — choose a harder one

> **Checkpoint 3:** You have a baseline score. Every subsequent model is compared against this number.

---

## Phase 6: Iterate

Try at least 3 different algorithms. Refer to `REFERENCE.md` for selection guidance.

- [ ] Model 1: baseline (from Phase 5)
- [ ] Model 2: a different algorithm family (e.g., if baseline was linear, try tree-based)
- [ ] Model 3: another algorithm family (e.g., ensemble, SVM, neural network)
- [ ] Optional Model 4+: more algorithms if time permits
- [ ] **All evaluation via cross-validation** on training set (5-fold stratified for classification)
- [ ] Create a comparison table:

```
| Model              | CV Mean | CV Std | Notes               |
|--------------------|---------|--------|---------------------|
| Logistic Regression| 0.XXX   | 0.0XX  | Baseline            |
| Random Forest      | 0.XXX   | 0.0XX  | Better on non-linear|
| XGBoost            | 0.XXX   | 0.0XX  | Best so far         |
```

- [ ] For top 2 models, check if the difference is statistically significant (paired t-test on CV folds, from Lab 11)

---

## Phase 7: Tune

Optimize hyperparameters for the best model from Phase 6:

- [ ] Define the search space (which hyperparameters, what ranges)
- [ ] Use `RandomizedSearchCV` or `GridSearchCV` (from Lab 11)
- [ ] Use nested cross-validation if you want an unbiased estimate
- [ ] Record best hyperparameters and best CV score
- [ ] Compare: default params vs tuned params — how much did tuning help?
- [ ] Beware of overfitting to the validation folds — if tuned score is much better than default, be skeptical

> **Checkpoint 4:** You have a tuned model with documented hyperparameters. You know how much tuning helped.

---

## Phase 8: Evaluate

This is the moment of truth. Use the held-out test set for the first (and only) time.

- [ ] Retrain best model on full training set with best hyperparameters
- [ ] Predict on test set
- [ ] Compute your primary metric and secondary metrics:
  - Classification: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
  - Regression: MAE, RMSE, R-squared, residual plot
- [ ] **Error analysis:** look at the misclassified/worst-predicted examples. Why did the model fail? Are there patterns?
- [ ] **Document limitations:** what the model can't do, where it fails, edge cases
- [ ] Compare test performance with CV estimate — if there's a big gap, investigate (data leakage? distribution shift?)

---

## Phase 9: Deploy

Package the model for use. Reference `ai_mlops` project Phases 8-9 for the pattern:

- [ ] Save the trained model (joblib, pickle, or torch.save)
- [ ] Save the preprocessing pipeline (same serialization)
- [ ] Create a prediction function that takes raw input and returns prediction
- [ ] Wrap in a FastAPI endpoint:
  ```python
  @app.post("/predict")
  def predict(data: InputSchema):
      processed = pipeline.transform(data)
      prediction = model.predict(processed)
      return {"prediction": prediction}
  ```
- [ ] Optional: Dockerize (see `ai_mlops` Phase 9 Level 2)
- [ ] Test the API with sample requests

---

## Phase 10: Present

- [ ] Write a project README covering:
  - Problem statement
  - Dataset description and source
  - Approach summary (which models, why)
  - Key results (best metric, comparison table)
  - How to run the code
  - Limitations and future work
- [ ] Clean up your notebook/code — remove dead cells, add section headers
- [ ] Host on GitHub
- [ ] Optional: record a 5-minute walkthrough or create a Streamlit demo

> **Checkpoint 5:** Your project is complete, documented, and shareable. Someone else can understand and reproduce your work.

---

## Suggested Projects

### 1. Credit Card Fraud Detection (Imbalanced Classification)
- **Dataset:** Kaggle credit card fraud dataset (~284K transactions, 0.17% fraud)
- **Challenge:** extreme class imbalance — accuracy is meaningless, optimize F1/recall
- **Key skills:** SMOTE, class weights, precision-recall tradeoff, threshold tuning
- **Algorithms to try:** Logistic Regression, Random Forest, XGBoost, Isolation Forest

### 2. Real Estate Price Prediction (Regression)
- **Dataset:** Ames Housing, California Housing, or Zillow data
- **Challenge:** many features, feature engineering matters (age of house, neighborhood encoding)
- **Key skills:** feature engineering, regularization, residual analysis
- **Algorithms to try:** Ridge, Lasso, Random Forest, Gradient Boosting

### 3. Customer Churn Prediction (Classification + Business)
- **Dataset:** Telco Customer Churn (Kaggle)
- **Challenge:** actionable predictions — which customers to retain, what interventions
- **Key skills:** feature importance, SHAP values, business metric translation
- **Algorithms to try:** Logistic Regression (interpretable), XGBoost (accurate), threshold optimization

### 4. Image Classifier (CNN)
- **Dataset:** Your own images (scrape or photograph), or a specialized subset of ImageNet
- **Challenge:** data collection/augmentation, transfer learning, small dataset techniques
- **Key skills:** data augmentation, fine-tuning pretrained models, learning rate scheduling
- **Algorithms to try:** Simple CNN, ResNet18 fine-tuned, MobileNet

### 5. Stock/News Sentiment Analysis (NLP + Optional Time Series)
- **Dataset:** Financial news headlines + stock price data
- **Challenge:** noisy text, sentiment is subjective, correlation with price is weak
- **Key skills:** text preprocessing, sentiment scoring, correlation analysis
- **Algorithms to try:** Naive Bayes, LSTM, simple ensemble of NLP + time series features

---

## Grading Rubric

Self-assess your project on these dimensions:

| Dimension             | Weak (1-2)                        | Solid (3-4)                            | Strong (5)                              |
|-----------------------|-----------------------------------|----------------------------------------|-----------------------------------------|
| **Problem Framing**   | Vague problem, no success metric  | Clear problem, metric defined          | Metric justified, stakeholders identified|
| **EDA Quality**       | Few plots, no insights            | Thorough exploration, key findings     | Insights drive modeling decisions        |
| **Model Comparison**  | Single model, no baselines        | 3+ models, cross-validated             | Statistical comparison, ablation study   |
| **Evaluation Honesty**| Reported train accuracy            | Proper train/test split, CV            | Nested CV, error analysis, limitations   |
| **Code Quality**      | Messy notebook, hardcoded values  | Organized, functions, pipeline          | Modular, reproducible, well-documented   |
| **Documentation**     | No README, unclear process        | README with results and instructions   | Full narrative, decisions explained       |
| **Deployment**        | No deployment                     | Saved model with load script           | API endpoint, Docker, tested             |

**Target:** 4+ on every dimension. A score of 3 is acceptable, 5 is exceptional.

---

## Summary Checklist
- [ ] Phase 1: Problem statement written
- [ ] Phase 2: Data acquired, dictionary documented, test set held out
- [ ] Phase 3: EDA completed with 5+ visualizations and written findings
- [ ] Phase 4: Preprocessing pipeline built with justified choices
- [ ] Phase 5: Baseline model trained and scored
- [ ] Phase 6: 3+ models compared with cross-validation
- [ ] Phase 7: Best model tuned with documented hyperparameters
- [ ] Phase 8: Test set evaluation done ONCE, error analysis completed
- [ ] Phase 9: Model packaged with prediction API
- [ ] Phase 10: README written, code cleaned, hosted on GitHub
