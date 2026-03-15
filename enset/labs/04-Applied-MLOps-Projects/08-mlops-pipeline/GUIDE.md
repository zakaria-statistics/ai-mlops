# Lab 08: Setting up an MLOps Pipeline

## 🎯 Lab Overview
Create reproducible and deployable machine learning workflows.

## 🏗️ Architecture
- **Pipeline**: Scikit-learn `Pipeline` (Impute -> Scale -> Model)
- **Persistence**: `Joblib` (Model Serialization)
- **Automation**: Modular preprocessing

## 🚀 Step-by-Step Guide
1. Build a Scikit-learn pipeline object.
2. Integrate data cleaning (Imputation) and scaling into the pipeline.
3. Fit the entire pipeline on training data in one command.
4. Serialize (save) the trained pipeline to a `.pkl` file.
5. Reload the model and perform 'inference' (prediction) on new data.

## 💡 Key Concepts
- MLOps, Data Leakage prevention, Model Serialization (Pickling), Inference.

---
*Generated based on ENSET Course Materials.*
