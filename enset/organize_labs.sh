#!/bin/bash

# Configuration
CONVERTED_DIR="converted"
LABS_DIR="labs"

echo "=========================================="
echo " Organizing Labs based on Roadmap"
echo "=========================================="

# Function to create lab directory and copy files
setup_lab() {
    local lab_path="$LABS_DIR/$1"
    mkdir -p "$lab_path"
    shift
    for file in "$@"; do
        if [ -f "$file" ]; then
            echo "  -> Seeding $lab_path with $(basename "$file")"
            cp "$file" "$lab_path/"
        else
            echo "  ⚠️ Warning: $file not found."
        fi
    done
}

# --- Section 01: Foundations ---
setup_lab "01-Foundations/01-eda" \
    "$CONVERTED_DIR/bddc2-deepLearning/Churn_Modelling.csv"

setup_lab "01-Foundations/02-linear-regression" \
    "$CONVERTED_DIR/Machine Learning -bdcc1 -2022-2023/Housing.csv" \
    "$CONVERTED_DIR/Machine Learning -bdcc1 -2022-2023/housing_project.ipynb"

setup_lab "01-Foundations/03-logistic-regression" \
    "$CONVERTED_DIR/Machine Learning -bdcc1 -2022-2023/advertising.csv" \
    "$CONVERTED_DIR/Machine Learning -bdcc1 -2022-2023/supervised_learning_exos_series_v1.ipynb"

# --- Section 02: Neural Networks Scratch ---
setup_lab "02-Neural-Networks-Scratch/04-xor-problem" \
    "$CONVERTED_DIR/DeepLearning-BDCC2-FC/1-MultiLayer NN- Xor problem.pdf" \
    "$CONVERTED_DIR/DeepLearning-BDCC2-FC/recall-LR-NN1.ipynb"

setup_lab "02-Neural-Networks-Scratch/05-backpropagation" \
    "$CONVERTED_DIR/DeepLearning-BDCC2-FC/6-NN-Back Propagation Algorithm - Step1.pdf"

# --- Section 03: Deep Learning Frameworks ---
setup_lab "03-Deep-Learning-Frameworks/06-ann-churn" \
    "$CONVERTED_DIR/bddc2-deepLearning/Churn_Modelling.csv" \
    "$CONVERTED_DIR/bddc2-deepLearning/DL Project 3-Churn_Modelling.pdf"

setup_lab "03-Deep-Learning-Frameworks/07-fashion-mnist" \
    "$CONVERTED_DIR/DeepLearning-BDCC2-FC/4-MultiLayer NN- Fashion problem.pdf"

# --- Section 04: Applied MLOps ---
setup_lab "04-Applied-MLOps-Projects/08-mlops-pipeline" \
    "$CONVERTED_DIR/bddc2-deepLearning/MLOps- Research and developpment VF.pdf"

setup_lab "04-Applied-MLOps-Projects/09-bank-marketing" \
    "$CONVERTED_DIR/bddc2-deepLearning/bank.csv"

setup_lab "04-Applied-MLOps-Projects/10-housing-advanced" \
    "$CONVERTED_DIR/Machine Learning -bdcc1 -2022-2023/Housing.csv"

echo "✅ Lab structure seeded."
