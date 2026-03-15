import os

labs_metadata = [
    {
        "path": "labs/01-Foundations/01-eda",
        "title": "Lab 01: Exploratory Data Analysis (EDA)",
        "goal": "Master the art of data understanding using the Churn Modelling dataset.",
        "architecture": """- **Data Source**: `Churn_Modelling.csv` (Bank customer data)
- **Tools**: Pandas (Data manipulation), Seaborn/Matplotlib (Visualization)
- **Process**: Data Loading -> Info/Describe -> Missing Values -> Visual Analysis -> Correlation Heatmap""",
        "steps": """1. Load the dataset and inspect the first few rows.
2. Identify data types and check for null values.
3. Visualize the distribution of the 'Exited' column.
4. Analyze numerical features (Age, Balance, etc.) using histograms.
5. Use a heatmap to find relationships between variables.""",
        "concepts": "Data Cleaning, Descriptive Statistics, Feature Correlation, Data Visualization."
    },
    {
        "path": "labs/01-Foundations/02-linear-regression",
        "title": "Lab 02: Linear Regression",
        "goal": "Predict continuous house prices using multiple features.",
        "architecture": """- **Data Source**: `Housing.csv` (Real estate features)
- **Model**: Scikit-learn `LinearRegression`
- **Evaluation**: Mean Squared Error (MSE), R-squared (R2)""",
        "steps": """1. Prepare data by encoding categorical variables (One-Hot Encoding).
2. Split the dataset into training (80%) and testing (20%) sets.
3. Fit the Linear Regression model to the training data.
4. Predict prices on the test set.
5. Evaluate performance using error metrics and scatter plots.""",
        "concepts": "Supervised Learning, Regression, One-Hot Encoding, Train-Test Split."
    },
    {
        "path": "labs/01-Foundations/03-logistic-regression",
        "title": "Lab 03: Logistic Regression",
        "goal": "Classify whether a user will click on an advertisement.",
        "architecture": """- **Data Source**: `advertising.csv` (User behavior data)
- **Model**: Scikit-learn `LogisticRegression`
- **Evaluation**: Confusion Matrix, Accuracy Score, Classification Report""",
        "steps": """1. Explore the relationship between 'Age' and 'Clicked on Ad'.
2. Select key features like 'Daily Time Spent' and 'Income'.
3. Train the logistic model on the behavioral data.
4. Predict the probability of a click.
5. Analyze the Precision-Recall trade-off using the Confusion Matrix.""",
        "concepts": "Binary Classification, Sigmoid Function, Confusion Matrix, Precision vs. Recall."
    },
    {
        "path": "labs/02-Neural-Networks-Scratch/04-xor-problem",
        "title": "Lab 04: The XOR Problem (From Scratch)",
        "goal": "Solve a non-linearly separable problem using a Multi-Layer Perceptron (MLP).",
        "architecture": """- **Data**: XOR Truth Table (4 samples)
- **Hidden Layers**: 1 layer with 2 neurons
- **Activation**: Sigmoid
- **Optimization**: Gradient Descent (Manual)""",
        "steps": """1. Define the XOR inputs and outputs.
2. Initialize weights and biases randomly.
3. Implement the Forward Pass (Dot product -> Sigmoid).
4. Implement Backpropagation (Calculate error -> Update weights).
5. Run for 10,000 iterations to observe the network 'learning' the XOR logic.""",
        "concepts": "Multi-Layer Perceptron (MLP), Backpropagation, Sigmoid Derivative, Gradient Descent."
    },
    {
        "path": "labs/02-Neural-Networks-Scratch/05-backpropagation",
        "title": "Lab 05: Backpropagation Step-by-Step",
        "goal": "Build a modular, object-oriented neural network framework from scratch.",
        "architecture": """- **Module 1**: `Layer` class (Weights/Biases management)
- **Module 2**: `Activation_ReLU` class
- **Module 3**: `Loss_MSE` class
- **Process**: Forward Propagation -> Loss Calculation -> Backward Propagation""",
        "steps": """1. Create a generic 'Layer' class for any input/output size.
2. Implement the ReLU activation function and its derivative.
3. Build a Mean Squared Error (MSE) loss component.
4. Link components together into a functional network.
5. Train on synthetic data to verify the backward pass gradients.""",
        "concepts": "Modular Programming, ReLU Activation, MSE Loss, Gradient Chain Rule."
    },
    {
        "path": "labs/03-Deep-Learning-Frameworks/06-ann-churn",
        "title": "Lab 06: ANN for Churn Prediction (Keras)",
        "goal": "Develop a professional-grade deep learning model using industry-standard tools.",
        "architecture": """- **Framework**: TensorFlow / Keras
- **Preprocessing**: `StandardScaler`, `LabelEncoder`
- **Regularization**: Dropout Layers
- **Optimizer**: Adam""",
        "steps": """1. Clean and scale the bank churn data.
2. Define a `Sequential` Keras model with Dense and Dropout layers.
3. Compile with 'binary_crossentropy' loss.
4. Train the model using batches and validation splits.
5. Plot the training/validation accuracy to detect overfitting.""",
        "concepts": "Deep Learning, Keras Sequential API, Dropout (Regularization), Adam Optimizer."
    },
    {
        "path": "labs/03-Deep-Learning-Frameworks/07-fashion-mnist",
        "title": "Lab 07: Fashion MNIST Image Classification",
        "goal": "Build a computer vision model to recognize clothing items.",
        "architecture": """- **Dataset**: Fashion MNIST (70,000 28x28 grayscale images)
- **Architecture**: Flatten -> Dense (128) -> Softmax (10 outputs)
- **Metrics**: Sparse Categorical Crossentropy""",
        "steps": """1. Load and normalize image pixel values (0 to 1).
2. Visualize samples from the 10 clothing categories.
3. Build a model that flattens 2D images into 1D vectors.
4. Train the model to categorize images.
5. Test the model's 'vision' on unseen clothing images.""",
        "concepts": "Computer Vision, Image Normalization, Softmax Activation, Categorical Classification."
    },
    {
        "path": "labs/04-Applied-MLOps-Projects/08-mlops-pipeline",
        "title": "Lab 08: Setting up an MLOps Pipeline",
        "goal": "Create reproducible and deployable machine learning workflows.",
        "architecture": """- **Pipeline**: Scikit-learn `Pipeline` (Impute -> Scale -> Model)
- **Persistence**: `Joblib` (Model Serialization)
- **Automation**: Modular preprocessing""",
        "steps": """1. Build a Scikit-learn pipeline object.
2. Integrate data cleaning (Imputation) and scaling into the pipeline.
3. Fit the entire pipeline on training data in one command.
4. Serialize (save) the trained pipeline to a `.pkl` file.
5. Reload the model and perform 'inference' (prediction) on new data.""",
        "concepts": "MLOps, Data Leakage prevention, Model Serialization (Pickling), Inference."
    },
    {
        "path": "labs/04-Applied-MLOps-Projects/09-bank-marketing",
        "title": "Lab 09: Bank Marketing Campaign Classification",
        "goal": "Solve a real-world imbalanced classification problem.",
        "architecture": """- **Model**: RandomForestClassifier
- **Handling Imbalance**: `class_weight='balanced'`
- **Evaluation**: ROC-AUC Score and Curve""",
        "steps": """1. Process the bank campaign dataset (`bank.csv`).
2. Address the class imbalance (most people say 'no' to marketing).
3. Train a Random Forest ensemble model.
4. Evaluate using the ROC-AUC score (more robust than simple accuracy).
5. Visualize the ROC Curve.""",
        "concepts": "Ensemble Learning, Imbalanced Datasets, ROC-AUC, Random Forest."
    },
    {
        "path": "labs/04-Applied-MLOps-Projects/10-housing-advanced",
        "title": "Lab 10: Advanced Housing Price Prediction",
        "goal": "Use hyperparameter tuning to reach state-of-the-art performance.",
        "architecture": """- **Algorithm**: Gradient Boosting Regressor
- **Optimization**: `GridSearchCV` (Auto-tuning)
- **Cross-Validation**: 3-fold CV""",
        "steps": """1. Initialize a Gradient Boosting model.
2. Define a 'grid' of hyperparameters (learning rate, depth, etc.).
3. Use `GridSearchCV` to automatically find the best combination.
4. Re-train the model using the optimal parameters.
5. Compare the 'Tuned' model performance against a baseline model.""",
        "concepts": "Hyperparameter Tuning, Gradient Boosting, Cross-Validation, Model Optimization."
    }
]

for lab in labs_metadata:
    guide_content = f"""# {lab['title']}

## 🎯 Lab Overview
{lab['goal']}

## 🏗️ Architecture
{lab['architecture']}

## 🚀 Step-by-Step Guide
{lab['steps']}

## 💡 Key Concepts
- {lab['concepts']}

---
*Generated based on ENSET Course Materials.*
"""
    os.makedirs(lab['path'], exist_ok=True)
    guide_path = os.path.join(lab['path'], "GUIDE.md")
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide_content)
    print(f"Created: {guide_path}")

print("\n✅ All GUIDES created successfully.")
