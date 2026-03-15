import json
import os

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}

def save_notebook(path, notebook_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=2)
    print(f"Generated: {path}")

# Lab 1
cells_1 = [
    markdown_cell("""# Lab 01: Exploratory Data Analysis (EDA)

**Goal:** Understand the Churn Modelling dataset by exploring its features, distributions, and missing values."""),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline"""),
    markdown_cell("""## 1. Load the Data"""),
    code_cell("""df = pd.read_csv('Churn_Modelling.csv')
df.head()"""),
    markdown_cell("""## 2. Basic Dataset Info"""),
    code_cell("""print(df.info())
print('\\nMissing values:\\n', df.isnull().sum())"""),
    markdown_cell("""## 3. Statistical Summary"""),
    code_cell("""df.describe()"""),
    markdown_cell("""## 4. Visualizations
Let's visualize the target variable `Exited` and numerical features."""),
    code_cell("""sns.countplot(x='Exited', data=df)
plt.title('Distribution of Target Variable (Exited)')
plt.show()"""),
    code_cell("""numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
df[numerical_features].hist(bins=15, figsize=(15, 10), layout=(2, 3))
plt.show()"""),
    markdown_cell("""## 5. Correlation Analysis"""),
    code_cell("""plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_features + ['Exited']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()""")
]
save_notebook('labs/01-Foundations/01-eda/lab-01.ipynb', create_notebook(cells_1))

# Lab 2
cells_2 = [
    markdown_cell("""# Lab 02: Linear Regression

**Goal:** Predict house prices using Linear Regression on the `Housing.csv` dataset."""),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score"""),
    markdown_cell("""## 1. Load and Prepare Data"""),
    code_cell("""df = pd.read_csv('Housing.csv')
df.head()"""),
    markdown_cell("""Convert categorical text data to numerical formats using one-hot encoding."""),
    code_cell("""categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.head()"""),
    markdown_cell("""## 2. Train-Test Split"""),
    code_cell("""X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training set size:', X_train.shape)
print('Test set size:', X_test.shape)"""),
    markdown_cell("""## 3. Train the Model"""),
    code_cell("""model = LinearRegression()
model.fit(X_train, y_train)
print('Model Coefficients:', model.coef_)"""),
    markdown_cell("""## 4. Evaluate the Model"""),
    code_cell("""y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')"""),
    code_cell("""plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()""")
]
save_notebook('labs/01-Foundations/02-linear-regression/lab-02.ipynb', create_notebook(cells_2))

# Lab 3
cells_3 = [
    markdown_cell("""# Lab 03: Logistic Regression

**Goal:** Build a classification model to predict user clicks on ads using `advertising.csv`."""),
    code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score"""),
    markdown_cell("""## 1. Data Loading"""),
    code_cell("""df = pd.read_csv('advertising.csv')
df.head()"""),
    markdown_cell("""## 2. Brief EDA"""),
    code_cell("""sns.histplot(df['Age'], bins=30, kde=True)
plt.show()"""),
    code_cell("""sns.jointplot(x='Age', y='Area Income', data=df)
plt.show()"""),
    markdown_cell("""## 3. Data Preprocessing"""),
    code_cell("""X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)"""),
    markdown_cell("""## 4. Model Training"""),
    code_cell("""logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)"""),
    markdown_cell("""## 5. Evaluation"""),
    code_cell("""predictions = logmodel.predict(X_test)

print('Classification Report:\\n', classification_report(y_test, predictions))
print('Accuracy:', accuracy_score(y_test, predictions))"""),
    code_cell("""cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()""")
]
save_notebook('labs/01-Foundations/03-logistic-regression/lab-03.ipynb', create_notebook(cells_3))

# Lab 4
cells_4 = [
    markdown_cell("""# Lab 04: The XOR Problem

**Goal:** Understand why a single-layer perceptron fails to solve the XOR problem and how a multi-layer network solves it from scratch."""),
    code_cell("""import numpy as np
import matplotlib.pyplot as plt"""),
    markdown_cell("""## 1. The XOR Data"""),
    code_cell("""X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='bwr', s=100)
plt.title('XOR Logic Gate (Non-linearly separable)')
plt.show()"""),
    markdown_cell("""## 2. Multi-Layer Neural Network from Scratch"""),
    code_cell("""def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Network Architecture (2 inputs, 2 hidden neurons, 1 output)
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

# Initialize weights and biases randomly
np.random.seed(42)
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))"""),
    markdown_cell("""## 3. Training Loop (Forward & Backpropagation)"""),
    code_cell("""learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    # Forward Pass
    hidden_layer_activation = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print('Final Hidden Weights:\\n', hidden_weights)
print('Final Output Weights:\\n', output_weights)"""),
    markdown_cell("""## 4. Final Predictions"""),
    code_cell("""print('Predictions after training:')
print(predicted_output)""")
]
save_notebook('labs/02-Neural-Networks-Scratch/04-xor-problem/lab-04.ipynb', create_notebook(cells_4))

# Lab 5
cells_5 = [
    markdown_cell("""# Lab 05: Backpropagation Step-by-Step

**Goal:** Implement a generalized modular backpropagation algorithm from scratch."""),
    code_cell("""import numpy as np"""),
    markdown_cell("""## 1. Network Components"""),
    code_cell("""class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues, learning_rate):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        # Update weights
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases"""),
    code_cell("""class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Loss_MSE:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)
    def backward(self, y_pred, y_true):
        self.dinputs = 2 * (y_pred - y_true) / y_pred.shape[0]"""),
    markdown_cell("""## 2. Test the Architecture"""),
    code_cell("""X = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])
y = np.array([[0.5], [0.8]])

layer1 = Layer(3, 4)
activation1 = Activation_ReLU()
layer2 = Layer(4, 1)
loss_function = Loss_MSE()

learning_rate = 0.01"""),
    code_cell("""for epoch in range(100):
    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    
    # Loss
    loss = loss_function.forward(layer2.output, y)
    
    # Backward pass
    loss_function.backward(layer2.output, y)
    layer2.backward(loss_function.dinputs, learning_rate)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs, learning_rate)
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.5f}')

print('\\nFinal Predictions:')
print(layer2.output)""")
]
save_notebook('labs/02-Neural-Networks-Scratch/05-backpropagation/lab-05.ipynb', create_notebook(cells_5))

# Lab 6
cells_6 = [
    markdown_cell("""# Lab 06: ANN for Churn Prediction

**Goal:** Build a deep learning model using TensorFlow/Keras to predict bank customer churn."""),
    code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout"""),
    markdown_cell("""## 1. Load Data"""),
    code_cell("""df = pd.read_csv('Churn_Modelling.csv')
df.head()"""),
    markdown_cell("""## 2. Preprocessing
Drop irrelevant columns, encode categorical variables, and scale features."""),
    code_cell("""X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']

# Encode Geography and Gender
le_gender = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])
X = pd.get_dummies(X, columns=['Geography'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""),
    markdown_cell("""## 3. Build ANN Model"""),
    code_cell("""model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()"""),
    markdown_cell("""## 4. Train Model"""),
    code_cell("""history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, verbose=1)"""),
    markdown_cell("""## 5. Evaluate"""),
    code_cell("""import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()"""),
    code_cell("""loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')""")
]
save_notebook('labs/03-Deep-Learning-Frameworks/06-ann-churn/lab-06.ipynb', create_notebook(cells_6))

# Lab 7
cells_7 = [
    markdown_cell("""# Lab 07: Fashion MNIST Image Classification

**Goal:** Build an ANN to classify clothing items using the Fashion MNIST dataset."""),
    code_cell("""import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np"""),
    markdown_cell("""## 1. Load Dataset"""),
    code_cell("""fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']"""),
    markdown_cell("""## 2. Visualize Data"""),
    code_cell("""plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()"""),
    markdown_cell("""## 3. Preprocessing"""),
    code_cell("""train_images = train_images / 255.0
test_images = test_images / 255.0"""),
    markdown_cell("""## 4. Build and Compile Model"""),
    code_cell("""model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])"""),
    markdown_cell("""## 5. Train"""),
    code_cell("""model.fit(train_images, train_labels, epochs=10)"""),
    markdown_cell("""## 6. Evaluate"""),
    code_cell("""test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\\nTest accuracy:', test_acc)""")
]
save_notebook('labs/03-Deep-Learning-Frameworks/07-fashion-mnist/lab-07.ipynb', create_notebook(cells_7))

# Lab 8
cells_8 = [
    markdown_cell("""# Lab 08: Setting up an MLOps Pipeline

**Goal:** Create a robust, reusable scikit-learn pipeline for preprocessing and modeling, and serialize it for deployment."""),
    code_cell("""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib"""),
    markdown_cell("""## 1. Create a Synthetic Dataset for Pipeline Testing"""),
    code_cell("""from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)"""),
    markdown_cell("""## 2. Define the Pipeline"""),
    code_cell("""numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Assuming we had categorical features, we would use ColumnTransformer. 
# For simplicity here, we just use numeric_transformer on all.
pipeline = Pipeline(steps=[
    ('preprocessor', numeric_transformer),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])"""),
    markdown_cell("""## 3. Train the Pipeline"""),
    code_cell("""pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print('Pipeline Accuracy:', accuracy_score(y_test, preds))"""),
    markdown_cell("""## 4. Serialization (The 'Ops' part)"""),
    code_cell("""joblib.dump(pipeline, 'model_pipeline.pkl')
print('Model saved to model_pipeline.pkl')"""),
    code_cell("""loaded_pipeline = joblib.load('model_pipeline.pkl')
loaded_preds = loaded_pipeline.predict(X_test)
print('Loaded Pipeline Accuracy:', accuracy_score(y_test, loaded_preds))""")
]
save_notebook('labs/04-Applied-MLOps-Projects/08-mlops-pipeline/lab-08.ipynb', create_notebook(cells_8))

# Lab 9
cells_9 = [
    markdown_cell("""# Lab 09: Bank Marketing Campaign Classification

**Goal:** Predict if a client will subscribe to a term deposit using advanced classification techniques and evaluation metrics."""),
    code_cell("""import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay"""),
    markdown_cell("""## 1. Load Data"""),
    code_cell("""df = pd.read_csv('bank.csv', sep=';') # Note the separator usually used in bank datasets
if len(df.columns) == 1: df = pd.read_csv('bank.csv') # Fallback to comma if sep is wrong
df.head()"""),
    markdown_cell("""## 2. Preprocessing"""),
    code_cell("""df = pd.get_dummies(df, drop_first=True)

# Assuming target is 'y_yes' or similar based on pd.get_dummies
target_col = [col for col in df.columns if 'y_' in col or col == 'y']
if target_col:
    y = df[target_col[-1]]
    X = df.drop(target_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('Data split successfully.')
else:
    print('Target column not found.')"""),
    markdown_cell("""## 3. Train Random Forest"""),
    code_cell("""rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)"""),
    markdown_cell("""## 4. Evaluation (ROC-AUC)"""),
    code_cell("""preds = rf.predict(X_test)
probs = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print('ROC AUC Score:', roc_auc_score(y_test, probs))"""),
    code_cell("""RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title('ROC Curve')
plt.show()""")
]
save_notebook('labs/04-Applied-MLOps-Projects/09-bank-marketing/lab-09.ipynb', create_notebook(cells_9))

# Lab 10
cells_10 = [
    markdown_cell("""# Lab 10: Advanced Housing Price Prediction

**Goal:** Use ensemble methods and hyperparameter tuning to get the best possible model for housing prices."""),
    code_cell("""import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score"""),
    markdown_cell("""## 1. Data Loading and Prep"""),
    code_cell("""df = pd.read_csv('Housing.csv')
df = pd.get_dummies(df, drop_first=True)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)"""),
    markdown_cell("""## 2. Gradient Boosting Regressor & Hyperparameter Tuning"""),
    code_cell("""gbr = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}

grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)"""),
    markdown_cell("""## 3. Results"""),
    code_cell("""print('Best Parameters:', grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print('Test R^2 Score:', r2_score(y_test, y_pred))
print('Test RMSE:', mean_squared_error(y_test, y_pred, squared=False))""")
]
save_notebook('labs/04-Applied-MLOps-Projects/10-housing-advanced/lab-10.ipynb', create_notebook(cells_10))

print("All labs generated successfully!")