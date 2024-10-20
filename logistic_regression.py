# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Prepare your dataset
# Let's create a simple dataset for binary classification
data = {
    'X1': [2, 4, 6, 8, 10, 12],
    'X2': [1, 3, 5, 7, 9, 11],
    'X3': [5, 10, 15, 20, 25, 30],
    'Y': [0, 0, 1, 1, 1, 1]  # Binary classification
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Split the dataset into features (X) and target variable (Y)
X = df[['X1', 'X2', 'X3']]  # Independent variables
Y = df['Y']  # Dependent variable (binary: 0 or 1)

# Step 4: Check class distribution in the entire dataset (Y)
print("Class distribution in the entire dataset (Y):", np.bincount(Y))

# Step 5: Split data into training and testing sets (80% train, 20% test) with stratification
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Step 6: Check class distribution in Y_train after split
print("Class distribution in Y_train:", np.bincount(Y_train))

# Step 7: Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 8: Make predictions using the test set
Y_pred = model.predict(X_test)

# Step 9: Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)

# Print model performance
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Print the model's coefficients (weights for each feature)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
