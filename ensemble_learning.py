# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Prepare your dataset
# For simplicity, let's create a small dataset for binary classification
data = {
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    'Y':  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # Binary classification
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Split the dataset into features (X) and target variable (Y)
X = df[['X1', 'X2']]  # Independent variables
Y = df['Y']           # Dependent variable (binary: 0 or 1)

# Step 4: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Step 6: Train the AdaBoost Classifier
# Using DecisionTreeClassifier as the base estimator
base_estimator = DecisionTreeClassifier(max_depth=1)  # Stumps
ada_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=42)
ada_model.fit(X_train, Y_train)

# Step 7: Make predictions using both models
rf_predictions = rf_model.predict(X_test)
ada_predictions = ada_model.predict(X_test)

# Step 8: Evaluate the models' performance
# Random Forest evaluation
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(Y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_test, rf_predictions))
print("Classification Report:\n", classification_report(Y_test, rf_predictions))

# AdaBoost evaluation
print("\nAdaBoost Classifier:")
print("Accuracy:", accuracy_score(Y_test, ada_predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_test, ada_predictions))
print("Classification Report:\n", classification_report(Y_test, ada_predictions))

# Optional: Visualize feature importances from Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [f'X{i+1}' for i in indices])
plt.xlim([-1, X.shape[1]])
plt.show()
