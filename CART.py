# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Step 2: Prepare your dataset
# Let's create a simple dataset for binary classification
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

# Step 5: Create and train the Decision Tree model (CART algorithm)
model = DecisionTreeClassifier(criterion='gini', random_state=42)  # 'gini' is the default criterion for CART
model.fit(X_train, Y_train)

# Step 6: Make predictions using the test set
Y_pred = model.predict(X_test)

# Step 7: Evaluate the model's performance
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)

# Print model performance
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Step 8 (Optional): Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(model, feature_names=['X1', 'X2'], class_names=['Class 0', 'Class 1'], filled=True)
plt.show()
