# Step 1: Import necessary libraries
# pip install scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Prepare your dataset
# For demonstration, let's create a simple dataset
# Suppose we have 3 independent variables (X1, X2, X3) and 1 dependent variable (Y)
data = {
    'X1': [2, 4, 6, 8, 10, 12],
    'X2': [1, 3, 5, 7, 9, 11],
    'X3': [5, 10, 15, 20, 25, 30],
    'Y': [10, 20, 30, 40, 50, 60]
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Split the dataset into features (X) and target variable (Y)
X = df[['X1', 'X2', 'X3']]  # Independent variables
Y = df['Y']  # Dependent variable

# Step 4: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 5: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 6: Make predictions using the test set
Y_pred = model.predict(X_test)

# Step 7: Evaluate the model's performance
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Print model performance
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Print the model's coefficients (weights for each feature)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
