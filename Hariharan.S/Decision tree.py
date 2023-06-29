import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

# Load the dataset
data = pd.read_csv('../datasets/insurance.csv')

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split the data into input features (X) and the target variable (y)
X = data.drop('charges', axis=1)  # Input features
y = data['charges']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Hariharan.J decision tree regressor and fit it to the training data
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate regression performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display regression performance metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)

# Calculate binary values based on median
median = y.median()
y_pred_binary = np.where(y_pred >= median, 1, 0)
y_test_binary = np.where(y_test >= median, 1, 0)

# Calculate classification performance metrics
accuracy = np.mean(y_pred_binary == y_test_binary)
precision = np.sum((y_pred_binary == 1) & (y_test_binary == 1)) / np.sum(y_pred_binary == 1)
recall = np.sum((y_pred_binary == 1) & (y_test_binary == 1)) / np.sum(y_test_binary == 1)
specificity = np.sum((y_pred_binary == 0) & (y_test_binary == 0)) / np.sum(y_test_binary == 0)
f1 = 2 * precision * recall / (precision + recall)
confusion_mat = confusion_matrix(y_test_binary, y_pred_binary)

# Display classification performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1 Score:", f1)
print("Confusion Matrix:\n", confusion_mat)
