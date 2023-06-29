import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
dataset = pd.read_csv('datasets/ds_salaries.csv')

X = dataset.drop(['salary', 'salary_currency', 'salary_in_usd'], axis=1)
y = dataset['salary_in_usd']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Lasso regression model
model = Lasso()

# Train the model
model.fit(X_train, y_train)

# Apply the model on the testing set
y_pred = model.predict(X_test)

# Define a threshold to convert predictions into binary values
threshold = 0.5
y_pred_binary = [1 if pred >= threshold else 0 for pred in y_pred]

# Calculate additional performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_binary)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred_binary, average='weighted')
recall = recall_score(y_test, y_pred_binary,average='weighted')
f1 = f1_score(y_test, y_pred_binary,average='weighted')

# Calculate specificity (true negative rate)
tn = len([1 for true, pred in zip(y_test, y_pred_binary) if true == 0 and pred == 0])
fp = len([1 for true, pred in zip(y_test, y_pred_binary) if true == 0 and pred == 1])
# specificity = tn / (tn + fp)

# Print the performance metrics
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
# print("Specificity:", specificity)
print("F1 Score:", f1)
