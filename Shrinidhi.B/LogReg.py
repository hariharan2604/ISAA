import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load the dataset from CSV
df = pd.read_csv("../datasets/AQI and Lat Long of Countries.csv")

# Select the relevant columns as features (X) and the target variable (y)
X = df[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
y = df['AQI Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an instance of Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Fit the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Predict the target variable on the scaled test data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

# Calculate precision, recall, F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Calculate AU-ROC score
auc_roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled), multi_class='ovr')

# Display the metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("AU-ROC:", auc_roc)
