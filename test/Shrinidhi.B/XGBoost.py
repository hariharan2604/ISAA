import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset from CSV
df = pd.read_csv("../AQI and Lat Long of Countries.csv")

# Select the relevant columns as features (X) and the target variable (y)
X = df[['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
y = df['AQI Category']

# Encode the class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create an instance of XGBoost model
model = xgb.XGBClassifier()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred = model.predict(X_test)

# Decode the predicted labels
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

# Calculate precision, recall, F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Calculate AU-ROC score
auc_roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

# Display the metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("AU-ROC:", auc_roc)
