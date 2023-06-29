import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Read the CSV file
df = pd.read_csv('../datasets/RICE.csv')

# Step 2: Preprocess the Data
target_columns = ['Pest Name', 'Location']  # Update with your target column names
feature_columns = [col for col in df.columns if col not in target_columns]
X = df[feature_columns]  # Features
y = df['Pest Name']  # Target variable

# Encode the target variable using label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encoding for categorical variables
categorical_cols = ['Collection Type']  # Update with your categorical column names
X = pd.get_dummies(X, columns=categorical_cols)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 4: Scale the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

# Step 6: Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred.argmax(axis=1))
y_test_labels = le.inverse_transform(y_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test_labels, y_pred_labels)
confusion_mat = confusion_matrix(y_test_labels, y_pred_labels)
precision = precision_score(y_test_labels, y_pred_labels, average='macro')
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')
auc_roc = roc_auc_score(to_categorical(y_test, num_classes=len(le.classes_)), y_pred, multi_class='ovr')

# Step 9: Print the evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", auc_roc)
