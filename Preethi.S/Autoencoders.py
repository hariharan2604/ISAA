import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load the data from CSV
data = pd.read_csv('../datasets/RICE.csv')

# Drop the 'Collection Type' column
data = data.drop('Collection Type', axis=1)

# Perform one-hot encoding for the 'Location' column
data_encoded = pd.get_dummies(data, columns=['Location'])

# Convert 'Pest Value' into categorical labels using binning
data_encoded['Pest Value'] = pd.cut(data_encoded['Pest Value'], bins=[-np.inf, 0, np.inf], labels=[0, 1])

# Convert the target variable to numeric type
data_encoded['Pest Value'] = data_encoded['Pest Value'].astype(int)

# Preprocess the data
X = data_encoded.drop(['Pest Value', 'Pest Name'], axis=1).values
y = data_encoded['Pest Value'].values.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)

autoencoder = Model(inputs=input_layer, outputs=decoder_layer)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=32, validation_data=(X_test_scaled, X_test_scaled))

# Extract features using the encoder part of the autoencoder
encoder = Model(inputs=input_layer, outputs=encoder_layer)
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

# Train Hariharan.J logistic regression classifier on the encoded features
classifier = LogisticRegression()
classifier.fit(X_train_encoded, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_encoded)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Display the classification metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)