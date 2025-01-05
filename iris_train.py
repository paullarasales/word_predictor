import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks #type: ignore
from tensorflow.keras import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Input #type: ignore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical #type: ignore

# Load dataset
data = pd.read_csv("synthetic_iris_dataset.csv")

# Display the dataset structure
print(data.head())
print("Species count:\n", data['species'].value_counts())

# Task 1: Mean of sepal length
mean_sepal_length = data["sepal_length"].mean()
print(f"The mean of sepal_length: {mean_sepal_length:.2f}")

# Task 2: Histogram of sepal length
plt.figure(figsize=(8, 6))
plt.hist(data["sepal_length"], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show(block=False)

# Encode species column to numeric for correlation
label_encoder = LabelEncoder()
data['species_encoded'] = label_encoder.fit_transform(data['species'])

data['ratios'] = data['petal_length'] / data['petal_width']
data["interaction_terms"] = data['sepal_length'] * data['petal_length']
# Drop non-numeric columns for correlation matrix
numeric_data = data.select_dtypes(include=[np.number])  # Only numeric columns
correlation = numeric_data.corr()  # Compute correlation matrix

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Prepare features and target
X = data.drop(['species', 'species_encoded'], axis=1)  # Exclude categorical columns
y = data['species_encoded']  # Encoded target variable

# Scale the features using Min-Max Scaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# One-hot encode the target variable for classification
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the Neural Network model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes in the target
])

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, lr_schedule]
)

# Plot training accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show(block=False)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

# Display classification report
from sklearn.metrics import classification_report
print(classification_report(actual_classes, predicted_classes))
