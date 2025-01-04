import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Input #type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical #type: ignore

data = pd.read_csv("synthetic_iris_dataset.csv")
print(data.head())

# Task one
mean_sepal_length = data["sepal_length"].mean()
print(f"The mean of sepal_length {mean_sepal_length:.2f}")

# Task two
plt.figure(figsize=(8, 6))
plt.hist(data["sepal_length"], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show(block=False)

# Feature Engineering
# data['area_approximation'] = data['sepal_length'] * data['sepal_width']
# print("New Feature Added")
# print(data.head())  

#Min-Max Normalization
data['petal_length_normalized'] = (data['petal_length'] - data['petal_length'].min()) / data['petal_length'].max() - data['petal_length'].min()
print(data[['petal_length', 'petal_length_normalized']].head())

X = data.drop('species', axis=1)
y = data['species']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Encoded species", y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print("shape 1", X_train.shape[1])
print("Training set size:", X_train.shape)
print("Test size set:", X_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train)
print(y_test)

model = tf.keras.Sequential([
    Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show(block=False)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

predictions = model.predict(X_test)
print(f"Predictions: {predictions}")

predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)
print(classification_report(actual_classes, predicted_classes))
print("Predicted classes:", predicted_classes)
print("Actual classes:", actual_classes)

conf_matrix = confusion_matrix(actual_classes, predicted_classes)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")