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
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("iris_dataset.csv")
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
data['area_approximation'] = data['sepal_length'] * data['sepal_width']
print("New Feature Added")
print(data.head())

# Min-Max Normalization
data['petal_length_normalized'] = (data['petal_length'] - data['petal_length'].min()) / data['petal_length'].max() - data['petal_length'].min()
print(data[['petal_length', 'petal_length_normalized']].head())

X = data.drop('species', axis=1)
y = data['species']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Encoded species", y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("shape 1", X_train.shape[1])
print("Training set size:", X_train.shape)
print("Test size set:", X_test.shape)

model = tf.keras.Sequential([
    Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()
