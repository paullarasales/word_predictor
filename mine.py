import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 

data = pd.read_csv("small_dataset.csv")

data = pd.get_dummies(data, columns=["location"])
print(data)

X = data.drop(columns=["price"])
print(X)
y = data["price"]
print(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("X_Scaled")
print(X_scaled)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
print("Reshape")
print(y_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])
print(model)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
print(model)

model.fit(X_train, y_train, epochs=20, batch_size=16)
y_pred = model.predict(X_test)

y_pred_rescaled = y_scaler.inverse_transform(y_pred)

mae = mean_absolute_error(y_test, y_pred_rescaled)
mse = mean_squared_error(y_test, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_rescaled)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

for i in range(5):
    print(f"Predicted: {y_pred_rescaled[i][0]:,.2f}, True: {y_scaler.inverse_transform(y_test[i].reshape(1, -1))[0][0]:,.2f}")