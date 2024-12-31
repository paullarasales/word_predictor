import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("small_dataset.csv")

print(data)

data["rooms_per_square_foot"] = data['bedrooms'] / data['square_feet']
data["bathrooms_per_bedroom"] = data['bathrooms'] / data['bedrooms']

data = pd.get_dummies(data, columns=["location"])

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
model.save("small_house_prediction_model.keras")
joblib.dump(scaler, "X_small_scaler.pkl")
joblib.dump(y_scaler, "y_small_scaler.pkl")

city_columns = [col for col in X.columns if col.startswith("location")]
with open("small_city_mappings.pkl", "wb") as f:
    joblib.dump(city_columns, f)

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

corr_matrix = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap with Engineered Features')
plt.show()

# Visualize feature engineering impacts
sns.pairplot(data[["square_feet", "bedrooms", "bathrooms", "rooms_per_square_foot", "bathrooms_per_bedroom", "price"]])
plt.show()


plt.scatter(y_scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred_rescaled)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Residual Plot')
plt.show()

sns.histplot(data['rooms_per_square_foot'], kde=True)
plt.title('Distribution of Rooms Per Square Foot')
plt.show()

sns.histplot(data['bathrooms_per_bedroom'], kde=True)
plt.title('Distribution of Bathrooms Per Bedroom')
plt.show()