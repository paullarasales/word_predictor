import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess dataset
data = pd.read_csv("house_data.csv")                

# Feature Engineering
data['rooms_per_square_foot'] = data['bedrooms'] / data['square_feet']
data['bathrooms_per_bedroom'] = data['bathrooms'] / data['bedrooms']

# One-hot encode categorical data
data = pd.get_dummies(data, columns=["location"])

# Features and target
X = data.drop(columns=["price"])
y = data["price"]

# Normalize features
scaler = StandardScaler()   
X_scaled = scaler.fit_transform(X)

# Scale target
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),   
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# Train the model   
model.fit(X_train, y_train, epochs=100, batch_size=16)

# Save the model and scalers
model.save("house_price_prediction_model.keras")
joblib.dump(scaler, "X_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

# Save city columns for future use
city_columns = [col for col in X.columns if col.startswith("location")]
with open("city_mappings.pkl", "wb") as f:
    joblib.dump(city_columns, f)

# Predict using the model
y_pred = model.predict(X_test)

# Inverse scale the predictions
y_pred_rescaled = y_scaler.inverse_transform(y_pred)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred_rescaled)
mse = mean_squared_error(y_test, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_rescaled)

# Print evaluation metrics
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Print first 5 predictions vs true values
for i in range(5):
    print(f"Predicted: {y_pred_rescaled[i][0]:,.2f}, True: {y_scaler.inverse_transform(y_test[i].reshape(1, -1))[0][0]:,.2f}")

# Calculate correlation matrix
corr_matrix = data.corr()

# Plot heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap with Engineered Features')
plt.show()

# Visualize feature engineering impacts
sns.pairplot(data[["square_feet", "bedrooms", "bathrooms", "rooms_per_square_foot", "bathrooms_per_bedroom", "price"]])
plt.show()

# Feature importances using Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train.ravel())

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()

# Residual plot
plt.scatter(y_scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred_rescaled)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Residual Plot')
plt.show()

# Distribution plots of engineered features
sns.histplot(data['rooms_per_square_foot'], kde=True)
plt.title('Distribution of Rooms Per Square Foot')
plt.show()

sns.histplot(data['bathrooms_per_bedroom'], kde=True)
plt.title('Distribution of Bathrooms Per Bedroom')
plt.show()
