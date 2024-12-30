import joblib
import tensorflow as tf
import pandas as pd

# Load the model
model = tf.keras.models.load_model("house_price_prediction_model.keras")

# Load the scalers
X_scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# User input for prediction
square_feet = float(input("Enter square feet: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
city = input("Enter city (New York, Chicago, Los Angeles): ")

# One-hot encode the city
# Create a dictionary of possible cities
city_dict = {"New York": [1, 0, 0], "Chicago": [0, 1, 0], "Los Angeles": [0, 0, 1]}

# Ensure the input city is valid
if city not in city_dict:
    raise ValueError("Invalid city. Please choose from 'New York', 'Chicago', or 'Los Angeles'.")

# Get the one-hot encoded vector for the input city
city_vector = city_dict[city]

# Create a DataFrame for input features
input_data = pd.DataFrame([[square_feet, bedrooms, bathrooms] + city_vector],
                          columns=['square_feet', 'bedrooms', 'bathrooms', 'location_New York', 'location_Chicago', 'location_Los Angeles'])

# Scale the input features using the X_scaler
X_scaled = X_scaler.transform(input_data)

# Predict the price
predicted_price_scaled = model.predict(X_scaled)

# Inverse scale the predicted price
predicted_price = y_scaler.inverse_transform(predicted_price_scaled)

# Display the result
print(f"The predicted house price is: ${predicted_price[0][0]:,.2f}")
