import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

model = tf.keras.models.load_model("small_house_prediction_model.keras")
X_scaler = joblib.load("X_small_scaler.pkl")
y_scaler = joblib.load("y_small_scaler.pkl")

with open("small_city_mappings.pkl", "rb") as f:
    city_columns = joblib.load(f)

def get_user_input():
    square_feet = float(input("Enter square meters: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathroooms: "))
    location = input("Enter location: ")
    return square_feet, bedrooms, bathrooms, location

def preprocess_input(square_feet, bedrooms, bathrooms, location):
    input_data = pd.DataFrame([{
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "rooms_per_square_foot": bedrooms / square_feet,
        "bathrooms_per_bedroom": bathrooms / bedrooms if bedrooms != 0 else 0
    }])

    for loc in city_columns: 
        input_data[loc] = 1 if loc == f"location_{location}" else 0


    missing_cols = set(city_columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    X_input_scaled = X_scaler.transform(input_data)
    print(X_input_scaled)
    return X_input_scaled


def predict_price(X_input_scaled):
    predicted_price_scaled = model.predict(X_input_scaled)
    predicted_price = y_scaler.inverse_transform(predicted_price_scaled)
    return predicted_price[0][0]

def main():
    print("House Price Prediction!!")
    while True:
        square_feet, bedrooms, bathrooms, location = get_user_input()
        try:
            X_input_scaled = preprocess_input(square_feet, bedrooms, bathrooms, location)
            price = predict_price(X_input_scaled)
            print(f"The predicted price is: {price:,.2f}")
        except Exception as e:
            print(f"Error:  {e}")

        another = input("Do you want to predict another house price? (yes/no): ").lower()
        if another != 'yes':
            print("Thank you for using the House Price Prediction. Goodbye!")
            break

if __name__ == "__main__":
    main()
