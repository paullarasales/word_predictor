import os
import joblib

# Constants
MODEL_FILE = 'house_price_model.pkl'

def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found. Please train the model first.")
        exit()
    return joblib.load(MODEL_FILE)

def predict_price(model, square_footage, num_bedrooms):
    features = [[square_footage, num_bedrooms]]
    prediction = model.predict(features)
    return prediction[0]

def main():
    # Ensure the model is ready
    model = load_model()
    
    print("Welcome to the House Price Prediction tool!")
    
    try:
        # Get user input
        square_footage = float(input("Enter square footage of the house: "))
        num_bedrooms = int(input("Enter number of bedrooms: "))
        
        # Perform prediction
        predicted_price = predict_price(model, square_footage, num_bedrooms)
        print(f"The predicted house price is: ${predicted_price:,.2f}")
    except ValueError:
        print("Invalid input. Please enter numerical values for square footage and number of bedrooms.")
    
if __name__ == "__main__":
    main()
