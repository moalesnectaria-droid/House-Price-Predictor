import joblib
import pandas as pd

print("=== House Price Predictor ===\n")

model = joblib.load("model/house_price_model.pkl")

feature_names = joblib.load("model/features.pkl")

data = {}

questions = {
    "MedInc": "Median income: ",
    "HouseAge": "House age: ",
    "AveRooms": "Average rooms: ",
    "AveBedrms": "Average bedrooms: ",
    "Population": "Population: ",
    "AveOccup": "Average occupancy: ",
    "Latitude": "Latitude: ",
    "Longitude": "Longitude: "
}

for feature in feature_names:

    value = float(input(questions[feature]))

    data[feature] = value

df = pd.DataFrame([data])

prediction = model.predict(df)[0]

print("\n======================")
print("Prediction")
print("======================")

print(f"Estimated house price: ${prediction * 100000:.2f}")