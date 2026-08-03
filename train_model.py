import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Loading California Housing Dataset...")

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

print(df.head())

X = df.drop("Price", axis=1)

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
}

best_model = None
best_name = None
best_r2 = -999

print("\n======================")
print("MODEL COMPARISON")
print("======================")

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"\n{name}")
    print("-"*30)

    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    if r2 > best_r2:

        best_r2 = r2
        best_model = model
        best_name = name


print("\n======================")
print("BEST MODEL")
print("======================")

print(best_name)
print(f"R² Score: {best_r2:.3f}")

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    best_model,
    "model/house_price_model.pkl"
)

joblib.dump(
    list(X.columns),
    "model/features.pkl"
)

print("\nModel saved successfully!")