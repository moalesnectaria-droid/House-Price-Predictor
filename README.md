# House Price Predictor

## Overview

This project applies Machine Learning regression algorithms to estimate California house prices using demographic and housing-related features.

The project compares multiple regression models, evaluates their performance using standard regression metrics, and automatically selects the best-performing model.

---

## Dataset

The project uses the **California Housing Dataset** provided by Scikit-Learn.

Features include:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

Target:

- Median House Value

---

## Machine Learning Models

The following regression algorithms were evaluated:

- Linear Regression
- Random Forest Regressor

The model with the highest **R² Score** is automatically selected and saved.

---

## Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Example:

```
Best Model:
Random Forest

MAE : 0.328
RMSE: 0.506
R²  : 0.805
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib

---

## Project Structure

```
House-Price-Predictor/

│
├── data/
├── model/
│   ├── house_price_model.pkl
│   └── features.pkl
│
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Usage

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train_model.py
```

Predict a house price

```bash
python predict.py
```

---

## Future Improvements

Possible future enhancements include:

- Gradient Boosting Regressor
- XGBoost
- Hyperparameter tuning
- Feature engineering
- Model deployment using FastAPI
- Interactive web interface

---

## Disclaimer

This project is intended for educational purposes and demonstrates Machine Learning regression techniques using a public dataset.