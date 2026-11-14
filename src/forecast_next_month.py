
import pandas as pd
import joblib
from pathlib import Path
from catboost import CatBoostRegressor

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "raw" / "tea_production.csv"
FEATURE_FILE = BASE_DIR / "data" / "processed" / "tea_features.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load historical data
history = pd.read_csv(DATA_FILE, comment="#")
history = history[["period", "value"]].copy()

history["period"] = pd.to_datetime(history["period"])
history = history.sort_values("period").reset_index(drop=True)

# Load prepared training features
train_df = pd.read_csv(FEATURE_FILE)

features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "month",
    "year",
    "rolling_mean_3"
]

X = train_df[features]
y = train_df["production_mn_kg"]

# Train production model on all available data
model = CatBoostRegressor(
    iterations=200,
    depth=3,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=False,
    random_seed=42
)

model.fit(X, y)

# Next month
next_month = history["period"].iloc[-1] + pd.DateOffset(months=1)

values = history["value"].astype(float)

next_features = pd.DataFrame([{
    "lag_1": values.iloc[-1],
    "lag_2": values.iloc[-2],
    "lag_3": values.iloc[-3],
    "lag_12": values.iloc[-12],
    "month": next_month.month,
    "year": next_month.year,
    "rolling_mean_3": values.iloc[-3:].mean()
}])

prediction = float(model.predict(next_features)[0])

print("\nTEAYIELD AI FORECAST")
print("Forecast month:", next_month.strftime("%Y-%m"))
print(f"Predicted production: {prediction:.3f} million kg")

joblib.dump(
    {"model": model, "features": features},
    MODEL_DIR / "catboost_production.pkl"
)

print("\nProduction model saved successfully!")
