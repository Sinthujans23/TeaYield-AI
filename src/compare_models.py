
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "tea_features.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["period"])
df = df.sort_values("period").reset_index(drop=True)

features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "month",
    "year",
    "rolling_mean_3"
]

X = df[features]
y = df["production_mn_kg"]

# Same chronological split as Random Forest
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Load previously trained Random Forest
rf_bundle = joblib.load(MODEL_DIR / "random_forest.pkl")
rf_model = rf_bundle["model"]

rf_predictions = rf_model.predict(X_test)

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

def evaluate(name, actual, predicted):
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }

results = pd.DataFrame([
    evaluate("Random Forest", y_test, rf_predictions),
    evaluate("XGBoost", y_test, xgb_predictions)
])

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

results.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

joblib.dump(
    {"model": xgb_model, "features": features},
    MODEL_DIR / "xgboost.pkl"
)

predictions = pd.DataFrame({
    "period": df["period"].iloc[split_index:].values,
    "actual": y_test.values,
    "random_forest": rf_predictions,
    "xgboost": xgb_predictions
})

predictions.to_csv(
    RESULTS_DIR / "comparison_predictions.csv",
    index=False
)

print("\nBest model by MAE:")
print(results.loc[results["MAE"].idxmin(), "Model"])

print("\nComparison completed!")
