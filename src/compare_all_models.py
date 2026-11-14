
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "tea_features.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load processed data
df = pd.read_csv(DATA_FILE, parse_dates=["period"])
df = df.sort_values("period").reset_index(drop=True)

features = [
    "lag_1", "lag_2", "lag_3", "lag_12",
    "month", "year", "rolling_mean_3"
]

X = df[features]
y = df["production_mn_kg"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Load previously trained models
rf_model = joblib.load(
    MODEL_DIR / "random_forest.pkl"
)["model"]

xgb_model = joblib.load(
    MODEL_DIR / "xgboost.pkl"
)["model"]

# LightGBM: small dataset-friendly settings
lgbm_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=7,
    min_child_samples=5,
    verbosity=-1,
    random_state=42
)

# CatBoost
catboost_model = CatBoostRegressor(
    iterations=200,
    depth=3,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=False,
    random_seed=42
)

lgbm_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "CatBoost": catboost_model
}

results = []
predictions_df = pd.DataFrame({
    "period": df["period"].iloc[split_index:].values,
    "actual": y_test.values
})

for name, model in models.items():
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions)
    })

    predictions_df[name] = predictions

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\nALL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    RESULTS_DIR / "all_model_comparison.csv",
    index=False
)

predictions_df.to_csv(
    RESULTS_DIR / "all_model_predictions.csv",
    index=False
)

joblib.dump(
    {"model": lgbm_model, "features": features},
    MODEL_DIR / "lightgbm.pkl"
)

joblib.dump(
    {"model": catboost_model, "features": features},
    MODEL_DIR / "catboost.pkl"
)

print("\nBest model by MAE:")
print(results_df.iloc[0]["Model"])

print("\nAll model comparison completed!")
