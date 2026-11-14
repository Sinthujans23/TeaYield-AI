
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "tea_features.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["period"])
df = df.sort_values("period").reset_index(drop=True)

features = [
    "lag_1", "lag_2", "lag_3", "lag_12",
    "month", "year", "rolling_mean_3"
]

X = df[features]
y = df["production_mn_kg"]

# Same chronological split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Load trained Random Forest
rf_model = joblib.load(
    MODEL_DIR / "random_forest.pkl"
)["model"]

rf_predictions = rf_model.predict(X_test)

# ANN with feature scaling
ann_model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="lbfgs",
        alpha=1.0,
        max_iter=2000,
        random_state=42
    )
)

ann_model.fit(X_train, y_train)

ann_predictions = ann_model.predict(X_test)

# Equal-weight hybrid
hybrid_predictions = (
    0.5 * rf_predictions +
    0.5 * ann_predictions
)

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
    evaluate("ANN", y_test, ann_predictions),
    evaluate("RF + ANN Hybrid", y_test, hybrid_predictions)
])

print("\nRF + ANN HYBRID COMPARISON")
print(results.to_string(index=False))

results.to_csv(
    RESULTS_DIR / "hybrid_rf_ann_comparison.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "period": df["period"].iloc[split_index:].values,
    "actual": y_test.values,
    "random_forest": rf_predictions,
    "ann": ann_predictions,
    "rf_ann_hybrid": hybrid_predictions
})

predictions_df.to_csv(
    RESULTS_DIR / "hybrid_rf_ann_predictions.csv",
    index=False
)

joblib.dump(
    {"model": ann_model, "features": features},
    MODEL_DIR / "ann.pkl"
)

print("\nSaved ANN model and hybrid predictions!")
