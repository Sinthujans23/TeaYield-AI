
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "tea_features.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["period"])
df = df.sort_values("period").reset_index(drop=True)

features = [
    "lag_1", "lag_2", "lag_3",
    "lag_12", "month", "year",
    "rolling_mean_3"
]

# Initial 36 training months, then forecast each following month
initial_train_size = 36

model_factories = {
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ),
    "LightGBM": lambda: LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=5,
        verbosity=-1,
        random_state=42
    ),
    "CatBoost": lambda: CatBoostRegressor(
        iterations=200,
        depth=3,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=42
    )
}

records = []

for i in range(initial_train_size, len(df)):
    train = df.iloc[:i]
    test = df.iloc[i:i+1]

    actual = float(test["production_mn_kg"].iloc[0])

    row = {
        "period": test["period"].iloc[0],
        "actual": actual,
        "Seasonal Naive": float(test["lag_12"].iloc[0])
    }

    for name, factory in model_factories.items():
        model = factory()

        model.fit(
            train[features],
            train["production_mn_kg"]
        )

        row[name] = float(
            model.predict(test[features])[0]
        )

    records.append(row)
    print(f"Evaluated {test['period'].iloc[0]:%Y-%m}")

predictions = pd.DataFrame(records)

metrics = []

for name in [
    "Seasonal Naive",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost"
]:
    mae = mean_absolute_error(
        predictions["actual"],
        predictions[name]
    )

    rmse = np.sqrt(mean_squared_error(
        predictions["actual"],
        predictions[name]
    ))

    metrics.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    })

metrics_df = pd.DataFrame(metrics).sort_values("MAE")

print("\nWALK-FORWARD VALIDATION RESULTS")
print(metrics_df.to_string(index=False))

predictions.to_csv(
    RESULTS_DIR / "walk_forward_predictions.csv",
    index=False
)

metrics_df.to_csv(
    RESULTS_DIR / "walk_forward_metrics.csv",
    index=False
)

print("\nWalk-forward validation completed!")
