
import pandas as pd
import numpy as np
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "tea_production.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load actual monthly data
df = pd.read_csv(DATA_FILE, comment="#")
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period").reset_index(drop=True)

series = pd.Series(
    df["value"].astype(float).values,
    index=df["period"]
).asfreq("MS")

# Use the same 12-month test period as previous models
train = series.iloc[:-12]
test = series.iloc[-12:]

# Walk-forward one-month forecasting
arima_preds = []
hybrid_preds = []

history = train.copy()

for date, actual in test.items():

    # ARIMA trained only on available past values
    arima_model = ARIMA(
        history,
        order=(2, 0, 1),
        trend="c"
    ).fit()

    arima_pred = float(
        arima_model.forecast(steps=1).iloc[0]
    )

    # Training residuals: actual minus ARIMA fitted value
    residuals = (
        history - arima_model.fittedvalues
    ).dropna()

    residual_df = pd.DataFrame({
        "residual": residuals
    })

    # Use only previously observed residuals
    residual_df["lag_1"] = residual_df["residual"].shift(1)
    residual_df["lag_2"] = residual_df["residual"].shift(2)
    residual_df["lag_3"] = residual_df["residual"].shift(3)

    residual_df = residual_df.dropna()

    if len(residual_df) >= 10:
        X_res = residual_df[
            ["lag_1", "lag_2", "lag_3"]
        ]
        y_res = residual_df["residual"]

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_res, y_res)

        next_features = pd.DataFrame([{
            "lag_1": residuals.iloc[-1],
            "lag_2": residuals.iloc[-2],
            "lag_3": residuals.iloc[-3]
        }])

        correction = float(
            rf.predict(next_features)[0]
        )
    else:
        correction = 0.0

    hybrid_pred = arima_pred + correction

    arima_preds.append(arima_pred)
    hybrid_preds.append(hybrid_pred)

    # Reveal actual value only after prediction
    history.loc[date] = actual

def evaluate(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        )
    }

arima_metrics = evaluate(test.values, arima_preds)
hybrid_metrics = evaluate(test.values, hybrid_preds)

print("\nARIMA Performance:", arima_metrics)
print("RF + ARIMA Hybrid Performance:", hybrid_metrics)

results = pd.DataFrame({
    "period": test.index,
    "actual": test.values,
    "arima": arima_preds,
    "rf_arima_hybrid": hybrid_preds
})

results.to_csv(
    RESULTS_DIR / "hybrid_arima_predictions.csv",
    index=False
)

print("\nSaved: results/hybrid_arima_predictions.csv")
