# TeaYield AI — Sri Lanka Monthly Tea Production Forecasting

An educational, reproducible machine-learning project comparing Random Forest, XGBoost, LightGBM and CatBoost with ARIMA+RF and RF+ANN hybrid experiments. Includes a Streamlit dashboard.

**Data is NOT included.** Download authentic monthly Sri Lankan tea production records from https://macrolab.lk/data/agriculture-and-fisheries/tea/ or https://srilankateaboard.lk/publications/ . Check usage rights and provenance. Convert to `data/raw/tea_production.csv` with columns `date,production_mn_kg` (e.g. `2024-01-01,20.5`). Do not use invented observations. More history is better; 68 rows is small for ANN.

## Quick start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train --data data/raw/tea_production.csv
streamlit run dashboard/app.py
```

Training uses chronological train/test splitting (last 20%, at least 12 rows), strictly lagged features, and an expanding-window evaluation for the held-out segment (one-step forecasts with observed prior months). No shuffled split. For production use, evaluate rolling multi-step forecasts too. Hybrid RF+ARIMA models ARIMA residuals using strictly past observations and sequential walk-forward forecasting. RF+ANN is a simple stacking-style blend with ANN trained on the training subset only; weights selected on a separate validation subset, never test. All results are experimental, not guaranteed improvements.

The dashboard loads results and displays the evaluation charts. Without a dataset/training it explains the setup instead of inventing results.

## Scope
National monthly **tea production** (million kg), not agronomic yield per hectare. Weather effects are not estimated until correctly dated weather data is obtained and joined without future leakage.
