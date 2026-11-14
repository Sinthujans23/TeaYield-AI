
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = BASE_DIR / "data" / "raw" / "tea_production.csv"

st.set_page_config(
    page_title="TeaYield AI",
    page_icon="🍃",
    layout="wide"
)

st.title("🍃 TeaYield AI")
st.caption("Sri Lankan Tea Production Forecasting & Model Analytics")

@st.cache_data
def load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)

df = load_csv(DATA_FILE, comment="#")
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Model Comparison",
    "Hybrid Models",
    "Predictions",
    "Next Month Forecast",
    "Walk-Forward Validation"
])

with tab1:
    st.header("Sri Lankan Tea Production Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Months", len(df))
    c2.metric("Average Production", f"{df['value'].mean():.2f} Mn kg")
    c3.metric("Latest Production", f"{df['value'].iloc[-1]:.2f} Mn kg")

    fig = px.line(
        df,
        x="period",
        y="value",
        title="Historical Monthly Tea Production",
        labels={
            "period": "Month",
            "value": "Production (Million kg)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Machine Learning Model Comparison")

    comparison_file = RESULTS_DIR / "all_model_comparison.csv"

    if comparison_file.exists():
        comparison = load_csv(comparison_file)

        st.dataframe(comparison, use_container_width=True)

        fig = px.bar(
            comparison,
            x="Model",
            y="MAE",
            color="Model",
            title="Model MAE Comparison"
        )

        st.plotly_chart(fig, use_container_width=True)

        best = comparison.loc[comparison["MAE"].idxmin()]
        st.success(
            f"Lowest MAE: {best['Model']} ({best['MAE']:.4f})"
        )
    else:
        st.warning("Run compare_all_models.py first.")

with tab3:
    st.header("Hybrid Model Experiments")

    arima_file = RESULTS_DIR / "hybrid_arima_predictions.csv"
    ann_file = RESULTS_DIR / "hybrid_rf_ann_comparison.csv"

    if arima_file.exists():
        arima_df = load_csv(arima_file)

        fig = go.Figure()

        for col in ["actual", "arima", "rf_arima_hybrid"]:
            fig.add_trace(
                go.Scatter(
                    x=arima_df["period"],
                    y=arima_df[col],
                    mode="lines+markers",
                    name=col
                )
            )

        fig.update_layout(
            title="ARIMA vs RF + ARIMA",
            yaxis_title="Production (Million kg)"
        )

        st.plotly_chart(fig, use_container_width=True)

    if ann_file.exists():
        ann_df = load_csv(ann_file)
        st.subheader("RF + ANN Comparison")
        st.dataframe(ann_df, use_container_width=True)

with tab4:
    st.header("Actual vs Predicted Production")

    predictions_file = RESULTS_DIR / "all_model_predictions.csv"

    if predictions_file.exists():
        predictions = load_csv(predictions_file)

        available_models = [
            col for col in predictions.columns
            if col not in ["period", "actual"]
        ]

        selected = st.selectbox(
            "Select Prediction Model",
            available_models
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=predictions["period"],
                y=predictions["actual"],
                name="Actual",
                mode="lines+markers"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=predictions["period"],
                y=predictions[selected],
                name=selected,
                mode="lines+markers"
            )
        )

        fig.update_layout(
            title=f"Actual vs {selected}",
            yaxis_title="Production (Million kg)"
        )

        st.plotly_chart(fig, use_container_width=True)


with tab5:
    st.header("Next Month Tea Production Forecast")

    model_path = BASE_DIR / "models" / "catboost_production.pkl"

    if model_path.exists():

        bundle = joblib.load(model_path)
        model = bundle["model"]
        features = bundle["features"]

        # Historical data
        history = df.sort_values("period").reset_index(drop=True)
        values = history["value"].astype(float)

        latest_month = history["period"].iloc[-1]
        next_month = latest_month + pd.DateOffset(months=1)

        # Create next-month input features
        input_data = pd.DataFrame([{
            "lag_1": values.iloc[-1],
            "lag_2": values.iloc[-2],
            "lag_3": values.iloc[-3],
            "lag_12": values.iloc[-12],
            "month": next_month.month,
            "year": next_month.year,
            "rolling_mean_3": values.iloc[-3:].mean()
        }])

        input_data = input_data[features]

        prediction = float(model.predict(input_data)[0])

        st.metric(
            label=f"Predicted Production – {next_month.strftime('%B %Y')}",
            value=f"{prediction:.3f} Million kg"
        )

        st.info(
            "This is a one-month-ahead forecast based on "
            "historical monthly tea production."
        )

        # Show historical data with forecast
        historical = history[["period", "value"]].copy()

        forecast_point = pd.DataFrame({
            "period": [next_month],
            "value": [prediction]
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=historical["period"],
            y=historical["value"],
            mode="lines+markers",
            name="Historical Production"
        ))

        fig.add_trace(go.Scatter(
            x=[latest_month, next_month],
            y=[values.iloc[-1], prediction],
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash")
        ))

        fig.update_layout(
            title="Historical Production and Next Month Forecast",
            xaxis_title="Month",
            yaxis_title="Production (Million kg)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Details")
        st.dataframe(forecast_point, use_container_width=True)

    else:
        st.warning(
            "Production model not found. "
            "Run python src/forecast_next_month.py first."
        )


with tab6:
    st.header("Walk-Forward Model Validation")

    metrics_file = RESULTS_DIR / "walk_forward_metrics.csv"
    predictions_file = RESULTS_DIR / "walk_forward_predictions.csv"

    if metrics_file.exists() and predictions_file.exists():

        metrics = pd.read_csv(metrics_file)
        predictions = pd.read_csv(
            predictions_file,
            parse_dates=["period"]
        )

        st.subheader("Model Performance")

        st.dataframe(
            metrics.sort_values("MAE"),
            use_container_width=True
        )

        fig_mae = px.bar(
            metrics,
            x="Model",
            y="MAE",
            color="Model",
            title="Walk-Forward MAE Comparison"
        )

        st.plotly_chart(
            fig_mae,
            use_container_width=True
        )

        st.subheader("Actual vs Predicted")

        available_models = [
            col for col in predictions.columns
            if col not in ["period", "actual"]
        ]

        selected_model = st.selectbox(
            "Choose Model",
            available_models,
            key="walk_forward_model"
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=predictions["period"],
            y=predictions["actual"],
            mode="lines+markers",
            name="Actual"
        ))

        fig.add_trace(go.Scatter(
            x=predictions["period"],
            y=predictions[selected_model],
            mode="lines+markers",
            name=selected_model
        ))

        fig.update_layout(
            title=f"Walk-Forward: {selected_model}",
            xaxis_title="Month",
            yaxis_title="Production (Million kg)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        best = metrics.loc[metrics["MAE"].idxmin()]

        st.success(
            f"Lowest MAE: {best['Model']} "
            f"({best['MAE']:.4f} million kg)"
        )

        st.caption(
            "Expanding-window, one-month-ahead evaluation. "
            "20 forecasts from January 2025 to August 2026. "
            "Model selection is preliminary."
        )

    else:
        st.warning(
            "Walk-forward results not found. "
            "Run python src/walk_forward_validation.py first."
        )

st.divider()
st.caption(
    "Academic prototype. Results based on a small historical dataset; "
    "forecasts require further validation."
)
