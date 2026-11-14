from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='TeaYield AI',page_icon='🍃',layout='wide')
st.title('🍃 TeaYield AI')
st.caption('Sri Lankan national monthly tea production — ensemble model benchmarking')
metrics_path=Path('results/metrics.csv'); predictions_path=Path('results/predictions.csv')
if not (metrics_path.exists() and predictions_path.exists()):
    st.info('No trained results found. Add authentic monthly production CSV and run: python -m src.train --data data/raw/tea_production.csv')
    st.stop()
metrics=pd.read_csv(metrics_path)
preds=pd.read_csv(predictions_path,parse_dates=['date'])
st.subheader('Model performance (held-out chronological test)')
st.dataframe(metrics,hide_index=True,use_container_width=True)
st.plotly_chart(px.bar(metrics,x='model',y='RMSE',title='RMSE by model (lower is better)'),use_container_width=True)
st.subheader('Observed vs predicted')
model=st.selectbox('Choose model',[c for c in preds.columns if c not in ('date','actual')])
chart=preds[['date','actual',model]].melt(id_vars='date',var_name='series',value_name='production_mn_kg')
st.plotly_chart(px.line(chart,x='date',y='production_mn_kg',color='series',markers=True,title='One-step held-out test forecasts (million kg)'),use_container_width=True)
st.warning('These are test-set one-step forecasts with earlier observed months, not live next-month forecasts. Do not interpret as future predictions.')
st.download_button('Download test predictions',preds.to_csv(index=False),'teayield_predictions.csv','text/csv')
