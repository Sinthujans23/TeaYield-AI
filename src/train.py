import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
from .data import load_data, features


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {'MAE':float(mean_absolute_error(y,pred)), 'RMSE':float(np.sqrt(mean_squared_error(y,pred))), 'sMAPE':float(np.mean(200*np.abs(y-pred)/(np.abs(y)+np.abs(pred)+1e-8)))}


def train(data_path):
    raw = load_data(data_path)
    df, cols = features(raw)
    ntest = max(12, int(np.ceil(len(df)*0.2)))
    train_df, test_df = df.iloc[:-ntest], df.iloc[-ntest:]
    if len(train_df) < 18:
        raise ValueError('Not enough training rows after lags and test split; get more history')
    # Validation for blend weights: last 20% of train (at least 6 rows).
    nval = max(6, int(len(train_df)*0.2))
    fit_df, val_df = train_df.iloc[:-nval], train_df.iloc[-nval:]
    if len(fit_df) < 12:
        raise ValueError('Need longer history for separate train/validation/test splits')
    Xtr,ytr=train_df[cols],train_df.production_mn_kg
    Xte,yte=test_df[cols],test_df.production_mn_kg
    Xfit,yfit=fit_df[cols],fit_df.production_mn_kg
    Xval,yval=val_df[cols],val_df.production_mn_kg
    factories={
      'Random Forest':lambda:RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1),
      'XGBoost':lambda:XGBRegressor(n_estimators=250,max_depth=3,learning_rate=0.04,subsample=0.85,random_state=42,n_jobs=2),
      'LightGBM':lambda:LGBMRegressor(n_estimators=200,max_depth=3,num_leaves=7,min_child_samples=5,verbosity=-1,random_state=42),
      'CatBoost':lambda:CatBoostRegressor(iterations=250,depth=3,learning_rate=0.04,verbose=False,random_seed=42),
    }
    preds={}; fitted={}
    for name,create in factories.items():
        model=create();model.fit(Xtr,ytr)
        preds[name]=np.asarray(model.predict(Xte));fitted[name]=model
    # RF+ANN convex blend: tune on validation, refit on all train, test once.
    def ann():return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(16,),alpha=10,max_iter=2000,early_stopping=False,random_state=42))
    rf_val=factories['Random Forest']();rf_val.fit(Xfit,yfit)
    ann_val=ann();ann_val.fit(Xfit,yfit)
    a,b=rf_val.predict(Xval),ann_val.predict(Xval)
    weight=min(np.linspace(0,1,11),key=lambda w:mean_absolute_error(yval,w*a+(1-w)*b))
    ann_full=ann();ann_full.fit(Xtr,ytr)
    preds['RF + ANN']=weight*preds['Random Forest']+(1-weight)*ann_full.predict(Xte)
    # ARIMA + RF residual correction: each test month fits ARIMA only on prior observed months.
    # Residual learner uses lagged ARIMA one-step errors, estimated in-sample on training period.
    # Fit ARIMA once on training and use its one-step in-sample predictions (after warmup).
    series=raw.set_index('date').production_mn_kg
    train_end=train_df.date.iloc[-1]
    history=series.loc[:train_end]
    try:
        base=ARIMA(history,order=(1,1,1)).fit()
        in_pred=base.predict(start=12,end=len(history)-1)
        residuals=history.iloc[12:].to_numpy()-np.asarray(in_pred)
        residual_frame=train_df.copy()
        residual_frame['arima_residual']=np.nan
        residual_map=dict(zip(history.iloc[12:].index,residuals))
        residual_frame['arima_residual']=residual_frame.date.map(residual_map)
        residual_frame=residual_frame.dropna(subset=['arima_residual'])
        residual_rf=RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=42,n_jobs=-1)
        residual_rf.fit(residual_frame[cols],residual_frame.arima_residual)
        hybrid=[]
        for _,row in test_df.iterrows():
            # One-step forecast: past observed production becomes available after each month.
            past=series.loc[series.index<row.date]
            arima_pred=float(ARIMA(past,order=(1,1,1)).fit().forecast(1).iloc[0])
            correction=float(residual_rf.predict(row[cols].to_frame().T)[0])
            hybrid.append(arima_pred+correction)
        preds['RF + ARIMA']=np.array(hybrid)
    except Exception as exc:
        print('RF + ARIMA unavailable:',exc)
    out=Path('results');out.mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    for name,model in fitted.items():
        joblib.dump({'model':model,'features':cols},Path('models')/(name.lower().replace(' ','_')+'.joblib'))
    result=pd.DataFrame({'date':test_df.date.dt.strftime('%Y-%m-%d'),'actual':yte.to_numpy()})
    scores=[]
    for name,pred in preds.items():
        result[name]=pred
        scores.append({'model':name,**metrics(yte,pred)})
    result.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(scores).sort_values('RMSE').to_csv(out/'metrics.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'train_rows':len(train_df),'test_rows':len(test_df),'first_month':str(raw.date.min().date()),'last_month':str(raw.date.max().date()),'rf_ann_weight':float(weight),'evaluation':'one-step with observed prior months'},indent=2))
    print(pd.DataFrame(scores).sort_values('RMSE').to_string(index=False))
    print('Saved results/metrics.csv and results/predictions.csv')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/raw/tea_production.csv')
    args=parser.parse_args();train(args.data)
