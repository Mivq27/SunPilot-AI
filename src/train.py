"""Original frozen 2022–2023 ERA5 model with corrected daylight evaluation mask.

Training model and hyperparameters match v1.0. No tuning based on test results.
The only scientific correction is daytime MAE computed for actual 08–18
intervals (radiation end-hour labels 09–18) instead of 08–17.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'klippan_hourly_2022_2025.csv'
MODEL_PATH = ROOT / 'models' / 'sunpilot_v1.joblib'
RESULTS_DIR = ROOT / 'results'

FEATURES = [
    'hour_sin', 'hour_cos', 'season_sin', 'season_cos',
    'radiation_lag_24', 'radiation_lag_48', 'radiation_lag_168',
    'cloud_lag_24', 'temperature_lag_24',
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
    df = df.sort_values('time_utc').reset_index(drop=True)
    local = df['time_utc'].dt.tz_convert('Europe/Stockholm')
    df['local_date'] = local.dt.strftime('%Y-%m-%d')
    df['local_hour'] = local.dt.hour
    hour = local.dt.hour
    day = local.dt.day_of_year
    df['hour_sin'] = np.sin(2*np.pi*hour/24)
    df['hour_cos'] = np.cos(2*np.pi*hour/24)
    df['season_sin'] = np.sin(2*np.pi*day/365.25)
    df['season_cos'] = np.cos(2*np.pi*day/365.25)
    for lag in (24, 48, 168):
        df[f'radiation_lag_{lag}'] = df['shortwave_radiation'].shift(lag)
    df['cloud_lag_24'] = df['cloud_cover'].shift(24)
    df['temperature_lag_24'] = df['temperature_2m'].shift(24)
    return df.dropna(subset=FEATURES + ['shortwave_radiation']).reset_index(drop=True)


def evaluate(name, df, predictions):
    actual = df['shortwave_radiation'].to_numpy(dtype=float)
    predicted = np.maximum(np.asarray(predictions,dtype=float),0)
    mae_all = mean_absolute_error(actual,predicted)
    # Open-Meteo hour-end labels 09–18 describe real 08:00–18:00.
    mask = df['local_hour'].between(9,18).to_numpy()
    mae_window = mean_absolute_error(actual[mask],predicted[mask])
    print(f'{name:<22} MAE all: {mae_all:8.2f} | MAE 08–18 actual: {mae_window:8.2f}')
    return {'mae_all_w_m2':round(float(mae_all),3),
            'mae_08_18_w_m2':round(float(mae_window),3)}


def save_predictions(df,ai_predictions,filename):
    output=df[['time_utc','local_date','local_hour','shortwave_radiation',
               'radiation_lag_24']].copy()
    output=output.rename(columns={'shortwave_radiation':'actual',
                                   'radiation_lag_24':'baseline'})
    output['ai_prediction']=np.maximum(ai_predictions,0)
    output.to_csv(RESULTS_DIR / filename,index=False)


def main():
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    df=load_data()
    years=df['time_utc'].dt.year
    train=df[years.isin([2022,2023])].copy()
    val=df[years==2024].copy()
    test=df[years==2025].copy()
    if min(len(train),len(val),len(test)) == 0:
        raise ValueError('One year split is empty')
    model=HistGradientBoostingRegressor(
        max_iter=180,learning_rate=0.07,max_leaf_nodes=31,
        min_samples_leaf=40,l2_regularization=1.0,random_state=42,
    )
    model.fit(train[FEATURES],train['shortwave_radiation'])
    pred_val=np.maximum(model.predict(val[FEATURES]),0)
    pred_test=np.maximum(model.predict(test[FEATURES]),0)
    metrics={
        'validation_2024':{
            'baseline':evaluate('Yesterday baseline 2024',val,val['radiation_lag_24'].to_numpy()),
            'ai':evaluate('SunPilot AI 2024',val,pred_val),
        },
        'test_2025':{
            'baseline':evaluate('Yesterday baseline 2025',test,test['radiation_lag_24'].to_numpy()),
            'ai':evaluate('SunPilot AI 2025',test,pred_test),
        },
        'evaluation_version':2,
        'hour_label_semantics':'hour_end',
        'operating_window':'08:00–18:00 actual execution time (end-hour labels 09:00–18:00)',
        'reference':'ERA5 hourly reanalysis, not measured solar output',
    }
    joblib.dump({'model':model,'features':FEATURES,'training_years':[2022,2023]},MODEL_PATH)
    save_predictions(val,pred_val,'validation_predictions_2024.csv')
    save_predictions(test,pred_test,'test_predictions_2025.csv')
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved model, corrected MAE, and 2024/2025 predictions. No test-time tuning performed.')


if __name__=='__main__':
    main()
