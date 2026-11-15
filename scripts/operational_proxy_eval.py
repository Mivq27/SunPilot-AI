"""OPTIONAL exploratory archived-forecast comparison (does NOT validate live AI).

This is a research probe, not a release prerequisite. It combines a stitched
historical forecast (past-feature proxy) with a distinct 'previous day 1'
forecast issued ~24 hours before valid time and verifies against ERA5.
STITCHED historical values may NOT reproduce exactly what was available in
one particular archived operational run. For a strict live backtest use
Open-Meteo's *Single Runs API*, the exact issuance timestamp, and model run.

This script never fits or changes the SunPilot model. It may require public
Open-Meteo access and network connectivity; API availability is not assumed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_planner import Location, Forecast, predict_ai_hourly
from src.scheduler import evaluate_frame

ROOT = Path(__file__).resolve().parents[1]
START_HIST = '2024-12-20'
HIST_URL = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
PREVIOUS_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'


def _fetch(url: str, start: str, end: str, fields: str) -> pd.DataFrame:
    params = {
        'latitude':56.1356, 'longitude':13.1309,
        'start_date': start, 'end_date': end,
        'timezone':'UTC', 'hourly':fields,
    }
    response = requests.get(url,params=params,timeout=(15,120))
    response.raise_for_status()
    payload=response.json()
    if payload.get('error'):
        raise RuntimeError(payload.get('reason','Archived forecast API error'))
    hourly=payload.get('hourly') or {}
    names=fields.split(',')
    for name in ['time',*names]:
        if name not in hourly:
            raise ValueError(f'Archive missing required field: {name}; available: {list(hourly)}')
    times=pd.to_datetime(hourly['time'],utc=True)
    result=pd.DataFrame({name:pd.to_numeric(hourly[name],errors='coerce') for name in names},index=times)
    result.index.name='time_utc'
    result=result.sort_index()
    if result.index.has_duplicates:
        raise ValueError('Archive returned duplicate valid times')
    # Preserve elapsed hours for 24/48/168-hour model lags even if an archive
    # leaves a gap. Missing hours remain NaN and are excluded downstream.
    if len(result) and (result.index.to_series().diff().dropna() != pd.Timedelta(hours=1)).any():
        result=result.reindex(pd.date_range(result.index[0],result.index[-1],freq='h',tz='UTC'))
        result.index.name='time_utc'
    return result


def compare_archived_proxy(reference: pd.DataFrame, historical_provider: pd.DataFrame,
                           previous_day1: pd.DataFrame, model, test_year=2025) -> dict:
    """No fitting; only compare forecasts and model against ERA5 target values."""
    loc=Location('Klippan, Sweden',56.1356,13.1309,'Europe/Stockholm')
    forecast=Forecast(loc,loc.timezone,historical_provider,datetime.now(timezone.utc))
    ai, _=predict_ai_hourly(forecast,model)
    aligned=pd.DataFrame({'ai_prediction':ai},index=historical_provider.index)
    aligned=aligned.join(previous_day1,how='inner')
    actual=reference[['shortwave_radiation']].rename(columns={'shortwave_radiation':'actual'})
    aligned=aligned.join(actual,how='inner')
    aligned=aligned.rename(columns={'shortwave_radiation_previous_day1':'baseline'})
    aligned['local_date']=aligned.index.tz_convert('Europe/Stockholm').strftime('%Y-%m-%d')
    aligned['local_hour']=aligned.index.tz_convert('Europe/Stockholm').hour
    aligned=aligned[(aligned.index.year==test_year) &
                    aligned[['actual','baseline','ai_prediction']].notna().all(axis=1)]
    if aligned.empty:
        raise RuntimeError('No matching archived operational samples; check API availability')
    hourly={}
    for name in ('baseline','ai_prediction'):
        hourly[name+'_mae_all_w_m2']=round(float(np.abs(aligned['actual']-aligned[name]).mean()),3)
        daymask=aligned['local_hour'].between(9,18)
        hourly[name+'_mae_08_18_w_m2']=round(float(np.abs(aligned.loc[daymask,'actual']-aligned.loc[daymask,name]).mean()),3)
    # Only fully available dates: incomplete archive days are skipped,
    # and explicit count is reported instead of falsely claiming 365.
    full_days = [g for _,g in aligned.reset_index().groupby('local_date')
                 if set(range(9,19)).issubset(set(g['local_hour'].astype(int)))]
    if not full_days:
        raise RuntimeError('Archive contains no full daytime evaluation days')
    subset=pd.concat(full_days,ignore_index=True)
    daily, outcomes=evaluate_frame(subset,test_year)
    return {
        'status':'EXPLORATORY_STITCHED_ARCHIVE_PROXY_NOT_EXACT_RUN_BACKTEST',
        'hourly_samples':len(aligned),
        'days_with_complete_daytime':len(daily),
        'prediction_mae':hourly,
        'schedule_comparison':outcomes,
        'limitations':[ 
            'Historical stitched feature inputs may not match the exact forecasts visible at a past decision time.',
            'At prediction horizon 1 day the source is previous_day1, not arbitrary up-to-16-day forecast skill.',
            'True live-model validation needs archived Single Runs at fixed issue times; do not claim this is that.',
            'ERA5 verifies model-estimated irradiance; no PV or CO2 measured.',
        ],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',default='2025-01-01')
    parser.add_argument('--end',default='2025-12-31')
    parser.add_argument('--output',default=str(ROOT/'results'/'exploratory_operational_proxy.json'))
    args=parser.parse_args()
    if not (args.start.startswith('2025-') and args.end.startswith('2025-')):
        raise ValueError('This fixed pre-registered probe is limited to the 2025 holdout year')
    reference=pd.read_csv(ROOT/'data'/'klippan_hourly_2022_2025.csv',parse_dates=['time_utc'])
    reference['time_utc']=pd.to_datetime(reference['time_utc'],utc=True)
    reference=reference.set_index('time_utc')
    model=joblib.load(ROOT/'models'/'sunpilot_v1.joblib')['model']
    beginning=min(args.start,START_HIST)
    features=_fetch(HIST_URL,beginning,args.end,'shortwave_radiation,cloud_cover,temperature_2m')
    baseline=_fetch(PREVIOUS_URL,args.start,args.end,'shortwave_radiation_previous_day1')
    result=compare_archived_proxy(reference,features,baseline,model,test_year=2025)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ARCHIVED PROXY (not a full operational backtest):')
    print(json.dumps({'hourly_samples':result['hourly_samples'],
                      'days':result['days_with_complete_daytime'],
                      'mae':result['prediction_mae'],
                      'ai_vs_fixed_pct':result['schedule_comparison']['ai_gain_vs_fixed_percent']},
                     indent=2))
    print('Saved',target)


if __name__=='__main__':
    main()
