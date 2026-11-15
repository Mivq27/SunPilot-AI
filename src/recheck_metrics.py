"""Recalculate MAE from FROZEN predictions with consistent hour-end semantics.

This does not train, refit or tune the model, and never leaks reference values
into predictions. An actual 08:00–18:00 task window comprises end-hour
labels 09:00–18:00, not 08:00–17:00.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def evaluate_saved_predictions(frame: pd.DataFrame) -> dict:
    actual = frame['actual'].to_numpy(dtype=float)
    baseline = frame['baseline'].to_numpy(dtype=float)
    ai = np.maximum(frame['ai_prediction'].to_numpy(dtype=float), 0)
    mask = frame['local_hour'].between(9, 18).to_numpy(dtype=bool)
    if not mask.any() or not (np.isfinite(actual).all() and np.isfinite(baseline).all() and np.isfinite(ai).all()):
        raise ValueError('No valid hourly benchmark data')
    def score(predicted):
        return {
            'mae_all_w_m2': round(float(np.abs(actual - predicted).mean()), 3),
            'mae_08_18_w_m2': round(float(np.abs(actual[mask] - predicted[mask]).mean()), 3),
        }
    return {'baseline':score(baseline), 'ai':score(ai)}


def main():
    years={}
    for year,file_name in [(2024,'validation_predictions_2024.csv'),(2025,'test_predictions_2025.csv')]:
        file_path = RESULTS / file_name
        if not file_path.exists():
            raise FileNotFoundError(f'Missing {file_path}. Run src/train.py with the frozen original model config.')
        frame=pd.read_csv(file_path)
        # The original model was split by UTC year, not local year. Retain
        # every saved UTC-year row so the all-hours MAE is exactly comparable
        # with the frozen v1.0 test, even at New Year's local offset.
        if frame.empty:
            raise RuntimeError(f'No observations for year {year}')
        years['validation_2024' if year == 2024 else 'test_2025'] = evaluate_saved_predictions(frame)
    results={
        **years,
        'evaluation_version':2,
        'hour_label_semantics':'hour_end',
        'operating_window':'08:00–18:00 actual execution time (end-hour labels 09:00–18:00)',
        'reference':'ERA5 hourly reanalysis, not measured solar output',
    }
    (RESULTS/'metrics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    for yr, m in years.items():
        print(yr,'AI:',m['ai'],'yesterday:',m['baseline'])
    print('Saved corrected daytime MAE and unchanged all-hours MAE to',RESULTS/'metrics.json')


if __name__=='__main__':
    main()
