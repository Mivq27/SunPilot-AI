"""Leakage-aware offline scheduling backtest with corrected hourly alignment.

Open-Meteo shortwave_radiation tagged 09:00 means [08:00,09:00).
A two-hour job starting at 12:00 therefore consumes labels 13:00,14:00.
The original v1.0 backtest used labels 12:00,13:00: a one-hour error.
This backtest uses 2024 validation and untouched 2025 predictions, and
reports the simple 'fixed noon' competitor rather than hiding it.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Support both `python src/scheduler.py` and `python -m src.scheduler`.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.solar_time import LOCAL_START, LOCAL_END, historical_day
from src.pv_scenario import hypothetical_self_use_kwh

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
TASK_DURATION = 2  # hours; original documented 2024/2025 benchmark
FIXED_START = 12   # start of actual job, in local time
BOOTSTRAP_REPS = 5000


def choose_best(values: np.ndarray, *, duration: int = TASK_DURATION) -> int:
    """Return actual job start hour from 08:00, not end-label hour."""
    values = np.asarray(values, dtype=float)
    if values.shape != (LOCAL_END - LOCAL_START,) or not np.isfinite(values).all():
        raise ValueError('Expected ten finite hourly values (08:00–18:00 intervals)')
    if not 1 <= duration <= len(values):
        raise ValueError('Duration is outside the historical 08:00–18:00 window')
    totals = np.array([values[i:i+duration].sum() for i in range(len(values)-duration+1)])
    tied = np.flatnonzero(np.isclose(totals, totals.max(), atol=1e-8))
    return LOCAL_START + int(min(tied, key=lambda i: (abs(LOCAL_START+int(i)-12), int(i))))


def solar_irradiation(values: np.ndarray, start_hour: int, *, duration: int = TASK_DURATION) -> float:
    offset = start_hour - LOCAL_START
    if not 0 <= offset <= (LOCAL_END-LOCAL_START-duration):
        raise ValueError('Start time is outside the historical task window')
    return float(np.asarray(values, dtype=float)[offset:offset+duration].sum())


def bootstrap_relative_gain(a: np.ndarray, b: np.ndarray, *, seed: int = 426,
                            repetitions: int = BOOTSTRAP_REPS) -> list[float]:
    """Paired day-level bootstrap interval for aggregate relative gain (%).

    A descriptive uncertainty interval; days can be seasonally correlated,
    so do not treat it as a formal independent-sample hypothesis test.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if len(aa) != len(bb) or len(aa) == 0 or np.any(~np.isfinite(aa+bb)):
        raise ValueError('Invalid paired bootstrap vectors')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(aa), size=(repetitions, len(aa)))
    aa_sum = aa[draws].sum(axis=1)
    bb_sum = bb[draws].sum(axis=1)
    allowed = bb_sum > 0
    if not np.any(allowed):
        raise ValueError('Baseline total must be positive')
    gain = 100 * (aa_sum[allowed] / bb_sum[allowed] - 1)
    return [round(float(np.percentile(gain, p)), 3) for p in (2.5, 97.5)]


def classify_days(df: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    diff = df[a].to_numpy(dtype=float) - df[b].to_numpy(dtype=float)
    return {
        'ai_better_pct': round(float(100 * (diff > 1e-8).mean()), 3),
        'ai_equal_pct': round(float(100 * (np.abs(diff) <= 1e-8).mean()), 3),
        'ai_worse_pct': round(float(100 * (diff < -1e-8).mean()), 3),
    }


def evaluate_frame(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    """Pure evaluator for saved predictions; no fitting, no test-time tuning."""
    columns = {'local_date', 'local_hour', 'actual', 'baseline', 'ai_prediction'}
    if not columns.issubset(df):
        raise ValueError(f'Missing prediction fields: {sorted(columns - set(df))}')
    records = []
    for date, group in df.groupby('local_date', sort=True):
        if not str(date).startswith(f'{year}-'):
            continue
        # Data are labelled at the END of their hour. A real 08:00–18:00
        # job window has end-hour labels 09–18, NOT 08–17.
        window = historical_day(group, date)
        actual = window['actual'].to_numpy(dtype=float)
        yesterday = window['baseline'].to_numpy(dtype=float)
        ai = window['ai_prediction'].to_numpy(dtype=float)
        if not all(np.isfinite(a).all() for a in (actual, yesterday, ai)):
            raise ValueError(f'Invalid solar data on {date}')
        starts = {
            'SunPilot AI': choose_best(ai),
            'Yesterday': choose_best(yesterday),
            'Fixed 12:00': FIXED_START,
            'Perfect knowledge': choose_best(actual),
        }
        measured = {name: solar_irradiation(actual, hour) for name, hour in starts.items()}
        if measured['Perfect knowledge'] < max(measured.values()) - 1e-7:
            raise AssertionError('Oracle did not dominate — possible alignment bug')
        pv = {name: hypothetical_self_use_kwh(actual[hour-LOCAL_START:hour-LOCAL_START+TASK_DURATION])
              for name, hour in starts.items()}
        records.append({
            'date': date,
            'ai_start': starts['SunPilot AI'],
            'baseline_start': starts['Yesterday'],
            'fixed_start': starts['Fixed 12:00'],
            'oracle_start': starts['Perfect knowledge'],
            'ai_actual_wh_m2': measured['SunPilot AI'],
            'baseline_actual_wh_m2': measured['Yesterday'],
            'fixed_actual_wh_m2': measured['Fixed 12:00'],
            'oracle_actual_wh_m2': measured['Perfect knowledge'],
            'ai_pv_scenario_kwh': pv['SunPilot AI'],
            'baseline_pv_scenario_kwh': pv['Yesterday'],
            'fixed_pv_scenario_kwh': pv['Fixed 12:00'],
        })
    if not records:
        raise ValueError(f'No eligible {year} observations')
    result = pd.DataFrame(records)
    names = {
        'Fixed 12:00': 'fixed_actual_wh_m2',
        'Yesterday': 'baseline_actual_wh_m2',
        'SunPilot AI': 'ai_actual_wh_m2',
        'Perfect knowledge': 'oracle_actual_wh_m2',
    }
    means = {name: round(float(result[field].mean()), 3) for name, field in names.items()}
    ai_v = result['ai_actual_wh_m2'].to_numpy()
    fix_v = result['fixed_actual_wh_m2'].to_numpy()
    yd_v = result['baseline_actual_wh_m2'].to_numpy()
    gain_fixed = 100 * (ai_v.sum() / fix_v.sum() - 1)
    gain_yesterday = 100 * (ai_v.sum() / yd_v.sum() - 1)
    worst = result.assign(delta=result['ai_actual_wh_m2'] - result['baseline_actual_wh_m2']).sort_values('delta').iloc[0]
    out = {
        'days': len(result),
        'mean_wh_m2': means,
        'ai_gain_vs_fixed_percent': round(float(gain_fixed), 3),
        'ai_gain_vs_yesterday_percent': round(float(gain_yesterday), 3),
        'ai_gain_vs_fixed_ci95_percent': bootstrap_relative_gain(ai_v, fix_v),
        'ai_gain_vs_yesterday_ci95_percent': bootstrap_relative_gain(ai_v, yd_v),
        'ai_vs_fixed_day_results': classify_days(result, 'ai_actual_wh_m2', 'fixed_actual_wh_m2'),
        'ai_vs_yesterday_day_results': classify_days(result, 'ai_actual_wh_m2', 'baseline_actual_wh_m2'),
        'worst_day_vs_yesterday': str(worst['date']),
        'worst_day_vs_yesterday_wh_m2': round(float(worst['delta']), 3),
        'pv_scenario_mean_kwh': {
            'SunPilot AI': round(float(result['ai_pv_scenario_kwh'].mean()), 6),
            'Yesterday': round(float(result['baseline_pv_scenario_kwh'].mean()), 6),
            'Fixed 12:00': round(float(result['fixed_pv_scenario_kwh'].mean()), 6),
        },
    }
    return result, out


def create_plot(metrics: dict) -> None:
    names = ['Fixed 12:00', 'Yesterday', 'SunPilot AI', 'Perfect knowledge']
    means = metrics['mean_wh_m2']
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, [means[n] for n in names])
    ax.bar_label(bars, fmt='%.1f', padding=3)
    ax.set_ylabel('Mean solar irradiation for 2-hour task (Wh/m²)')
    ax.set_title('SunPilot: corrected hour-end alignment / 2025 test')
    ax.tick_params(axis='x', rotation=12)
    fig.tight_layout()
    fig.savefig(RESULTS / 'schedule_comparison_2025.png', dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    years = {}
    for year, filename in ((2024, 'validation_predictions_2024.csv'), (2025, 'test_predictions_2025.csv')):
        path = RESULTS / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}. Run python src/train.py first.')
        frame = pd.read_csv(path)
        per_day, metrics = evaluate_frame(frame, year)
        per_day.to_csv(RESULTS / f'schedule_evaluation_{year}.csv', index=False)
        years['validation_2024' if year == 2024 else 'test_2025'] = metrics
        print(f'\n== VERIFIED HOURLY ALIGNMENT {year} ==')
        print(f'Days evaluated: {metrics["days"]}')
        print('Mean irradiation (Wh/m²):')
        for name, val in metrics['mean_wh_m2'].items():
            print(f'  {name:20}: {val:.3f}')
        print(f'AI vs yesterday: {metrics["ai_gain_vs_yesterday_percent"]:+.3f}% '
              f'(95% paired bootstrap: {metrics["ai_gain_vs_yesterday_ci95_percent"]})')
        print(f'AI vs fixed noon: {metrics["ai_gain_vs_fixed_percent"]:+.3f}% '
              f'(95% paired bootstrap: {metrics["ai_gain_vs_fixed_ci95_percent"]})')
        print('Equal/better/worse by day vs fixed:', metrics['ai_vs_fixed_day_results'])
    meta = {
        **years,
        'task_duration_hours': TASK_DURATION,
        'operating_window': '08:00–18:00 Europe/Stockholm actual job-time',
        'hour_label_semantics': 'hour_end',
        'hour_end_labels': '09:00–18:00 local inclusive',
        'evaluation_version': 2,
        'uncertainty_note': 'Paired day bootstrap CI is descriptive; serial and seasonal correlation remain.',
        'pv_scenario': {
            'panel_area_m2': 4.0, 'nominal_efficiency': 0.20, 'computing_load_w': 300.0,
            'note': 'ILLUSTRATIVE horizontal-panel potential; NOT measured power, PV output or CO2 savings.',
        },
        'note': 'ERA5 historical reanalysis + saved held-out model predictions. Oracle uses actual solar data only to evaluate decisions.',
    }
    (RESULTS / 'scheduler_metrics.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    create_plot(years['test_2025'])
    (RESULTS / 'scientific_audit.md').write_text(
        '# SunPilot AI — corrected scheduling audit\n\n'
        '## Time alignment\n\n'
        'Open-Meteo hourly shortwave_radiation is a mean of the preceding hour. '
        'For real tasks from 08:00 to 18:00 local, the ten hourly labels used are **09:00–18:00**, '
        'not 08:00–17:00. This change corrects the v1.0 historical backtest.\n\n'
        '## 2025 held-out test\n\n'
        f'AI vs previous-day scheduling: **{years["test_2025"]["ai_gain_vs_yesterday_percent"]:+.3f}%** '
        f'(paired day bootstrap 95% interval: {years["test_2025"]["ai_gain_vs_yesterday_ci95_percent"]}).\n\n'
        f'AI vs fixed noon: **{years["test_2025"]["ai_gain_vs_fixed_percent"]:+.3f}%** '
        f'(paired day bootstrap 95% interval: {years["test_2025"]["ai_gain_vs_fixed_ci95_percent"]}).\n\n'
        '## Limitations\n\n'
        '- ERA5 reanalysis is not an archived operational forecast; live AI transfer has not been validated.\n'
        '- Historical task lengths are exactly two hours in aggregate benchmark.\n'
        '- Irradiance is not electricity production and cannot establish climate benefit alone.\n'
        '- Hypothetical PV self-use assumes 4 m² of horizontal panels, 20% efficiency, '
        '300 W computer, no batteries or shading; this is **not** measured or avoided CO2.\n'
        '- Validation year (2024) and test year (2025) were not used for new model fitting here.\n',
        encoding='utf-8',
    )
    print('\nSaved corrected metrics, experiment CSVs, graph, and scientific_audit.md')


if __name__ == '__main__':
    main()
