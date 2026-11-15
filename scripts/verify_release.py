"""Refuse to distribute a v1.1 build containing invalid v1.0 scheduling metrics."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path) -> list[str]:
    errors=[]
    for required in (
        'sunpilot_desktop_v1.py', 'src/live_planner.py', 'src/windows_scheduler.py',
        'src/solar_time.py', 'src/pv_scenario.py',
        'data/klippan_hourly_2022_2025.csv', 'models/sunpilot_v1.joblib',
        'results/metrics.json', 'results/scheduler_metrics.json',
        'results/schedule_comparison_2025.png',
        'docs/SCIENTIFIC_AUDIT.md',
    ):
        if not (root/required).is_file():
            errors.append(f'Missing: {required}')
    metrics = root/'results/scheduler_metrics.json'
    if metrics.is_file():
        try:
            content = json.loads(metrics.read_text(encoding='utf-8'))
            if content.get('evaluation_version') != 2 or content.get('hour_label_semantics') != 'hour_end':
                errors.append('Scheduling metrics are from the OLD v1.0 misaligned evaluation. Run python src/scheduler.py first.')
            for key, expected_days in [('validation_2024',366),('test_2025',365)]:
                if content.get(key,{}).get('days') != expected_days:
                    errors.append(f'{key}: expected {expected_days} days of evaluation')
                required_keys=('ai_gain_vs_yesterday_ci95_percent','ai_gain_vs_fixed_ci95_percent', 'mean_wh_m2')
                for name in required_keys:
                    if name not in content.get(key,{}):
                        errors.append(f'{key}: no {name}')
        except (ValueError, OSError) as exc:
            errors.append(f'Cannot validate scheduling metrics: {exc}')
    forecast_metrics = root/'results/metrics.json'
    if forecast_metrics.is_file():
        try:
            content = json.loads(forecast_metrics.read_text(encoding='utf-8'))
            if content.get('evaluation_version') != 2 or content.get('hour_label_semantics') != 'hour_end':
                errors.append('Prediction daytime MAE is from the old misaligned window. Run python src/recheck_metrics.py first.')
        except (OSError, ValueError) as exc:
            errors.append(f'Cannot validate prediction metrics: {exc}')
    return errors


def main():
    failures=verify(ROOT)
    if failures:
        print('SUNPILOT v1.1 RELEASE BLOCKED:')
        print('\n'.join(f' - {err}' for err in failures))
        sys.exit(1)
    print('SUNPILOT v1.1 RELEASE CHECK PASSED: scientific assets and corrected metrics verified.')


if __name__ == '__main__':
    main()
