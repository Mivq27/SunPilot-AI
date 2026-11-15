"""Fail-closed distribution guard: old mislabeled science cannot ship as v1.1."""
import json
from scripts.verify_release import verify


def test_gate_rejects_old_metrics_and_accepts_corrected(tmp_path):
    for required in [
        'sunpilot_desktop_v1.py', 'src/live_planner.py', 'src/windows_scheduler.py',
        'src/solar_time.py', 'src/pv_scenario.py',
        'data/klippan_hourly_2022_2025.csv', 'models/sunpilot_v1.joblib',
        'results/metrics.json', 'results/scheduler_metrics.json',
        'results/schedule_comparison_2025.png', 'docs/SCIENTIFIC_AUDIT.md',
    ]:
        f=tmp_path/required
        f.parent.mkdir(parents=True,exist_ok=True)
        f.write_text('{}')
    failures=verify(tmp_path)
    assert any('OLD v1.0' in err for err in failures)
    assert any('daytime MAE' in err for err in failures)
    corrected={
        'evaluation_version':2,'hour_label_semantics':'hour_end',
        'validation_2024':{'days':366,'mean_wh_m2':{},
                           'ai_gain_vs_yesterday_ci95_percent':[0,1],
                           'ai_gain_vs_fixed_ci95_percent':[0,1]},
        'test_2025':{'days':365,'mean_wh_m2':{},
                      'ai_gain_vs_yesterday_ci95_percent':[0,1],
                      'ai_gain_vs_fixed_ci95_percent':[0,1]},
    }
    (tmp_path/'results/scheduler_metrics.json').write_text(json.dumps(corrected))
    (tmp_path/'results/metrics.json').write_text(json.dumps({'evaluation_version':2,'hour_label_semantics':'hour_end'}))
    assert not verify(tmp_path)
