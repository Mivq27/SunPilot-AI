"""Offline full-year integration audit with synthetic data only.

NEVER use these synthetic results in the public project submission.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src import scheduler, recheck_metrics


def test_full_years_audit_outputs(monkeypatch,tmp_path):
    monkeypatch.setattr(scheduler, 'RESULTS', tmp_path)
    monkeypatch.setattr(recheck_metrics, 'RESULTS', tmp_path)
    for year, name in [(2024,'validation_predictions_2024.csv'), (2025,'test_predictions_2025.csv')]:
        rows=[]
        for date in pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D'):
            for end_hour in range(8,19):
                ghi = max(0,500-70*abs(end_hour-13))
                rows.append({'local_date':date.strftime('%Y-%m-%d'), 'local_hour':end_hour,
                             'actual':ghi,'baseline':ghi*0.80, 'ai_prediction':ghi*0.98})
        pd.DataFrame(rows).to_csv(tmp_path/name,index=False)
    recheck_metrics.main()
    scheduler.main()
    results=json.loads((tmp_path/'scheduler_metrics.json').read_text(encoding='utf-8'))
    mae=json.loads((tmp_path/'metrics.json').read_text(encoding='utf-8'))
    assert results['evaluation_version']==2
    assert results['test_2025']['days']==365
    assert results['validation_2024']['days']==366
    assert mae['evaluation_version']==2
    assert (tmp_path/'scientific_audit.md').exists()
    assert (tmp_path/'schedule_comparison_2025.png').stat().st_size>1000
