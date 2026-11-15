"""Synthetic implementation check; NOT evidence of operational forecast skill."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scripts.operational_proxy_eval import compare_archived_proxy


class FakeModel:
    def predict(self, features):
        return np.maximum(features['radiation_lag_24'].to_numpy(dtype=float),0)


def test_archived_proxy_requires_real_baselines_and_reports_sample_count():
    ix=pd.date_range('2025-06-01T00:00:00Z',periods=30*24,freq='h')
    n=len(ix)
    daily_solar=np.clip(np.sin(np.pi*(ix.hour.to_numpy()-6)/12),0,None)*300
    hist=pd.DataFrame({'shortwave_radiation':daily_solar,
                       'cloud_cover':np.full(n,20.),'temperature_2m':np.full(n,10.)},index=ix)
    provider=pd.DataFrame({'shortwave_radiation_previous_day1':daily_solar*0.98},index=ix)
    ref=pd.DataFrame({'shortwave_radiation':daily_solar},index=ix)
    result=compare_archived_proxy(ref,hist,provider,FakeModel(),2025)
    assert result['days_with_complete_daytime']>0
    assert result['hourly_samples']>0
    assert 'EXPLORATORY' in result['status']
    assert 'ai_gain_vs_fixed_percent' in result['schedule_comparison']
