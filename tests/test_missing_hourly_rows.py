"""Missing entire forecast timestamps should not disable usable ranges."""
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import pytest

from src.live_planner import fetch_forecast, Location, build_plan


class FakeModel:
    def predict(self, features):
        return np.maximum(features['radiation_lag_24'].to_numpy(dtype=float),0)


def test_missing_entire_hour_is_reindexed_and_skipped(monkeypatch):
    import src.live_planner as live
    ix=pd.date_range('2026-11-06T00:00:00Z', periods=26*24, freq='h')
    gap=ix[20*24]
    ix=ix.delete(20*24)
    raw={
        'timezone':'Europe/Stockholm',
        'hourly': {
            'time':list(ix.as_unit('s').astype('int64')),
            'shortwave_radiation':[100.0]*len(ix),
            'cloud_cover':[10]*len(ix),
            'temperature_2m':[5]*len(ix),
        },
    }
    class Response:
        def raise_for_status(self): pass
        def json(self):return raw
    monkeypatch.setattr(live.requests,'get',lambda *a,**k: Response())
    f=fetch_forecast(Location('Klippan',56.13,13.13,'Europe/Stockholm'))
    assert len(f.hourly)==26*24
    assert f.hourly.loc[gap,'shortwave_radiation'] != f.hourly.loc[gap,'shortwave_radiation']
    plan=build_plan(f,FakeModel(),90,1,0,datetime(2026,11,15,9,7,tzinfo=timezone.utc))
    assert plan.estimated_wh_m2>0


def test_invalid_subhour_timestamps_are_rejected(monkeypatch):
    import src.live_planner as live
    ix=pd.date_range('2026-11-06T00:00:00Z', periods=26*24, freq='h')
    ix=ix.delete(100).insert(100,ix[100]+pd.Timedelta(minutes=13))
    payload={
        'timezone':'Europe/Stockholm',
        'hourly':{'time':list(ix.as_unit('s').astype('int64')),
                  'shortwave_radiation':[100.]*len(ix),
                  'cloud_cover':[10]*len(ix),'temperature_2m':[5.]*len(ix)},
    }
    class Response:
        def raise_for_status(self):pass
        def json(self):return payload
    monkeypatch.setattr(live.requests,'get',lambda *a,**k: Response())
    with pytest.raises(ValueError,match='whole hours'):
        fetch_forecast(Location('Klippan',56.13,13.13,'Europe/Stockholm'))
