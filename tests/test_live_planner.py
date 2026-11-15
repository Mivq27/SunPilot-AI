import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_planner import (
    Location, Forecast, build_plan, minute_window, best_historical_start,
    historical_energy, format_hhmm, human_duration, _candidate_indices,
)

class FakeModel:
    def predict(self, x):
        assert list(x.columns)==['hour_sin','hour_cos','season_sin','season_cos','radiation_lag_24','radiation_lag_48','radiation_lag_168','cloud_lag_24','temperature_lag_24']
        return np.maximum(x['radiation_lag_24'].to_numpy()*0.85,0)

def make_weather(tz='Europe/Stockholm', loc=None):
    start=pd.Timestamp('2026-11-06T00:00:00Z')
    ix=pd.date_range(start, periods=24*26, freq='h')
    local=ix.tz_convert(tz)
    rad=np.clip(380*np.sin(np.pi*(local.hour.to_numpy()-7)/10),0,None)
    df=pd.DataFrame({'shortwave_radiation':rad,'temperature_2m':np.full(len(ix),12.0),'cloud_cover':np.full(len(ix),30.0)},index=ix)
    return Forecast(loc or Location('Test Stockholm',59.3,18.06,tz),tz,df,datetime.now(timezone.utc))

def test_90_min_today():
    f=make_weather()
    p=build_plan(f,FakeModel(),90,0,0,datetime(2026,11,15,9,7,tzinfo=timezone.utc))
    assert p.duration_minutes==90
    assert p.start_utc > datetime(2026,11,15,9,7,tzinfo=timezone.utc)
    assert p.start_utc.minute%15==0
    assert p.end_utc-p.start_utc==timedelta(minutes=90)
    assert p.start_utc.astimezone(ZoneInfo(f.timezone)).date()==datetime(2026,11,15).date()
    assert p.estimated_wh_m2>=0

def test_20_hour_overnight():
    f=make_weather()
    p=build_plan(f,FakeModel(),20*60+37,1,1,datetime(2026,11,15,9,7,tzinfo=timezone.utc))
    assert p.duration_minutes==1237
    assert p.end_utc-p.start_utc==timedelta(minutes=1237)
    assert p.start_utc.astimezone(ZoneInfo(f.timezone)).date()==datetime(2026,11,16).date()

def test_very_long_within_horizon():
    f=make_weather()
    p=build_plan(f,FakeModel(),8*24*60,3,0,datetime(2026,11,15,9,7,tzinfo=timezone.utc))
    assert p.duration_minutes == 8*24*60

def test_impossible_duration():
    f=make_weather()
    with pytest.raises(ValueError,match='forecast|fit'):
        build_plan(f,FakeModel(),30*24*60,3,0,datetime(2026,11,15,9,7,tzinfo=timezone.utc))

def test_local_time_selection_los_angeles():
    f=make_weather('America/Los_Angeles',Location('Los Angeles, USA',34.05,-118.2,'America/Los_Angeles'))
    p=build_plan(f,FakeModel(),75,1,0,datetime(2026,11,15,9,7,tzinfo=timezone.utc))
    assert p.start_utc.astimezone(ZoneInfo(f.timezone)).date() == datetime(2026,11,16).date()

def test_historical_minutes():
    actual = np.array([0, 25, 100, 200, 300, 500, 650, 200, 90, 0.],dtype=float)
    starts,scores=minute_window(actual,135)
    assert starts[0]==0
    assert starts[-1]<=600-135
    assert np.isclose(historical_energy(actual,int(starts[3]),135),scores[3])
    best,energy=best_historical_start(actual,135)
    assert 0<=best<=600-135
    assert np.isclose(historical_energy(actual,best,135),energy)
    assert len(format_hhmm(best,135))==11
    assert human_duration(135)=='2 h, 15 min'



def test_fetch_forecast_unix_timestamps(monkeypatch):
    import src.live_planner as l
    ix=pd.date_range('2026-11-06T00:00:00Z',periods=24*24,freq='h')
    payload={
        'timezone':'Europe/Stockholm',
        'hourly':{
            'time':list(ix.as_unit('s').astype('int64')),
            'shortwave_radiation':[float(x%9) for x in range(len(ix))],
            'cloud_cover':[10]*len(ix),
            'temperature_2m':[2.0]*len(ix),
        }
    }
    class Response:
        def raise_for_status(self): return None
        def json(self): return payload
    calls=[]
    def get(url,params,timeout):
        calls.append((url,params,timeout))
        return Response()
    monkeypatch.setattr(l.requests,'get',get)
    fc=l.fetch_forecast(Location('MalmÃ¶',55.6,13.,'Europe/Stockholm'))
    assert fc.timezone == 'Europe/Stockholm'
    assert str(fc.hourly.index.tz) == 'UTC'
    assert calls[0][1]['forecast_days']==16
    assert calls[0][1]['timeformat']=='unixtime'


def test_location_ip_explicit_call_only(monkeypatch):
    import src.live_planner as l
    data={'success':True,'city':'GÃ¶teborg','country':'Sweden','latitude':57.7,'longitude':11.9,'timezone':{'id':'Europe/Stockholm'}}
    class Response:
        def raise_for_status(self):return None
        def json(self): return data
    requests=[]
    def get(url,timeout):
        requests.append(url)
        return Response()
    monkeypatch.setattr(l.requests,'get',get)
    place=l.location_from_ip()
    assert place.name.startswith('GÃ¶teborg')
    assert place.timezone=='Europe/Stockholm'
    assert requests == [l.IP_URL]

