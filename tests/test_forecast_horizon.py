"""Regression tests: missing radiation data near end of forecast horizon."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_planner import Forecast, Location, build_plan, fetch_forecast


class FakeModel:
    def predict(self, features):
        return np.maximum(features["radiation_lag_24"].to_numpy(dtype=float), 0)


def forecast_with_tail_gaps():
    times = pd.date_range("2026-11-06T00:00:00Z", periods=24 * 26, freq="h")
    radians = np.maximum(0.0, 300 * np.sin(np.pi * (times.hour.to_numpy() - 7) / 12))
    frame = pd.DataFrame(
        {
            "shortwave_radiation": radians,
            "cloud_cover": 20.0,
            "temperature_2m": 5.0,
        }, index=times
    )
    frame.iloc[-60:, frame.columns.get_loc("shortwave_radiation")] = np.nan
    location = Location("Klippan, Sweden", 56.1356, 13.1309, "Europe/Stockholm")
    return Forecast(location, "Europe/Stockholm", frame, datetime.now(timezone.utc))


def test_short_task_works_even_when_last_60_hours_missing():
    f = forecast_with_tail_gaps()
    now = datetime(2026, 11, 15, 9, 7, tzinfo=timezone.utc)
    plan = build_plan(f, FakeModel(), 90, 1, 0, now)
    assert plan.estimated_wh_m2 >= 0
    assert plan.end_utc > plan.start_utc


def test_range_planner_skips_incomplete_intervals():
    f = forecast_with_tail_gaps()
    now = datetime(2026, 11, 15, 9, 7, tzinfo=timezone.utc)
    plan = build_plan(f, FakeModel(), 180, 3, 0, now)
    last_valid_end = f.hourly["shortwave_radiation"].last_valid_index()
    assert plan.end_utc <= last_valid_end.to_pydatetime()


def test_experimental_ai_skips_missing_tail():
    f = forecast_with_tail_gaps()
    now = datetime(2026, 11, 15, 9, 7, tzinfo=timezone.utc)
    plan = build_plan(f, FakeModel(), 75, 3, 1, now)
    assert plan.method == "SunPilot AI (experimental)"


def test_no_usable_solar_data_explains_problem():
    f = forecast_with_tail_gaps()
    f.hourly["shortwave_radiation"] = np.nan
    with pytest.raises(ValueError, match="no complete solar forecast"):
        build_plan(f, FakeModel(), 90, 1, 0,
                   datetime(2026, 11, 15, 9, 7, tzinfo=timezone.utc))


def test_download_accepts_missing_forecast_tail(monkeypatch):
    times = pd.date_range("2026-11-06T00:00:00Z", periods=24 * 24, freq="h")
    rad = [100.0] * len(times)
    rad[-42:] = [None] * 42
    payload = {
        "timezone": "Europe/Stockholm",
        "hourly": {
            "time": list(times.astype("int64") // 10**9),
            "shortwave_radiation": rad,
            "cloud_cover": [10] * len(times),
            "temperature_2m": [5.0] * len(times),
        },
    }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr("src.live_planner.requests.get", lambda *args, **kwargs: Response())
    forecast = fetch_forecast(Location("Klippan", 56.13, 13.13))
    assert len(forecast.hourly) == len(times)
    assert forecast.hourly["shortwave_radiation"].isna().sum() == 42
