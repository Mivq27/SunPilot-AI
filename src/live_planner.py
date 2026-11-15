"""SunPilot's location-aware live solar planner.

No Qt dependencies. The forecast is a weather-model estimate, NOT PV output.
The experimental ML was trained on Klippan/ERA5 and is NOT externally validated.
For 1-minute durations, hourly radiation is assumed uniform within each hour.
Open-Meteo hourly shortwave radiation refers to the preceding hour; a value at
12:00 is allocated to [11:00, 12:00) for the live integral.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
IP_URL = "https://ipwho.is/"
FEATURES = [
    "hour_sin", "hour_cos", "season_sin", "season_cos",
    "radiation_lag_24", "radiation_lag_48", "radiation_lag_168",
    "cloud_lag_24", "temperature_lag_24",
]


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str | None = None

    def __post_init__(self):
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("Coordinates must be valid latitude/longitude degrees")
        if self.timezone:
            ZoneInfo(self.timezone)


@dataclass
class Forecast:
    location: Location
    timezone: str
    hourly: pd.DataFrame   # UTC aware index, timestamp represents end of hour.
    fetched_at: datetime


@dataclass
class Plan:
    start_utc: datetime
    end_utc: datetime
    estimated_wh_m2: float
    method: str
    comparison: dict[str, tuple[datetime, float]]
    timezone: str
    locations: str
    duration_minutes: int
    chart_utc: pd.DatetimeIndex
    chart_provider: np.ndarray
    chart_ai: np.ndarray
    chart_previous: np.ndarray
    ai_note: str


def geocode_city(query: str) -> list[Location]:
    query = query.strip()
    if len(query) < 2:
        raise ValueError("Type at least two characters to search for a city")
    response = requests.get(GEO_URL, params={"name": query, "count": 8, "language": "en", "format": "json"}, timeout=(7, 18))
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data.get("reason", "Location search failed"))
    places = []
    for result in data.get("results", []):
        name = ", ".join(filter(None, [result.get("name"), result.get("admin1"), result.get("country")]))
        places.append(Location(name, float(result["latitude"]), float(result["longitude"]), result.get("timezone")))
    return places


def location_from_ip() -> Location:
    """Opt-in coarse public IP lookup. Never precise device GPS."""
    response = requests.get(IP_URL, timeout=(7, 18))
    response.raise_for_status()
    data = response.json()
    if not data.get("success", False):
        raise RuntimeError(data.get("message", "Approximate IP location unavailable"))
    city = data.get("city") or data.get("region") or "Approximate area"
    country = data.get("country") or ""
    tz = data.get("timezone", {})
    tz_name = tz.get("id") if isinstance(tz, dict) else None
    return Location(", ".join(filter(None, (city, country))) + " (approx. IP)",
                    float(data["latitude"]), float(data["longitude"]), tz_name)


def fetch_forecast(location: Location) -> Forecast:
    response = requests.get(FORECAST_URL, params={
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": "auto",
        "timeformat": "unixtime",  # Correct through DST and for every timezone.
        "past_days": 8,           # 168-hour lag features, plus safety margin.
        "forecast_days": 16,      # Provider maximum, not an artificial app limit.
        "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
    }, timeout=(9, 40))
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Forecast service rejected the request"))
    tz = payload.get("timezone")
    if not tz:
        raise ValueError("Forecast response has no location timezone")
    ZoneInfo(tz)
    hourly = payload.get("hourly") or {}
    for key in ("time", "shortwave_radiation", "cloud_cover", "temperature_2m"):
        if key not in hourly:
            raise ValueError(f"Forecast missing required field: {key}")
    times = pd.to_datetime(hourly["time"], unit="s", utc=True)
    df = pd.DataFrame({
        "shortwave_radiation": pd.to_numeric(hourly["shortwave_radiation"], errors="coerce"),
        "cloud_cover": pd.to_numeric(hourly["cloud_cover"], errors="coerce"),
        "temperature_2m": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
    }, index=pd.DatetimeIndex(times, name="time_utc"))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < 168 + 24:
        raise ValueError("The forecast did not include enough past hourly data")
    if not bool((df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()):
        raise ValueError("Forecast contains gaps in the hourly timeline")
    # Forecast models can stop supplying particular variables before the
    # requested 16-day horizon. Missing tail hours are normal: keep them as
    # missing and let the scheduler consider only complete task windows.
    if not np.isfinite(df["shortwave_radiation"].to_numpy(dtype=float)).any():
        raise ValueError("No solar-radiation values were provided for this location")
    return Forecast(location, tz, df, datetime.now(timezone.utc))


def predict_ai_hourly(forecast: Forecast, model) -> tuple[np.ndarray, np.ndarray]:
    """Same feature schema as historical training, for exploratory transfer only."""
    frame = forecast.hourly
    loc = frame.index.tz_convert(forecast.timezone)
    hour = loc.hour.to_numpy()
    doy = loc.dayofyear.to_numpy()
    features = pd.DataFrame(index=frame.index)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    features["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for shift in (24, 48, 168):
        features[f"radiation_lag_{shift}"] = frame["shortwave_radiation"].shift(shift)
    features["cloud_lag_24"] = frame["cloud_cover"].shift(24)
    features["temperature_lag_24"] = frame["temperature_2m"].shift(24)
    valid = np.isfinite(features[FEATURES].to_numpy(dtype=float)).all(axis=1)
    ai = np.full(len(frame), np.nan, dtype=float)
    ai[valid] = np.maximum(model.predict(features.loc[valid, FEATURES]), 0)
    previous = features["radiation_lag_24"].to_numpy(dtype=float)
    return ai, previous


def _rounded_up_quarter(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    # Strictly future (time + one minute) and nearest following 15-min mark.
    ts = pd.Timestamp(dt + timedelta(minutes=1)).ceil("15min")
    return ts.to_pydatetime()


def _candidate_indices(times: pd.DatetimeIndex, location_tz: str, when: int,
                       now_utc: datetime, n_minutes: int) -> np.ndarray:
    """Every 15 minutes on the selected start day/range, including night.

    `when`: 0 today, 1 tomorrow, 2 within 7 days, 3 within 16 days.
    """
    now_local = now_utc.astimezone(ZoneInfo(location_tz))
    today = now_local.date()
    candidate_step = 15
    first = _rounded_up_quarter(now_utc)
    offset = int((pd.Timestamp(first) - times[0]) / pd.Timedelta(minutes=1))
    offset = max(offset, 0)
    # Forecast is hourly, but integration is minute-resolution with a stepwise
    # constant assumption. Starts can be on quarter-hour boundaries.
    end = len(times) - n_minutes
    if end < offset:
        return np.array([], dtype=int)
    starts = np.arange(offset, end + 1, dtype=int)
    # Use absolute UTC quarters to avoid DST/local ambiguous times.
    absolute_minutes = (times[0].value // 60_000_000_000) + starts
    starts = starts[absolute_minutes % candidate_step == 0]
    if not len(starts):
        return starts
    local_dates = times[starts].tz_convert(location_tz).date
    day_delta = np.array([(d - today).days for d in local_dates])
    if when == 0:
        eligible = day_delta == 0
    elif when == 1:
        eligible = day_delta == 1
    elif when == 2:
        eligible = (day_delta >= 0) & (day_delta < 7)
    elif when == 3:
        eligible = (day_delta >= 0) & (day_delta < 16)
    else:
        raise ValueError("Unknown scheduling range")
    return starts[eligible]


def _choose(scores: np.ndarray, candidates: np.ndarray) -> tuple[int, float]:
    if not len(candidates):
        raise ValueError("No future start time fits this task within the available forecast. Choose a wider date range or a shorter task.")
    vals = scores[candidates]
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError(
            "No complete forecast interval is available for this task and date range. "
            "Try a shorter task, an earlier date range, or the weather-forecast method."
        )
    # Ignore incomplete forecast intervals without hiding missing values.
    possible = candidates[finite]
    possible_scores = vals[finite]
    # Earliest start breaks ties; avoid arbitrary bias across days.
    idx = int(np.argmax(possible_scores))
    return int(possible[idx]), float(possible_scores[idx])


def build_plan(forecast: Forecast, model, duration_minutes: int, when: int,
               method: int, now_utc: datetime | None = None) -> Plan:
    if duration_minutes < 1:
        raise ValueError("Duration must be at least one minute")
    now_utc = now_utc or datetime.now(timezone.utc)
    now_utc = now_utc.astimezone(timezone.utc)
    df = forecast.hourly
    ai_hourly, prev_hourly = predict_ai_hourly(forecast, model)
    # Open-Meteo hourly radiation represents preceding hour [timestamp-1h, timestamp).
    # We can align all three methods to that same interval and integrate exactly
    # under a piecewise-constant approximation.
    timeline = pd.date_range(start=df.index[0] - pd.Timedelta(hours=1),
                             periods=len(df) * 60, freq="min", tz="UTC")
    provider_hourly = np.maximum(df["shortwave_radiation"].to_numpy(dtype=float), 0)
    series = {
        "Weather forecast": np.repeat(provider_hourly, 60) / 60.0,
        "SunPilot AI (experimental)": np.repeat(ai_hourly, 60) / 60.0,
        "Yesterday proxy": np.repeat(prev_hourly, 60) / 60.0,
    }
    candidates = _candidate_indices(timeline, forecast.timezone, when, now_utc, duration_minutes)
    if not len(candidates):
        raise ValueError("This task cannot fit within the remaining forecast period. Choose an earlier start range or shorten the task.")
    scores = {}
    choices = {}
    # Cumulative-sum trick: O(n) for any duration; no duration cap in the app.
    for name, minute_irradiance in series.items():
        safe = np.where(np.isfinite(minute_irradiance), minute_irradiance, 0.0)
        cumulative = np.r_[0.0, np.cumsum(safe)]
        sums = cumulative[duration_minutes:] - cumulative[:-duration_minutes]
        eligible = np.isfinite(minute_irradiance).astype(np.int32)
        valid_prefix = np.r_[0, np.cumsum(eligible)]
        valid = (valid_prefix[duration_minutes:] - valid_prefix[:-duration_minutes] == duration_minutes)
        sums[~valid] = np.nan
        scores[name] = sums

    # Every recommendation must have a usable provider value. Compare
    # alternatives only on intervals where the provider can score the result.
    usable = candidates[np.isfinite(scores["Weather forecast"][candidates])]
    if not len(usable):
        raise ValueError(
            "There is no complete solar forecast for a task of this duration "
            "in the selected period. Try a shorter task or an earlier range."
        )
    for name, sums in scores.items():
        if np.isfinite(sums[usable]).any():
            choices[name] = _choose(sums, usable)

    chosen_name = "SunPilot AI (experimental)" if method == 1 else "Weather forecast"
    if chosen_name not in choices:
        raise ValueError(
            "The experimental AI does not have enough valid input data for "
            "this period. Choose Weather forecast or a different date range."
        )
    start_position, _ = choices[chosen_name]
    starts = {key: (timeline[pos].to_pydatetime(), float(scores["Weather forecast"][pos]))
              for key, (pos, _) in choices.items()}
    if not np.isfinite(starts[chosen_name][1]):
        raise ValueError("No valid provider estimate for the selected AI interval")
    start = timeline[start_position].to_pydatetime()
    end = start + timedelta(minutes=duration_minutes)
    # Chart only hours near chosen period: enough context without plotting thousands
    # of minute-level points. For very long jobs show all hourly samples, at most 384.
    hour_range_start = start - timedelta(hours=6)
    hour_range_end = end + timedelta(hours=6)
    chart_mask = np.array([hour_range_start <= t.to_pydatetime() - timedelta(hours=1) <= hour_range_end
                           for t in df.index])
    chart_times = df.index[chart_mask] - pd.Timedelta(hours=1)
    if not len(chart_times):
        raise ValueError("No forecast chart data in the chosen interval")
    note = "AI trained on Klippan (Sweden) ERA5 data; live inputs and other regions are not validated."
    return Plan(start, end, starts[chosen_name][1], chosen_name, starts,
                forecast.timezone, forecast.location.name, duration_minutes,
                chart_times, provider_hourly[chart_mask], ai_hourly[chart_mask],
                prev_hourly[chart_mask], note)


def minute_window(values_hourly: np.ndarray, duration_minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Historical 08:00–18:00 window, every quarter-hour start; per-minute estimate.

    Keeps the original model and data untouched. Data for 2025 is not used to
    retrain/tune the model. The 10-hour daily window is a methodological limit.
    """
    values = np.asarray(values_hourly, dtype=float)
    if len(values) != 10 or not np.isfinite(values).all():
        raise ValueError("Historical day requires ten valid hourly values")
    if duration_minutes < 1 or duration_minutes > 600:
        raise ValueError("The historical experiment covers 08:00–18:00; this task cannot fit in one day. The Live Planner can handle longer tasks.")
    series = np.repeat(values, 60) / 60.0
    starts = np.arange(0, 600 - duration_minutes + 1, 15, dtype=int)
    c = np.r_[0.0, np.cumsum(series)]
    results = c[starts + duration_minutes] - c[starts]
    return starts, results


def best_historical_start(values_hourly: np.ndarray, duration_minutes: int) -> tuple[int, float]:
    starts, results = minute_window(values_hourly, duration_minutes)
    best = np.flatnonzero(np.isclose(results, results.max(), atol=1e-8))
    # Historical tie-breaking retains the 12:00 preference from the original backtest.
    selection = min(best, key=lambda i: (abs(int(starts[i]) - 240), int(starts[i])))
    return int(starts[selection]), float(results[selection])


def historical_energy(values_hourly: np.ndarray, start_minutes: int, duration_minutes: int) -> float:
    series = np.repeat(np.asarray(values_hourly, dtype=float), 60) / 60.0
    return float(np.sum(series[start_minutes:start_minutes + duration_minutes]))


def format_hhmm(start_minutes: int, duration_minutes: int) -> str:
    start = 8 * 60 + start_minutes
    end = start + duration_minutes
    return f"{start//60:02d}:{start%60:02d}–{end//60:02d}:{end%60:02d}"


def human_duration(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return ", ".join(filter(None, [f"{h} h" if h else "", f"{m} min" if m else ""])) or "0 min"
