# SunPilot AI

**Location-aware solar scheduling for flexible computing workloads**  
Built for the Veritas AI International High School Hackathon 2026 · AI + Climate

SunPilot is a native Windows desktop application with a historical research
simulator and a usable forecast-based planning mode. It recommends when to run
flexible workloads such as rendering or Python scripts to coincide with higher
forecast solar irradiance. **It does not execute programs automatically.**

## Desktop v0.5 — what's new

- **Choose any location** by searching for a city (Open-Meteo Geocoding API),
  entering latitude and longitude, or explicitly requesting an **approximate**
  location lookup using the device's public IP. IP lookup is **not GPS**.
- **No arbitrary task-duration cap:** enter hours and minutes (at least 1 minute).
  Jobs may run overnight or span multiple days. The only live scheduling limit
  is the provider's actual forecast coverage (at most 16 forecast days).
- **Flexible start ranges:** today, tomorrow, next 7 days, or up to 16 days.
- **Quarter-hour start times** with **minute-length** tasks: integration uses
  hourly forecasts, assumed constant within each hour; this is *not* a true
  minute-resolution forecast.
- Select either the **Open-Meteo forecast** (default, practical choice) or the
  **SunPilot ML experiment** (not live-validated).
- UI with adaptive scrolling, plots, explanations and comparisons.
- Historical single-day simulator supports minutes for jobs that fit the
  original **08:00–18:00** evaluation window. This research constraint is
  separate from the unrestricted-duration live planner.

## Quick start

Tested with Python 3.13 on Windows 11.

```powershell
cd "$env:USERPROFILE\Projects\SunPilot-AI"
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install PySide6 tzdata
.\.venv\Scripts\python.exe .\sunpilot_desktop_v05.py
```

The repository includes `models/sunpilot_v1.joblib` and
`data/klippan_hourly_2022_2025.csv`, which are needed at startup.
Live forecast and geocoding require internet access. Historical simulation
works offline.

### How to use

1. Open **Plan a task**. The initial location is Klippan, Sweden unless you
   have saved another location.
2. Enter a city in the search box and pick a match, or use coordinates.
   **Approximate my location** is an optional opt-in public-IP lookup.
3. Choose your task, start range, and duration in hours and minutes.
4. Click **Find best time** to retrieve a forecast and compare methods.
5. Copy the recommendation if desired. No program is launched automatically.

Local location preferences are saved in Windows QSettings. When you select
**Approximate my location**, an HTTPS request to `ipwho.is` reveals your
public IP to that provider. City lookup sends the search terms to Open-Meteo;
forecast lookup sends the selected coordinates to Open-Meteo. No IP lookup is
triggered at startup.

### Build Windows app (.exe)

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller tzdata
powershell -ExecutionPolicy Bypass -File .\build_sunpilot_v05.ps1
.\dist\SunPilotAI\SunPilotAI.exe
```

When distributing the `--onedir` build, **share the entire `dist/SunPilotAI`
folder**, not just the executable. Test on Windows before publishing a release.

## Research question

Can an ML radiation forecast choose better hours for flexible computing
workloads than simple schedules?

### Dataset & model

- Location: Klippan, Sweden, at approximately 56.1356° N, 13.1309° E.
- Data: Open-Meteo **ERA5 historical reanalysis** for 2022–2025, 35,064
  hourly observations.
- Model: scikit-learn `HistGradientBoostingRegressor`.
- Training years: 2022–2023; validation: 2024; held-out test: 2025.
- Inputs: local hour/season and 24-, 48-, 168-hour radiation, temperature
  and cloud-cover lag features.

2025 prediction error (MAE, W/m²):

| Method | All hours | 08:00–18:00 local |
|---|---:|---:|
| Previous-day baseline | 39.98 | 85.96 |
| SunPilot AI | 37.29 | 79.70 |

2025 scheduling evaluation, **two-hour tasks**, observed solar irradiation:

| Method | Mean Wh/m² per task |
|---|---:|
| Fixed noon | 673.01 |
| Previous-day baseline | 662.06 |
| SunPilot AI | 671.81 |
| Perfect knowledge (unattainable upper bound) | 727.45 |

The ML schedule improved **1.47%** over yesterday-based scheduling, but was
**0.18% worse** than fixed noon. Better prediction error did not guarantee
better scheduling decisions. These are *historical simulations*, not measured
power/CO₂ savings. For other task lengths, the UI displays individual-day
simulations; the saved aggregate benchmark remains for two-hour tasks only.

### Reproducing the historical experiment

```powershell
.\.venv\Scripts\python.exe .\src\download_data.py
.\.venv\Scripts\python.exe .\src\train.py
.\.venv\Scripts\python.exe .\src\scheduler.py
```

## Limitations and transparency

- **Historical**: ERA5 reanalysis is not a real-time operational weather forecast.
- **Live default**: Open-Meteo's current weather model provides the forecast;
  this is not our ML forecast and not measured solar power.
- **Live experimental ML**: same pretrained model, supplied with forecast-model
  lagged inputs, not its original ERA5 data. No archived operational forecast
  validation has been completed; accuracy in new locations is unknown.
- Forecast horizon is at most **16 days**; SunPilot rejects workloads that
  cannot fit in the available time series. Long-range forecasts are uncertain.
- Start times are rounded to **15 minutes**, with arbitrary minute durations;
  underlying weather values remain **hourly**. Solar radiation is integrated
  using piecewise-constant hourly estimates and should not be mistaken for
  sub-hourly observations.
- Solar irradiation in Wh/m² is not electricity production or verified climate
  impact. No PV system characteristics or grid carbon-intensity data are used.
- No task execution, task monitoring, GPS access, or location telemetry beyond
  the user-triggered third-party requests described above.

## Source & acknowledgment

Weather: [Open-Meteo](https://open-meteo.com/) (data attribution, CC BY 4.0);
location search: Open-Meteo Geocoding API (GeoNames-derived data);
optional approximate IP lookup: [ipwho.is](https://ipwho.is/).

ChatGPT was used for planning, code generation, debugging and documentation.
The project team is responsible for understanding, testing and accurately
reporting the code and results.
