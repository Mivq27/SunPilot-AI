# SunPilot AI

**Location-aware solar scheduling for flexible computing workloads**  
Built for the Veritas AI International High School Hackathon 2026 · AI + Climate

SunPilot is a native Windows desktop application with a historical research
simulator and a usable forecast-based planning mode. It recommends when to run
flexible workloads such as rendering or Python scripts to coincide with higher
forecast solar irradiance. **It can optionally launch a user-chosen Windows `.exe` at a confirmed, one-time schedule using Windows Task Scheduler.**

## Desktop v1.0 — release candidate

- **Optional Windows automation:** choose a trusted `.exe`, use SunPilot's suggested time or manually choose a local date/time, confirm, and schedule it once with Windows Task Scheduler. View and cancel schedules from the app. **Nothing runs until you opt in.**
- **Simplified planner:** removed Task type (it never changed the numerical forecast or scheduling decision).
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
- **Scroll-safe duration inputs:** the mouse wheel scrolls the page instead of
  silently changing the selected hours or minutes. The same applies to dates,
  coordinates and dropdowns (unless their menu is expanded).
- **More comfortable scrolling:** wheel animations, native touchpad deltas,
  graphs and tables that let the page scroll, and responsive columns.
- **Release interface:** clean Plan, Automate, Simulator, Research tabs; clear
  always-visible Weather Forecast / SunPilot AI method choice; simple scheduling
  and optional details, responsive layout and wheel-safe inputs.
- **Forecast-tail fix (v0.5.1):** incomplete weather hours at the end of a
  forecast no longer invalidate earlier usable planning windows.
- Historical single-day simulator supports minutes for jobs that fit the
  original **08:00–18:00** evaluation window. This research constraint is
  separate from the unrestricted-duration live planner.

### Release status

The scheduling engine tests have passed, but the v1 graphical layout and
packaged Windows executable must still be verified on Windows before calling
this a final release. Source mode and an executable build are not identical.

## Quick start

Tested with Python 3.13 on Windows 11.

```powershell
cd "$env:USERPROFILE\Projects\SunPilot-AI"
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe .\sunpilot_desktop_v1.py
```

The repository includes `models/sunpilot_v1.joblib` and
`data/klippan_hourly_2022_2025.csv`, which are needed at startup.
Live forecast and geocoding require internet access. Historical simulation
works offline.

### How to use

1. Open **Plan a task**. The initial location is Klippan, Sweden unless you
   have saved another location.
2. Click **Change location** to search a city or opt in to approximate
   location by public IP. Coordinates are available under **Enter coordinates**.
3. Choose the start range and duration in hours and minutes. Select **Weather forecast (Recommended)** or **SunPilot AI (Experimental)** using the two large method cards.
   Scroll with the mouse wheel without accidentally changing the inputs.
4. Click **Find best time** to retrieve a forecast. Weather forecast is selected by default. The experimental AI option is visible on the main page.
5. Copy the recommendation if desired. No program is launched automatically.

Local location preferences are saved in Windows QSettings. When you select
**Approximate my location**, an HTTPS request to `ipwho.is` reveals your
public IP to that provider. City lookup sends the search terms to Open-Meteo;
forecast lookup sends the selected coordinates to Open-Meteo. No IP lookup is
triggered at startup.

### Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q .\tests\test_live_planner.py .\tests\test_forecast_horizon.py .\tests\test_windows_scheduler.py
```

23 planning and Windows scheduling tests passed during preparation. The graphical interface
must be checked on Windows (especially wheel behavior and resizing): it could
not be visually launched in the development environment used for this patch.

### Scheduling a Windows `.exe` (optional)

1. In **Plan a task**, choose a start date and task duration, then click **Find best time**.
2. In **Automate**, click **Choose .exe** and select an executable you trust.
3. Keep **Use SunPilot's recommended time** enabled, or uncheck it and choose a manual time in **this PC's local timezone**. A recommendation in another region's timezone is automatically converted to the PC's local time.
4. Click **Schedule .exe**, read the full path/time, and confirm **Yes**. This creates **one Windows scheduled task**, without launching the executable immediately.
5. Open **View scheduled launches** to view or cancel a task. You may close SunPilot after registration.

**Important limitations:** Windows must be **awake**, the **same user must be signed in** at execution time, and the target `.exe` must remain at its chosen path. The program runs with the signed-in user's **normal permissions** (no administrator escalation). Windows policies may prevent task creation. Neither planned workload duration nor solar availability stops the launched program: SunPilot only schedules its **start**, and doesn't enforce a runtime. SunPilot cannot guarantee the program's successful execution or actual power/carbon savings.

The date/time picker for manual scheduling uses the PC's timezone. Only schedule programs you trust; you are responsible for what they do. To inspect the registered task in Windows, open **Task Scheduler**, search for `SunPilotAI_...`, and check its history.

### Release checklist

- Run the 23 tests and launch `sunpilot_desktop_v1.py` locally.
- Test both method choices for today and tomorrow and one different city.
- Test resizing and scroll, including wheel over the hour/minute inputs.
- Validate manual .exe scheduling on Windows (e.g. Notepad), then cancel a task.
- Build the Windows .exe and launch the **packaged executable** separately.
- Zip the full `dist/SunPilotAI` folder (not the .exe alone) for GitHub Releases.
- Clearly distinguish the validated historical research from the experimental live ML mode.

### Build Windows app (.exe)

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller tzdata
powershell -ExecutionPolicy Bypass -File .\build_sunpilot_v1.ps1
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

## Interface and accessibility

The desktop UI has three pages: **Plan a task**, **Simulator** and **Results**.
In Plan a task, location and advanced scientific controls are collapsed by
initial default to keep the interface uncluttered. All fields still support
keyboard input, and mouse-wheel movement over spinboxes, dates and dropdowns
will no longer modify their values unexpectedly. On small screens the page
scrolls vertically instead of clipping the controls.

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
- One-time Windows execution uses the built-in Task Scheduler, requires a
  user-approved executable, and is not a live power-consumption controller.
  The app can list and cancel its own scheduled tasks, but does not monitor
  CPU usage, electricity consumption, or the launched process after startup.
- No GPS access or location telemetry beyond user-triggered third-party requests.

## Source & acknowledgment

Weather: [Open-Meteo](https://open-meteo.com/) (data attribution, CC BY 4.0);
location search: Open-Meteo Geocoding API (GeoNames-derived data);
optional approximate IP lookup: [ipwho.is](https://ipwho.is/).

ChatGPT was used for planning, code generation, debugging and documentation.
The project team is responsible for understanding, testing and accurately
reporting the code and results.
