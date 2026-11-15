# SunPilot AI

**Solar-aware scheduling for flexible computing workloads**  
**Veritas AI International High School Hackathon 2026 · AI + Climate**

[Download tested Windows v1.1](https://github.com/Mivq27/SunPilot-AI/releases/tag/v1.1) · [Source](https://github.com/Mivq27/SunPilot-AI)

*Previous Version* [SunPilot-AI v1.0](https://github.com/Mivq27/SunPilot-AI/releases/tag/v1.0)

> **Scientific audit update (candidate v1.1):** The original v1.0 historical scheduling evaluation used hour-end radiation values as if they marked the *start* of each hour. This package corrects that one-hour offset in both the simulator and offline backtest. **The original v1.0 scheduling percentages have been superseded by the re-evaluated v1.1 results below.** We do not retroactively modify or delete the original release; it remains available for reproducibility.

## The user and the decision

SunPilot is built for a person with a flexible computer workload, such as a long 3D render, who may also have rooftop solar. They choose their location, the job's duration and a start range. The app proposes a time with higher expected *incident sunlight*, then optionally schedules a trusted Windows `.exe` to start after explicit confirmation. It does not control power supply, stop the application or guarantee renewable electricity consumption.

## What makes this an AI project?

Our research component trains a `HistGradientBoostingRegressor` to predict hourly solar radiation from seasonal/hourly features and 24-, 48-, 168-hour lagged weather data. The model is fitted on **ERA5 2022–2023**, validated on **2024**, and evaluated on **2025**, with strictly chronological splits. The experiment compares AI with yesterday's radiation and a fixed-noon task schedule.

The **practical live default** instead uses an **Open-Meteo operational weather forecast** to recommend task start times. **SunPilot AI (experimental)** is a separate selectable method. The ERA5-trained model has not been rigorously validated on the operational forecast inputs or on other geographic regions. This distinction matters; using the weather API does not itself constitute our own AI innovation.

## Results — evidence, not advertising

**Frozen historical radiation forecast test, 2025, ERA5:**

| Forecast | MAE, all hours (W/m²) |
|---|---:|
| Previous-day persistence | 39.983 |
| SunPilot AI | **37.291** |

**2025 daytime-only MAE (08:00–18:00 physical hours):** AI **81.048 W/m²**, persistence **87.560 W/m²**. The original daytime mask had a one-hour offset and has been corrected without retraining.

**2025 held-out scheduling benchmark — 365 days, two-hour jobs (Wh/m² per task):**

| Strategy | Average observed incident irradiation |
|---|---:|
| SunPilot AI | **671.808** |
| Previous-day persistence | 661.649 |
| Fixed 12:00 | **681.170** |
| Perfect knowledge (unattainable upper bound) | 727.753 |

Our AI's mean scheduled irradiation is **1.535% higher than yesterday-based scheduling** (paired daily bootstrap 95% CI: **[+0.126%, +2.982%]**) but **1.374% lower than fixed noon** (95% CI: **[−2.408%, −0.377%]**). Against fixed noon, AI wins on **32.055%** of days, ties on **38.904%**, and loses on **29.041%**. These results do not establish AI superiority over fixed scheduling. They describe ERA5 historical simulations, not measured PV production or avoided emissions.

**2024 validation:** AI **+1.822%** versus yesterday (95% CI **[+0.114%, +3.706%]**) and **−1.483%** versus fixed noon (95% CI **[−2.555%, −0.428%]**). The corrected evaluations are available in `results/scheduler_metrics.json` and `results/scientific_audit.md`.

**Climate impact:** We have *not* measured PV power, grid carbon intensity, kWh saved or avoided CO₂. An optional **illustrative** horizontal-panel self-consumption calculation shows how one might translate solar irradiance into a simplified PV-use proxy for a specific user. This is a scenario, not evidence of real emissions reductions.

## Run on Windows

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe .\sunpilot_desktop_v1.py
```

The repository includes the model and ERA5 dataset used by the historical UI. The live planner and city search require internet. You can optionally request coarse IP-based location, but this is **not GPS**; the app does not request IP lookup at startup.

To schedule a program, open **Plan**, get a recommendation, then open **Automate** and choose a trusted `.exe`. Review the final time and confirm. Windows must be awake and the same user signed in when it starts. On another PC, system permissions may prevent scheduling.

## Scientific verification before shipping another Windows ZIP

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q .\tests
.\.venv\Scripts\python.exe .\src\recheck_metrics.py
.\.venv\Scripts\python.exe .\src\scheduler.py
.\.venv\Scripts\python.exe .\scripts\verify_release.py
powershell -ExecutionPolicy Bypass -File .\build_sunpilot_v11.ps1
```

If the prediction CSV files do not exist (they are generated, not committed), run `python src/train.py` **using the existing frozen model specification** first. This retraining must be explicitly acknowledged; do not change the hyperparameters after seeing the 2025 result. The new scheduler uses existing saved predictions whenever available and does **not** train on test data.

The build script intentionally refuses to package old scheduling metrics. **Do not upload a new release without checking the new metrics and the GUI/Task Scheduler in Windows.**

## Reproducibility

- `src/download_data.py`: ERA5 historical data from Open-Meteo, 35,064 hourly rows, Klippan (Sweden), 2022–2025.
- `src/train.py`: original chronological ML training and MAE evaluation.
- `src/solar_time.py`: the hour-end timestamp semantics for 08:00–18:00 tasks.
- `src/recheck_metrics.py`: corrects the daytime prediction MAE mask without refitting.
- `src/scheduler.py`: corrected two-hour historical decision backtest.
- `src/pv_scenario.py`: openly hypothetical PV self-consumption illustration.
- `src/live_planner.py`: location-aware live time planner, with incomplete hours excluded.
- `requirements.txt` / `requirements-desktop.txt`: focused dependencies; no longer installs legacy Gradio/WebEngine stacks for a native Qt app.
- `src/windows_scheduler.py`: explicit one-time Windows task registration.
- `tests/`: unit tests plus regression tests for timestamp alignment and missing forecast rows.
- `docs/SCIENTIFIC_AUDIT.md`: interpretations, known limitations, and a proposed real-time validation protocol.
- `docs/JUDGE_WRITEUP.md`: concise one-page submission narrative.

## Main limitations

1. The climate benefit is a **hypothesis**, not a measured result: maximizing solar radiation does not necessarily minimize electricity-grid carbon intensity.
2. ERA5 is reanalysis, not a physical PV meter and not an archived operational prediction.
3. The historical model's deployment on live forecast inputs and arbitrary locations has not been validated.
4. A 15-minute start grid and minute-length tasks are **integrated from hourly weather data**; these are approximations, not true one-minute forecasts.
5. Windows scheduling starts a program once but does not monitor its completion or total electricity consumption.

## Credits

- Weather/reanalysis/operational forecasts: [Open-Meteo](https://open-meteo.com/) and underlying ERA5. Respect the provider's data licence and attribution rules.
- Forecast archives for future evaluation: [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) and [Single Runs API](https://open-meteo.com/en/docs/single-runs-api).
- Python, pandas, NumPy, scikit-learn, PySide6, Matplotlib, PyInstaller and other open-source dependencies.
- **ChatGPT was used extensively for planning, implementation, debugging, evaluation design and documentation.** The author is responsible for understanding the code and honestly reporting its results.

This is a research prototype for scheduling **potential** solar availability, not a verified energy-saving system.
