# SunPilot AI

**Solar-Aware Compute Scheduling**

An experimental AI-powered Windows desktop application that predicts solar radiation and recommends when to execute flexible computing workloads.

Built for the Veritas AI International High School Hackathon 2026.

**Track:** AI + Climate

## Features

- Native Windows desktop application using PySide6
- Historical solar radiation simulator (2024–2025)
- Machine learning predictions using scikit-learn
- Comparison against scheduling baselines
- Experimental live planning using Open-Meteo forecasts
- Interactive charts and scheduling recommendations

## Research Question

Can machine learning predictions improve solar-aware
computing schedules compared with simple scheduling rules?

## Machine Learning

Model: HistGradientBoostingRegressor

Training: 2022–2023
Validation: 2024
Test: 2025

Dataset: ERA5 historical weather data for Klippan, Sweden.

### Prediction results (2025)

| Model | MAE (W/m²) |
|---|---:|
| Yesterday baseline | 39.98 |
| SunPilot AI | 37.29 |

SunPilot achieved approximately 6.73% lower overall MAE
than the yesterday-based prediction baseline.

### Scheduling results (2025)

Average solar irradiation per two-hour task:

| Strategy | Wh/m² |
|---|---:|
| Fixed noon schedule | 673.01 |
| Yesterday baseline | 662.06 |
| SunPilot AI | 671.81 |
| Perfect knowledge (upper bound) | 727.45 |

SunPilot improved over yesterday-based scheduling by
1.47%, but performed 0.18% worse than the fixed noon
schedule.

Better forecasting does not necessarily imply better
scheduling decisions.

## Running the Desktop Application

Install Python 3.13 and the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the desktop application:

```powershell
python sunpilot_desktop_v2.py
```

The historical simulator works with the included dataset.
The live planner requires internet access.

## Reproducing the Experiments

```powershell
python src/download_data.py
python src/train.py
python src/scheduler.py
```

## Limitations

- Historical weather data is based on ERA5 reanalysis.
- The live forecasting mode is experimental and has not
  been validated against archived operational forecasts.
- Solar radiation is not equivalent to actual PV production.
- No verified carbon-emission savings are claimed.
- The application recommends schedules but does not
  automatically execute system tasks.
- The perfect-knowledge strategy uses observed values
  and is not an achievable forecast.

## Data Attribution

Weather data by Open-Meteo.com

https://open-meteo.com/

Data licensed under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/

## Technologies

Python, scikit-learn, pandas, NumPy, PySide6,
Matplotlib, Open-Meteo and PyInstaller.

ChatGPT was used as a development assistant for
planning, code generation, debugging and documentation.
The team is responsible for understanding and
validating the implementation.

## Status

Hackathon prototype — November 2026.
