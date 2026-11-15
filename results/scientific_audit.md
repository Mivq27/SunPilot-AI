# SunPilot AI — corrected scheduling audit

## Time alignment

Open-Meteo hourly shortwave_radiation is a mean of the preceding hour. For real tasks from 08:00 to 18:00 local, the ten hourly labels used are **09:00–18:00**, not 08:00–17:00. This change corrects the v1.0 historical backtest.

## 2025 held-out test

AI vs previous-day scheduling: **+1.535%** (paired day bootstrap 95% interval: [0.126, 2.982]).

AI vs fixed noon: **-1.374%** (paired day bootstrap 95% interval: [-2.408, -0.377]).

## Limitations

- ERA5 reanalysis is not an archived operational forecast; live AI transfer has not been validated.
- Historical task lengths are exactly two hours in aggregate benchmark.
- Irradiance is not electricity production and cannot establish climate benefit alone.
- Hypothetical PV self-use assumes 4 m² of horizontal panels, 20% efficiency, 300 W computer, no batteries or shading; this is **not** measured or avoided CO2.
- Validation year (2024) and test year (2025) were not used for new model fitting here.
