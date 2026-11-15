# SunPilot AI

**Track:** AI + Climate  
**Team:** Enrique Lamas

## The problem

People with flexible computing workloads—such as a long 3D render—often start them immediately, even though they might be able to delay them. Someone with rooftop solar may prefer to run these jobs when sunlight availability is higher. We wanted to test whether machine learning could help select better times and make the result usable from a Windows desktop.

## What we built

SunPilot is a Windows desktop application with two complementary functions. **Plan** accepts a location, future date range and task duration, then displays a time recommendation using a weather forecast or our experimental machine-learning model. **Automate** can optionally register a one-time launch of a user-approved `.exe` using Windows Task Scheduler. The **Simulator** compares predictions and selected times with historical ERA5 reference data. **Research** explains our methodology and failures.

## What we added

We extended a local solar-radiation forecasting experiment into a decision-based simulator and Windows task scheduler. We also tested prediction accuracy and the effect of those predictions on scheduling decisions, rather than reporting model accuracy alone.

## How it works

We downloaded **35,064 hourly records** of Open-Meteo ERA5 data for Klippan, Sweden, from 2022–2025. A scikit-learn `HistGradientBoostingRegressor` uses local hour and season and 24-, 48- and 168-hour weather/radiation lags. We trained on 2022–2023, validated on 2024 and kept 2025 as an out-of-sample test. No random time shuffling is used.

## Results and where it fails

On the historical 2025 **all-hours** test, SunPilot's radiation MAE was **37.291 W/m²** versus **39.983 W/m²** for a simple previous-day baseline (about **6.73% lower error**). Better prediction does **not** guarantee better decisions. We discovered that our original scheduling evaluation had a one-hour interpretation error: hourly radiation describes the **preceding** hour. We corrected the simulator and backtest using end-of-hour-aligned radiation. For 365 held-out days in 2025, with two-hour jobs, our AI achieved **671.808 Wh/m²** of incident irradiation per scheduled task, compared with **661.649 Wh/m²** for the previous-day schedule and **681.170 Wh/m²** for fixed noon. Relative to yesterday, AI improved **+1.535%** (95% paired bootstrap CI **[+0.126%, +2.982%]**); relative to fixed noon, it was **−1.374%** (95% CI **[−2.408%, −0.377%]**). AI does **not** beat fixed noon. After correcting the physical-hour alignment, its daytime MAE was **81.048 W/m²** versus **87.560 W/m²** for previous-day persistence. The perfect-information upper bound was **727.753 Wh/m²**; it is not an achievable forecast. These are simulations of incident irradiation, not PV generation or carbon savings.

The experimental AI has **not** been independently validated on operational forecast data or outside Klippan. The practical default uses the Open-Meteo weather forecast directly. No actual PV consumption or CO₂ savings were measured. We include a clearly hypothetical rooftop-PV self-consumption scenario, but not as real emissions evidence.

## What we would do next

Reconstruct archival operational forecasts available at each historical scheduling decision, compare our model to the forecast itself and to fixed noon, and validate the system with real rooftop PV output and measured computer power usage. Only deploy an AI recommendation as default if it improves operational decisions on unseen data.

## Sources and AI credits

Open-Meteo historical ERA5 and Forecast APIs; Open-Meteo Previous Runs / Single Runs forecast archives (planned validation); Python, pandas, NumPy, scikit-learn, PySide6, Matplotlib, PyInstaller. **ChatGPT was used extensively for design, code generation, testing and technical documentation.** Source and reproducible tests: https://github.com/Mivq27/SunPilot-AI.

