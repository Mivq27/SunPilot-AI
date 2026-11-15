# SunPilot AI — Scientific and engineering audit

## 1. Correcting time semantics

Open-Meteo documents `shortwave_radiation` as the *average over the preceding hour*, not an instantaneous reading or the hour that follows the label. Thus:

| End timestamp | Interval represented |
|---|---|
| 09:00 | 08:00–09:00 |
| 13:00 | 12:00–13:00 |
| 14:00 | 13:00–14:00 |

A **12:00–14:00** two-hour computing task must be scored using observations timestamped **13:00 and 14:00**. The v1.0 algorithm incorrectly read 12:00 and 13:00. Both the offline `src/scheduler.py` and the interactive simulator now share the same corrected convention, with 09:00–18:00 end-labels representing real operations 08:00–18:00. This is checked by synthetic tests.

Corrected scheduling values cannot be derived from the previous aggregate metrics: they **must be recomputed** from hourly held-out predictions. The report is intentionally unpublished until that computation is run. **Overall** prediction MAEs do not change, because each forecast was always evaluated against radiation at its own timestamp. **Daytime-only MAEs do change**: the original daytime mask also selected end-hour labels 08:00–17:00, which refer to physical intervals 07:00–17:00. `src/recheck_metrics.py` corrects the daytime mask to 09:00–18:00 without fitting or tuning any model.

Source: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

## 2. What the forecasting experiment actually proves

- Geographic target: Klippan, Sweden. Historical timestamps and ERA5 values, 2022–2025.
- ML: gradient-boosted regression with nine features. Train: 2022–2023; validation: 2024; test: 2025.
- Prediction performance in 2025: SunPilot 37.291 W/m² overall MAE vs 39.983 W/m² persistence MAE (6.73% improvement).
- This does **not** establish superior scheduling against a fixed-noon heuristic or a professional weather model.
- The corrected decision test uses the exact two-hour clock interval, fixed-noon and yesterday baselines, day win/tie/loss rates, worst-case daily failure, and paired day bootstrap 95% descriptive intervals.
- Do not tune the predictor to the 2025 test or switch metrics to find a favorable score.

## 3. Who might actually benefit?

**Specific user:** an owner of rooftop photovoltaic panels running flexible computer work (for example offline 3D renders), who can choose a start time and who wants to maximize *potential concurrent solar availability*.

**Current input:** user location, start range, duration, optional Windows .exe; operational Open-Meteo forecast for the main recommended mode; experimental ML for comparison.

**Output:** proposed start/end time, predicted incident solar irradiation in Wh/m², optional one-time Windows start task, historical backtest explanation.

**What is not established:** whether any particular system supplies that job with extra on-site renewable electricity, or whether the recommendation causes avoided carbon emissions. These require a power meter or PV inverter logs, workload demand in watts, household baseline consumption, panel geometry, and (for grid claims) time-varying grid carbon-intensity data.

## 4. A transparent illustrative energy-use scenario

The corrected evaluator also calculates, as a **separate hypothetical metric**, direct PV-powered energy for a 2-hour job under these freely chosen assumptions:

- Horizontal panel equivalent area: **4 m²**.
- Nominal conversion efficiency: **20%**.
- Fixed computing electricity demand: **300 W**.
- No battery, shading, inverter losses or other household loads.

For each full hour, the *proxy* of PV-supplied job energy is `min(GHI * 4 m² * 0.20, 300 W) × 1 hour / 1000`. Because production and consumption are capped, this differs from merely maximizing total GHI and gives a concrete decision-oriented analysis. It is not measured, and it is not CO₂ avoided. Changing these assumptions will change any result.

For a real impact study, integrate measured actual solar and demand at >=15-minute resolution, compare the result with a fixed baseline, and perform a holdout pilot on a roof-PV installation.

## 5. Why operational AI remains unvalidated

The live app receives a current weather forecast from Open-Meteo. The model was trained on ERA5 reanalysis and uses shifted provider inputs in the live mode. This **domain shift** makes its historical MAE an unreliable estimate of live performance, especially away from Klippan. It cannot be fixed by simply deploying the pretrained model or renaming the weather API as AI.

### Valid future experiment (do not claim it has already run)

1. Obtain **historical operational model runs** (ideally complete [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) snapshots at the decision time) and align their issue times, target valid times, forecasts and ERA5 verifying values.
2. Define a forecast horizon (e.g. scheduling next-day 08:00–18:00 at 20:00 the evening before), and preserve *only data available at issue time*.
3. Compare direct provider forecast, simple fixed-noon and yesterday baselines, plus a new forecast-calibration ML model trained with **earlier archived forecasts**, validated on later forecasts and tested on untouched recent forecasts.
4. Report both radiation MAE and operational scheduling outcome, with a confidence interval and at least one failure case.
5. Deploy AI as a recommended method only if it provides a robust decision advantage on unseen operational data. Otherwise retain Open-Meteo direct scheduling and keep the AI experimental.

The Open-Meteo [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) supports fixed forecast lead-time comparisons, but one should use **Single Runs** to reconstruct an exact historic decision and prevent lookahead bias.

## 6. Architecture and QA risks

- `sunpilot_desktop_v1.py` is a ~1,800-line GUI file. This is acceptable for a 48-hour student prototype, but in a maintained product split it into views, view models, data/service layers and a reusable plotting module.
- `tests/test_v1_ui_contract.py` only checks source structure, not screenshot layout, screen-reader accessibility or end-to-end interaction.
- `tests/test_windows_scheduler.py` mocks system calls; it **does not** demonstrate a successful real Task Scheduler launch. A physical Windows smoke test is essential.
- An unsigned Windows .exe may trigger SmartScreen. It must be extracted with its `_internal` directory.
- Privacy: IP geolocation is opt-in and approximate. Never turn on precise GPS or transmit personal data without consent.
- Release versioning: preserve v1.0 and publish a new v1.1 once the scientific results and binary are verified, rather than silently changing a published release.

## 7. Release gate

The `scripts/verify_release.py` command will fail if the `results/scheduler_metrics.json` still contains the old (uncorrected) metrics. Recalculate from the existing prediction CSVs, inspect the newly generated 2025 results, then run the gate before building a new Windows ZIP.
