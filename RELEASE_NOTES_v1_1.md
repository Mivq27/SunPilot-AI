# SunPilot AI v1.1 — Scientific audit release candidate

**This update supersedes the v1.0 historical evaluation, but retains the original v1.0 release unchanged.**

## Improvements

- Corrected one-hour radiation timestamp offset: an end-of-hour solar mean labelled 09:00 describes 08:00–09:00. The historical simulator and backtest now use the same physical job intervals.
- Corrected daytime-only MAE window and provided a no-retraining tool to recompute it from frozen model predictions.
- Added reproducible paired bootstrap uncertainty intervals, daily win/tie/loss rates, worst-case day, fixed-noon comparison, and corrected benchmark chart.
- Added **illustrative** rooftop PV direct self-consumption potential for an openly specified area/efficiency/computing power scenario. NOT measured electricity, financial, or CO₂ savings.
- Fixed missing forecast timestamp handling and removed deprecated Pandas accessors.
- Extended unit and scientific regression tests.
- Added a release gate that refuses to compile while obsolete metrics are present.
- Improved documentation of the AI's experimental nature, domain shift, and operational-data validation plan.

## Verified historical test results (2025)

- All-hours radiation prediction MAE: AI **37.291 W/m²**, yesterday **39.983 W/m²** (**6.73%** relative improvement).
- Daytime (08:00–18:00 physical hours) MAE: AI **81.048 W/m²**, yesterday **87.560 W/m²**.
- Mean observed incident solar irradiation per 2-hour scheduled job: AI **671.808 Wh/m²**, yesterday **661.649 Wh/m²**, fixed noon **681.170 Wh/m²**.
- AI vs yesterday **+1.535%** (paired bootstrap 95% CI **[+0.126%, +2.982%]**).
- AI vs fixed noon **−1.374%** (95% CI **[−2.408%, −0.377%]**).
- The result **does not** demonstrate that AI improves on a simple fixed-noon schedule.

## Required before upload

1. Run the refreshed metrics and scheduling evaluator using the original saved predictions.
2. Read the new figures; do not reuse v1.0 historical scheduling percentages.
3. Confirm the GUI in Windows, especially simulator charts and live forecast behavior.
4. Smoke-test the scheduled .exe using a trusted program.
5. Rebuild a new ZIP and publish **v1.1**, not overwrite the existing v1.0 tag.

**No claim of proven carbon savings or improved live-forecast accuracy is made.**
