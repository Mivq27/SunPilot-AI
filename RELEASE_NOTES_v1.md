# SunPilot AI v1.0 — Windows desktop release candidate

Native Windows application for solar-aware scheduling of flexible computing jobs.

## Highlights

- Clean **Plan**, **Automate**, **Simulator**, **Research** navigation, without numbered tabs.
- Prominent choice between **Weather forecast (recommended)** and **SunPilot AI (experimental)**.
- Location lookup by city, optional approximate IP geolocation, or manual coordinates.
- Task duration in hours and minutes; recommends starts on 15-minute boundaries, using interpolated hourly forecast information.
- One-time Windows `.exe` launch scheduling **only after user confirmation**.
- Historical 2024–2025 validation/backtest and model metrics.
- Mouse wheel scrolls the page without accidentally changing numeric controls.

## Research honesty

On 2025 held-out historical data, model prediction MAE was 37.29 W/m² versus 39.98 W/m² for yesterday baseline. Scheduling with our AI was 1.47% ahead of yesterday but 0.18% below fixed noon for 2-hour tasks. Live AI is not yet validated on archived operational forecasts; no actual PV power/carbon savings are claimed.

## Install on Windows

Download `SunPilotAI-Windows-v1.0.zip`, extract it and open `SunPilotAI/SunPilotAI.exe`.
The full `SunPilotAI` folder is needed. Internet is required for live forecasts.
Windows must be awake and signed in for scheduled program launches.

## Source

https://github.com/Mivq27/SunPilot-AI
