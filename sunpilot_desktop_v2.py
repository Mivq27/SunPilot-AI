"""SunPilot AI Desktop v2 — native Qt app with historical + live planning.

Historical mode uses the trained ERA5-based model and historical measurements.
Live mode downloads forecast-provider data, builds the original lagged features,
and runs the same model. Live mode is exploratory and not yet validated.

Run from the SunPilot-AI project root using Python 3.13.
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from PySide6.QtCore import QDate, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDateEdit, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Data files bundled by PyInstaller live in sys._MEIPASS.
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
os.chdir(ROOT)
from src.train import FEATURES, load_data  # noqa: E402

LAT = 56.1356
LON = 13.1309
LOCAL_TZ = ZoneInfo("Europe/Stockholm")
START_HOUR, END_HOUR = 8, 18
FIXED_START = 12
BG = "#0b1020"
SURFACE = "#151c2f"
TEXT = "#edf2fc"
MUTED = "#9ca9bf"
ORANGE = "#f97316"
BLUE = "#38bdf8"
GREEN = "#34d399"


class LiveForecastWorker(QThread):
    data_ready = Signal(object)
    error = Signal(str)

    def run(self) -> None:
        try:
            # Forecast API gives current and recent forecast-model estimates.
            # past_days are NOT asserted to be ground-truth measurements.
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": LAT,
                    "longitude": LON,
                    "timezone": "UTC",
                    "past_days": 8,
                    "forecast_days": 3,
                    "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
                },
                timeout=(10, 35),
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(payload.get("reason", "Open-Meteo error"))
            hourly = payload.get("hourly", {})
            fields = ["time", "shortwave_radiation", "cloud_cover", "temperature_2m"]
            for field in fields:
                if field not in hourly:
                    raise ValueError(f"Missing forecast variable: {field}")
            frame = pd.DataFrame(hourly)[fields].rename(columns={"time": "time_utc"})
            frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
            frame = frame.sort_values("time_utc").drop_duplicates("time_utc")
            frame = frame.set_index("time_utc")
            for name in fields[1:]:
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
            if frame.empty:
                raise RuntimeError("Forecast API returned no hourly data")
            self.data_ready.emit(frame)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


def best_start(values: np.ndarray, duration: int, possible_starts: list[int]) -> int:
    if not possible_starts:
        raise ValueError("No available time slots for this task today")
    scores = {
        start: float(np.sum(values[start-START_HOUR:start-START_HOUR+duration]))
        for start in possible_starts
    }
    top = max(scores.values())
    tied = [start for start, score in scores.items() if np.isclose(score, top, atol=1e-8)]
    return min(tied, key=lambda start: abs(start-12))


def model_features_for_day(frame: pd.DataFrame, local_day: str) -> pd.DataFrame:
    """Build the SAME features used during training, using forecast-provider inputs.

    Feature values at t are read exclusively from t-24h, t-48h or t-168h,
    and date/time. The provider inputs for those lags may themselves be
    forecasts, not actual observed data; this domain shift is disclosed in UI.
    """
    day_index = frame.index.tz_convert("Europe/Stockholm")
    mask = np.array([timestamp.strftime("%Y-%m-%d") == local_day
                     and START_HOUR <= timestamp.hour < END_HOUR
                     for timestamp in day_index])
    selected = frame.loc[mask].copy().sort_index()
    hours = selected.index.tz_convert("Europe/Stockholm").hour.tolist()
    if hours != list(range(START_HOUR, END_HOUR)):
        raise RuntimeError(f"Forecast coverage is incomplete for {local_day}: {hours}")

    rows = []
    for timestamp in selected.index:
        local_ts = timestamp.tz_convert("Europe/Stockholm")
        hour = local_ts.hour
        day_of_year = local_ts.dayofyear
        lag24 = timestamp - pd.Timedelta(hours=24)
        lag48 = timestamp - pd.Timedelta(hours=48)
        lag168 = timestamp - pd.Timedelta(hours=168)
        for lag in (lag24, lag48, lag168):
            if lag not in frame.index:
                raise RuntimeError(f"Missing historical input for {timestamp}: {lag}")
        rows.append({
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "season_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "season_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "radiation_lag_24": frame.loc[lag24, "shortwave_radiation"],
            "radiation_lag_48": frame.loc[lag48, "shortwave_radiation"],
            "radiation_lag_168": frame.loc[lag168, "shortwave_radiation"],
            "cloud_lag_24": frame.loc[lag24, "cloud_cover"],
            "temperature_lag_24": frame.loc[lag24, "temperature_2m"],
        })
    features = pd.DataFrame(rows, columns=FEATURES)
    if features.isna().any().any():
        raise RuntimeError("Forecast contains missing values in ML model inputs")
    return features


class SunPilotWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SunPilot AI | Solar-Aware Compute Scheduling")
        self.resize(1210, 880)
        self.setMinimumSize(970, 715)

        artifact_path = ROOT / "models" / "sunpilot_v1.joblib"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Trained model missing: {artifact_path}")
        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        if list(artifact["features"]) != list(FEATURES):
            raise ValueError("Model features differ from src.train.FEATURES")

        data = load_data()
        data = data[data["local_date"].str[:4].isin(["2024", "2025"])].copy()
        if data.empty:
            raise RuntimeError("Historical data for 2024–2025 is missing")
        data["ai_prediction"] = np.maximum(self.model.predict(data[FEATURES]), 0)
        self.data = data
        self.worker: LiveForecastWorker | None = None
        self.recommendation_text = ""

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(24, 18, 24, 18)
        outer.setSpacing(12)
        header = QLabel("SUNPILOT  <span style='color:#f97316'>AI</span>")
        header.setObjectName("title")
        header.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(header)
        intro = QLabel("Solar-Aware Compute Scheduling  /  Klippan, Sweden  /  Veritas AI Hackathon 2026")
        intro.setObjectName("muted")
        outer.addWidget(intro)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.build_historical_tab()
        self.build_live_tab()
        self.setStyleSheet(f"""
            QWidget, QMainWindow {{ background:{BG}; color:{TEXT}; font-size:13px; }}
            QLabel#title {{ font-size:28px; font-weight:800; }}
            QLabel#muted {{ color:{MUTED}; font-size:12px; }}
            QLabel#summary {{ background:{SURFACE}; padding:12px 15px;
                             border:1px solid #283149; border-radius:8px; }}
            QFrame#panel {{ background:{SURFACE}; border:1px solid #283149; border-radius:9px; }}
            QDateEdit, QSpinBox, QComboBox {{ background:#212b42; color:{TEXT};
                            border:1px solid #38445f; border-radius:6px; padding:7px 10px; }}
            QPushButton {{ color:{TEXT}; background:#29334c; border:1px solid #3b4a67;
                           border-radius:6px; padding:10px 18px; }}
            QPushButton#primaryButton {{ background:{ORANGE}; color:white;
                                        border:0px; font-weight:700; }}
            QPushButton#primaryButton:hover {{ background:#ea580c; }}
            QPushButton:disabled {{ background:#3d4656; color:#a2aab7; }}
            QTabWidget::pane {{ border:0px; }}
            QTabBar::tab {{ background:{SURFACE}; color:{MUTED}; padding:10px 22px; }}
            QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {ORANGE}; }}
            QTableWidget {{ background:{SURFACE}; alternate-background-color:#1b2540;
                           color:{TEXT}; gridline-color:#2a3650; border:1px solid #283149; }}
            QHeaderView::section {{ background:#222b42; color:{TEXT}; border:0px;
                                    padding:8px; font-weight:bold; }}
        """)
        self.analyze_historical()

    @staticmethod
    def table(rows: int, headers: list[str]) -> QTableWidget:
        widget = QTableWidget(rows, len(headers))
        widget.setHorizontalHeaderLabels(headers)
        widget.verticalHeader().setVisible(False)
        widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        widget.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        widget.setAlternatingRowColors(True)
        widget.setFixedHeight(168)
        widget.horizontalHeader().setStretchLastSection(True)
        widget.setColumnWidth(0, 225)
        widget.setColumnWidth(1, 240)
        return widget

    @staticmethod
    def put_rows(widget: QTableWidget, rows: list[list[str]]) -> None:
        widget.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                item = QTableWidgetItem(value)
                if j == 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                widget.setItem(i, j, item)

    def build_historical_tab(self) -> None:
        page = QWidget()
        self.tabs.addTab(page, "Historical Simulator")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 13, 4, 6)
        layout.setSpacing(12)
        row = QFrame()
        row.setObjectName("panel")
        controls = QHBoxLayout(row)
        controls.addWidget(QLabel("Historical date"))
        self.date = QDateEdit(QDate(2025, 8, 26))
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.date.setCalendarPopup(True)
        self.date.setDateRange(QDate(2024, 1, 1), QDate(2025, 12, 31))
        controls.addWidget(self.date)
        controls.addWidget(QLabel("Task duration"))
        self.duration = QSpinBox()
        self.duration.setRange(1, 4)
        self.duration.setSuffix(" h")
        self.duration.setValue(2)
        controls.addWidget(self.duration)
        self.analyze_button = QPushButton("Analyze with AI")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self.analyze_historical)
        controls.addWidget(self.analyze_button)
        controls.addStretch(1)
        layout.addWidget(row)
        self.summary = QLabel()
        self.summary.setObjectName("summary")
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.history_fig = Figure(figsize=(11.2, 5.6), facecolor=BG)
        self.history_canvas = FigureCanvas(self.history_fig)
        self.history_canvas.setMinimumHeight(310)
        layout.addWidget(self.history_canvas, 1)
        self.history_table = self.table(4, ["Strategy", "Scheduled time", "Observed irradiation (Wh/m²)"])
        layout.addWidget(self.history_table)
        hint = QLabel("Historical ERA5 backtest • actual observations are only used to score decisions • 2025 is the held-out test year")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def analyze_historical(self) -> None:
        try:
            date = self.date.date().toString("yyyy-MM-dd")
            duration = self.duration.value()
            day = self.data[(self.data["local_date"] == date)
                            & self.data["local_hour"].between(START_HOUR, END_HOUR-1)]
            day = day.sort_values("local_hour")
            if day["local_hour"].tolist() != list(range(START_HOUR, END_HOUR)):
                raise ValueError(f"Incomplete historical day: {date}")
            actual = day["shortwave_radiation"].to_numpy(dtype=float)
            ai = day["ai_prediction"].to_numpy(dtype=float)
            previous = day["radiation_lag_24"].to_numpy(dtype=float)
            starts = list(range(START_HOUR, END_HOUR-duration+1))
            chosen = {
                "SunPilot AI": best_start(ai, duration, starts),
                "Yesterday baseline": best_start(previous, duration, starts),
                "Fixed noon": FIXED_START,
                "Perfect knowledge *": best_start(actual, duration, starts),
            }
            def observed_sum(start: int) -> float:
                return float(sum(actual[start-START_HOUR:start-START_HOUR+duration]))
            results = {name: observed_sum(start) for name, start in chosen.items()}
            selected = chosen["SunPilot AI"]
            gain_fixed = results["SunPilot AI"] - results["Fixed noon"]
            gain_yesterday = results["SunPilot AI"] - results["Yesterday baseline"]
            self.summary.setText(
                f"<span style='color:{MUTED}'>AI selected</span> &nbsp; "
                f"<b style='font-size:18px'>{selected:02d}:00–{selected+duration:02d}:00</b> &nbsp; "
                f"<span style='color:{MUTED}'>vs fixed</span> {gain_fixed:+.1f} Wh/m² &nbsp; "
                f"<span style='color:{MUTED}'>vs yesterday</span> {gain_yesterday:+.1f} Wh/m²"
            )
            self.put_rows(self.history_table, [
                [name, f"{start:02d}:00–{start+duration:02d}:00", f"{results[name]:.1f}"]
                for name, start in chosen.items()
            ])
            self.plot_historical(actual, ai, previous, chosen, results, duration, date)
            self.statusBar().showMessage(f"Historical backtest: {date} / {duration}h / 08–18 local")
        except Exception as exc:
            QMessageBox.critical(self, "Historical simulation error", str(exc))

    def build_live_tab(self) -> None:
        page = QWidget()
        self.tabs.addTab(page, "Live Solar Planner")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 13, 4, 6)
        layout.setSpacing(12)
        bar = QFrame()
        bar.setObjectName("panel")
        controls = QHBoxLayout(bar)
        controls.addWidget(QLabel("Schedule for"))
        self.live_day = QComboBox()
        self.live_day.addItems(["Today", "Tomorrow"])
        controls.addWidget(self.live_day)
        controls.addWidget(QLabel("Task duration"))
        self.live_duration = QSpinBox()
        self.live_duration.setRange(1, 4)
        self.live_duration.setSuffix(" h")
        self.live_duration.setValue(2)
        controls.addWidget(self.live_duration)
        self.live_button = QPushButton("Get forecast & plan")
        self.live_button.setObjectName("primaryButton")
        self.live_button.clicked.connect(self.fetch_live)
        controls.addWidget(self.live_button)
        self.copy_button = QPushButton("Copy recommendation")
        self.copy_button.clicked.connect(self.copy_recommendation)
        self.copy_button.setEnabled(False)
        controls.addWidget(self.copy_button)
        controls.addStretch(1)
        layout.addWidget(bar)
        self.live_summary = QLabel("Press 'Get forecast & plan' to fetch today's weather outlook and run the trained AI model.")
        self.live_summary.setObjectName("summary")
        self.live_summary.setTextFormat(Qt.TextFormat.RichText)
        self.live_summary.setWordWrap(True)
        layout.addWidget(self.live_summary)
        self.live_fig = Figure(figsize=(11.2, 5.5), facecolor=BG)
        self.live_canvas = FigureCanvas(self.live_fig)
        self.live_canvas.setMinimumHeight(290)
        layout.addWidget(self.live_canvas, 1)
        self.live_table = self.table(3, ["Strategy", "Suggested time", "Forecast-estimated irradiation (Wh/m²)"])
        layout.addWidget(self.live_table)
        caveat = QLabel(
            "LIVE EXPERIMENTAL MODE • Requires internet. The trained AI uses past forecast-model "
            "estimates as proxies for historical ERA5 inputs. Performance has NOT been validated "
            "in this live setting. Values are predicted, not actual PV production or CO₂ savings. "
            "This app recommends a time; it does not launch programs."
        )
        caveat.setObjectName("muted")
        caveat.setWordWrap(True)
        layout.addWidget(caveat)

    def fetch_live(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.live_button.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.live_summary.setText("Downloading weather forecasts from Open-Meteo…")
        self.worker = LiveForecastWorker()
        self.worker.data_ready.connect(self.render_live)
        self.worker.error.connect(self.live_error)
        self.worker.finished.connect(self.live_finished)
        self.worker.start()

    def live_finished(self) -> None:
        self.live_button.setEnabled(True)
        self.worker = None

    def live_error(self, message: str) -> None:
        self.live_summary.setText("Forecast request failed. Check your internet connection and try again.")
        QMessageBox.warning(self, "Live forecast error", message)

    def render_live(self, forecast: pd.DataFrame) -> None:
        try:
            now = datetime.now(LOCAL_TZ)
            target = (now.date() + timedelta(days=self.live_day.currentIndex()))
            target_day = target.isoformat()
            duration = self.live_duration.value()
            features = model_features_for_day(forecast, target_day)
            target_index = forecast.index.tz_convert("Europe/Stockholm")
            subset = forecast.loc[[dt.date() == target and START_HOUR <= dt.hour < END_HOUR
                                   for dt in target_index]].sort_index()
            direct_forecast = subset["shortwave_radiation"].to_numpy(dtype=float)
            ai_prediction = np.maximum(self.model.predict(features), 0)
            previous = features["radiation_lag_24"].to_numpy(dtype=float)
            if not np.isfinite(direct_forecast).all() or not np.isfinite(ai_prediction).all():
                raise ValueError("Forecast contains missing or invalid solar radiation values")

            starts = list(range(START_HOUR, END_HOUR-duration+1))
            if target == now.date():
                # Do not recommend past times: next whole hour if already in an hour.
                minimum_start = now.hour + (1 if (now.minute or now.second or now.microsecond) else 0)
                starts = [s for s in starts if s >= minimum_start]
            if not starts:
                raise RuntimeError("No eligible time slots remain today. Choose Tomorrow or a shorter task.")

            ai_start = best_start(ai_prediction, duration, starts)
            provider_start = best_start(direct_forecast, duration, starts)
            yesterday_start = best_start(previous, duration, starts)
            fixed_start = FIXED_START if FIXED_START in starts else None
            selected_times = [
                ("SunPilot AI (experimental)", ai_start),
                ("Open-Meteo forecast", provider_start),
                ("Yesterday proxy", yesterday_start),
            ]
            if fixed_start is not None:
                selected_times.append(("Fixed noon", fixed_start))

            def estimated_energy(start: int) -> float:
                k = start - START_HOUR
                return float(np.sum(direct_forecast[k:k+duration]))
            rows = [
                [name, f"{start:02d}:00–{start+duration:02d}:00", f"{estimated_energy(start):.1f}"]
                for name, start in selected_times
            ]
            self.put_rows(self.live_table, rows)
            self.live_table.setFixedHeight(min(230, 40 + 35 * len(rows)))
            difference = estimated_energy(ai_start) - estimated_energy(provider_start)
            self.recommendation_text = (
                f"SunPilot AI (experimental) — {target_day}, Klippan, Sweden. "
                f"Schedule a {duration}h task at {ai_start:02d}:00–{ai_start+duration:02d}:00 "
                f"Europe/Stockholm. This is a forecast-based recommendation, not a guarantee."
            )
            self.copy_button.setEnabled(True)
            self.live_summary.setText(
                f"<span style='color:{MUTED}'>AI recommended time for {target_day}</span> &nbsp; "
                f"<b style='font-size:19px'>{ai_start:02d}:00–{ai_start+duration:02d}:00</b> &nbsp; "
                f"<span style='color:{MUTED}'>vs forecast-optimal (estimated)</span> {difference:+.1f} Wh/m²"
            )
            self.plot_live(ai_prediction, direct_forecast, previous, ai_start, duration, target_day)
            self.statusBar().showMessage(
                f"Live plan: {target_day} / {duration}h / forecast retrieved from Open-Meteo / not validated"
            )
        except Exception as exc:
            self.live_error(f"{type(exc).__name__}: {exc}")

    def copy_recommendation(self) -> None:
        if self.recommendation_text:
            QApplication.clipboard().setText(self.recommendation_text)
            self.statusBar().showMessage("Recommendation copied to clipboard")

    @staticmethod
    def prepare_axes(fig: Figure):
        fig.clear()
        top, bottom = fig.subplots(2, 1, gridspec_kw={"height_ratios": [2.2, 1.0]})
        fig.subplots_adjust(left=0.18, right=0.97, top=0.91, bottom=0.11, hspace=0.55)
        for ax in (top, bottom):
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.grid(axis="y", color="#34415a", alpha=0.35)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#34415a")
        return top, bottom

    def plot_historical(
        self, actual: np.ndarray, ai: np.ndarray, previous: np.ndarray,
        choices: dict[str, int], results: dict[str, float], duration: int, date: str,
    ) -> None:
        a, b = self.prepare_axes(self.history_fig)
        hours = np.arange(START_HOUR, END_HOUR)
        a.plot(hours, actual, color=BLUE, marker="o", label="ERA5 actual (evaluation)")
        a.plot(hours, ai, color=ORANGE, marker=".", label="SunPilot AI")
        a.plot(hours, previous, color=MUTED, linestyle="--", label="Yesterday baseline")
        start = choices["SunPilot AI"]
        a.axvspan(start-0.5, start+duration-0.5, color=ORANGE, alpha=.17)
        a.set_xlim(7.5, 17.5)
        a.set_xticks(range(8, 18))
        a.set_ylabel("Radiation (W/m²)", color=MUTED)
        a.set_title(f"Historical prediction • {date}", color=TEXT, loc="left", fontsize=12)
        a.legend(loc="upper right", facecolor=SURFACE, edgecolor="#34415a", labelcolor=TEXT, fontsize=8)
        names = list(results)
        vals = list(results.values())
        b.barh(names, vals, color=[ORANGE, MUTED, "#64748b", BLUE], height=.6)
        b.invert_yaxis()
        b.set_xlabel("Observed irradiation (Wh/m² per task)", color=MUTED)
        b.set_xlim(0, max(vals + [1]) * 1.20)
        for index, value in enumerate(vals):
            b.text(value + max(vals + [1])*.012, index, f"{value:.1f}", color=TEXT, va="center", fontsize=9)
        self.history_canvas.draw_idle()

    def plot_live(
        self, ai: np.ndarray, official: np.ndarray, previous: np.ndarray,
        start: int, duration: int, target_day: str,
    ) -> None:
        a, b = self.prepare_axes(self.live_fig)
        hours = np.arange(START_HOUR, END_HOUR)
        a.plot(hours, ai, color=ORANGE, marker="o", label="SunPilot AI prediction")
        a.plot(hours, official, color=BLUE, marker="o", label="Open-Meteo forecast")
        a.plot(hours, previous, color=MUTED, linestyle="--", label="Previous-day proxy")
        a.axvspan(start-.5, start+duration-.5, color=ORANGE, alpha=.18)
        a.set_xlim(7.5, 17.5)
        a.set_xticks(range(8, 18))
        a.set_ylabel("Forecast radiation (W/m²)", color=MUTED)
        a.set_title(f"Today's available forecast • {target_day}", color=TEXT, loc="left", fontsize=12)
        a.legend(loc="upper right", facecolor=SURFACE, edgecolor="#34415a", labelcolor=TEXT, fontsize=8)
        b.barh(["SunPilot AI", "Open-Meteo"],
               [float(np.mean(ai)), float(np.mean(official))],
               color=[ORANGE, BLUE], height=.5)
        b.invert_yaxis()
        b.set_xlabel("Mean predicted radiation 08–18 (W/m²)", color=MUTED)
        b.set_xlim(0, max(float(np.mean(ai)), float(np.mean(official)), 1)*1.18)
        self.live_canvas.draw_idle()


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    try:
        window = SunPilotWindow()
    except Exception:
        QMessageBox.critical(None, "SunPilot AI startup error", traceback.format_exc())
        raise
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
