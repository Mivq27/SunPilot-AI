"""SunPilot AI Desktop v0.5 — location-aware, flexible-duration native app.

Historical research: ERA5 2022–2025, trained model saved in models/.
Practical live planner: Open-Meteo forecast, with opt-in experimental ML.

Run from the SunPilot-AI project folder:
        python sunpilot_desktop_v05.py

No Gradio, browser, localhost server or automatic task execution.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
# Required in the frozen .exe so joblib can restore the trained estimator.
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QDate, QThread, Qt, Signal, QSettings
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QInputDialog,
    QDateEdit,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Standalone pure-Python planning engine — also used by unit tests.
from src.live_planner import (
    Location, Forecast, geocode_city, location_from_ip, fetch_forecast,
    build_plan, minute_window, best_historical_start, historical_energy,
    format_hhmm, human_duration,
)

# Paths must work both from Python and the PyInstaller --onedir bundle.
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
DATA_FILE = ROOT / "data" / "klippan_hourly_2022_2025.csv"
MODEL_FILE = ROOT / "models" / "sunpilot_v1.joblib"
SCHEDULER_METRICS = ROOT / "results" / "scheduler_metrics.json"
PREDICTION_METRICS = ROOT / "results" / "metrics.json"

LOCAL_TZ = ZoneInfo("Europe/Stockholm")
DEFAULT_LOCATION = Location("Klippan, Sweden", 56.1356, 13.1309, "Europe/Stockholm")
FIRST_HOUR, LAST_HOUR = 8, 18  # Local time, 18:00 end-exclusive.

BG = "#0A101B"
SURFACE = "#141F30"
SURFACE_2 = "#1B2A40"
BORDER = "#304059"
TEXT = "#F0F5FF"
MUTED = "#A8B7CD"
SUBTLE = "#8294AF"
AMBER = "#FF9B45"
BLUE = "#68BDF2"
TEAL = "#62D8BD"
RED = "#FCA5A5"

EXPECTED_FEATURES = [
    "hour_sin", "hour_cos", "season_sin", "season_cos",
    "radiation_lag_24", "radiation_lag_48", "radiation_lag_168",
    "cloud_lag_24", "temperature_lag_24",
]


# ---------- ML and scheduling (the same feature definition as src/train.py) ----------

def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.sort_values("time_utc").reset_index(drop=True)
    local = df["time_utc"].dt.tz_convert("Europe/Stockholm")
    hour = local.dt.hour
    day = local.dt.dayofyear
    df["local_date"] = local.dt.strftime("%Y-%m-%d")
    df["local_hour"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["season_sin"] = np.sin(2 * np.pi * day / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * day / 365.25)
    for lag in (24, 48, 168):
        df[f"radiation_lag_{lag}"] = df["shortwave_radiation"].shift(lag)
    df["cloud_lag_24"] = df["cloud_cover"].shift(24)
    df["temperature_lag_24"] = df["temperature_2m"].shift(24)
    return df


def best_start(values: np.ndarray, duration: int, starts: list[int]) -> int:
    """Sum hourly radiation for each allowed slot, break ties near noon.

    Only use scheduled-window values; no same-day observations in ML features.
    """
    if not starts:
        raise ValueError("No available hours remain. Choose tomorrow or a shorter task.")
    values = np.asarray(values, dtype=float)
    if values.shape != (LAST_HOUR - FIRST_HOUR,) or not np.isfinite(values).all():
        raise ValueError("Incomplete or invalid hourly solar data.")
    if not 1 <= duration <= LAST_HOUR - FIRST_HOUR:
        raise ValueError("Task duration is outside the supported window.")
    scores = {s: float(values[s - FIRST_HOUR:s - FIRST_HOUR + duration].sum()) for s in starts}
    high = max(scores.values())
    tied = [s for s, score in scores.items() if np.isclose(score, high, atol=1e-8)]
    return min(tied, key=lambda s: (abs(s - 12), s))


def solar_sum(values: np.ndarray, start: int, duration: int) -> float:
    k = start - FIRST_HOUR
    return float(np.asarray(values, dtype=float)[k:k + duration].sum())


def clock_range(start: int, duration: int) -> str:
    return f"{start:02d}:00–{start + duration:02d}:00"


def live_day_arrays(frame: pd.DataFrame, target_day, model_features: list[str]):
    """Return 10 target hours, target-input features and prior-day proxy.

    Open-Meteo's earlier timestamps are *not* asserted to be ground truth.
    Values at t-24, t-48, and t-168 form a domain-shifted ML input.
    """
    indexed = frame.set_index("time_utc").sort_index()
    if not isinstance(indexed.index, pd.DatetimeIndex) or indexed.index.tz is None:
        raise ValueError("Forecast timestamps must be timezone-aware.")
    if indexed.index.has_duplicates:
        indexed = indexed[~indexed.index.duplicated(keep="last")]

    local = indexed.index.tz_convert("Europe/Stockholm")
    mask = np.array([t.date() == target_day and FIRST_HOUR <= t.hour < LAST_HOUR for t in local])
    chosen = indexed.loc[mask]
    hours = chosen.index.tz_convert("Europe/Stockholm").hour.tolist()
    if hours != list(range(FIRST_HOUR, LAST_HOUR)):
        raise ValueError("The forecast has incomplete daytime coverage for this date.")

    forecast_values = chosen["shortwave_radiation"].to_numpy(dtype=float)
    if not np.isfinite(forecast_values).all():
        raise ValueError("The forecast contains missing radiation values.")
    rows = []
    for t in chosen.index:
        local_t = t.tz_convert("Europe/Stockholm")
        l24, l48, l168 = (t - pd.Timedelta(hours=h) for h in (24, 48, 168))
        if any(l not in indexed.index for l in (l24, l48, l168)):
            raise ValueError("The weather service did not provide enough past hours for AI inputs.")
        doy, hr = local_t.dayofyear, local_t.hour
        rows.append({
            "hour_sin": np.sin(2 * np.pi * hr / 24),
            "hour_cos": np.cos(2 * np.pi * hr / 24),
            "season_sin": np.sin(2 * np.pi * doy / 365.25),
            "season_cos": np.cos(2 * np.pi * doy / 365.25),
            "radiation_lag_24": indexed.at[l24, "shortwave_radiation"],
            "radiation_lag_48": indexed.at[l48, "shortwave_radiation"],
            "radiation_lag_168": indexed.at[l168, "shortwave_radiation"],
            "cloud_lag_24": indexed.at[l24, "cloud_cover"],
            "temperature_lag_24": indexed.at[l24, "temperature_2m"],
        })
    features = pd.DataFrame(rows, columns=model_features)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("The weather service returned incomplete AI inputs.")
    return forecast_values, features, features["radiation_lag_24"].to_numpy(dtype=float)


class WeatherWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, location: Location):
        super().__init__()
        self.location = location

    def run(self):
        try:
            self.ready.emit(fetch_forecast(self.location))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class LocationWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, mode: str, query: str = ""):
        super().__init__()
        self.mode = mode
        self.query = query

    def run(self):
        try:
            result = location_from_ip() if self.mode == "ip" else geocode_city(self.query)
            self.ready.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ---------- Reusable Qt UI components ----------

def label(text: str = "", role: str | None = None, rich: bool = False) -> QLabel:
    w = QLabel(text)
    w.setWordWrap(True)
    if role:
        w.setObjectName(role)
    w.setTextFormat(Qt.TextFormat.RichText if rich else Qt.TextFormat.PlainText)
    return w


class SunGlyph(QWidget):
    """Small scalable vector mark. No external icons or image resources."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(40, 40)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(AMBER), 2.5))
        from math import cos, sin, pi
        cx, cy = 20, 20
        p.drawEllipse(13, 13, 14, 14)
        for i in range(8):
            a = 2 * pi * i / 8
            p.drawLine(int(cx + 12 * cos(a)), int(cy + 12 * sin(a)),
                       int(cx + 17 * cos(a)), int(cy + 17 * sin(a)))
        p.end()


class Card(QFrame):
    def __init__(self, margins=(22, 20, 22, 20), spacing=12):
        super().__init__()
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(spacing)


class Field(QWidget):
    def __init__(self, title: str, control: QWidget, description: str = ""):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(7)
        body.addWidget(label(title, "fieldLabel"))
        control.setMinimumHeight(43)
        control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        body.addWidget(control)
        if description:
            control.setToolTip(description)


def available_page_width(widget: QWidget) -> int:
    """Use the viewport, not a stale layout minimum, for breakpoints."""
    p = widget.parentWidget()
    while p is not None:
        if isinstance(p, QScrollArea):
            return max(0, p.viewport().width() - 55)
        p = p.parentWidget()
    return widget.width()


class ResponsiveFields(Card):
    """Four columns on wide screens, two on medium, one on narrow."""

    def __init__(self, fields: list[Field], action: QPushButton):
        super().__init__(margins=(22, 20, 22, 20), spacing=16)
        self.fields = fields
        self.action = action
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(15)
        self.grid.setVerticalSpacing(16)
        self.body.addLayout(self.grid)
        self._columns = 0
        self._reflow(2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = available_page_width(self)
        cols = min(len(self.fields), 4 if width >= 1090 else 2 if width >= 590 else 1)
        if cols != self._columns:
            self._reflow(cols)

    def _reflow(self, cols: int):
        self._columns = cols
        for field in self.fields:
            self.grid.removeWidget(field)
        self.grid.removeWidget(self.action)
        for i in range(self.grid.columnCount()):
            self.grid.setColumnStretch(i, 0)
        for i, field in enumerate(self.fields):
            self.grid.addWidget(field, i // cols, i % cols)
        for j in range(cols):
            self.grid.setColumnStretch(j, 1)
        rows = (len(self.fields) + cols - 1) // cols
        self.action.setMinimumHeight(46)
        self.grid.addWidget(self.action, rows, 0, 1, cols)


class MetricCard(Card):
    def __init__(self, heading: str, value: str, detail: str, accent: str = AMBER):
        super().__init__(margins=(18, 16, 18, 16), spacing=8)
        self.value = label(value, "metricValue")
        self.value.setStyleSheet(f"color:{accent};")
        self.body.addWidget(label(heading, "eyebrow"))
        self.body.addWidget(self.value)
        self.body.addWidget(label(detail, "muted"))


class ResponsiveMetrics(QWidget):
    def __init__(self, cards: list[MetricCard]):
        super().__init__()
        self.cards = cards
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(14)
        self._cols = 0
        self._reflow(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = available_page_width(self)
        cols = 3 if width >= 950 else 2 if width >= 610 else 1
        if cols != self._cols:
            self._reflow(cols)

    def _reflow(self, cols: int):
        self._cols = cols
        for c in self.cards:
            self.grid.removeWidget(c)
        for i in range(self.grid.columnCount()):
            self.grid.setColumnStretch(i, 0)
        for i, c in enumerate(self.cards):
            self.grid.addWidget(c, i // cols, i % cols)
        for i in range(cols):
            self.grid.setColumnStretch(i, 1)


def data_table() -> QTableWidget:
    """Stretch columns instead of resizeColumnsToContents (which broke v0.3)."""
    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(["Method", "Time (local)", "Irradiation · Wh/m²"])
    table.horizontalHeader().setToolTip("Wh/m² = estimated sunlight incident on one square meter during the task")
    table.setAlternatingRowColors(True)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.verticalHeader().setVisible(False)
    table.setShowGrid(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setWordWrap(False)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    head = table.horizontalHeader()
    head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    head.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
    head.setMinimumSectionSize(95)
    head.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    table.setMinimumHeight(180)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return table


def fill_table(table: QTableWidget, rows: list[tuple[str, str, str]], emphasize: int = 0):
    table.setRowCount(len(rows))
    for i, row in enumerate(rows):
        table.setRowHeight(i, 42)
        for j, value in enumerate(row):
            item = QTableWidgetItem(value)
            item.setToolTip(value)
            if i == emphasize:
                item.setForeground(QColor(AMBER if j == 0 else TEXT))
                f = QFont()
                f.setBold(True)
                item.setFont(f)
            if j == 2:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(i, j, item)
    table.setMinimumHeight(42 * len(rows) + 41)


def make_chart(figure: Figure, canvas: FigureCanvasQTAgg, forecast: np.ndarray,
               ai: np.ndarray, yesterday: np.ndarray, chosen: int, duration: int,
               historical: bool):
    figure.clear()
    axis = figure.add_subplot(111)
    axis.set_facecolor(SURFACE)
    xs = np.arange(FIRST_HOUR, LAST_HOUR)
    axis.plot(xs, forecast, "o-", lw=2.6, ms=5, color=BLUE,
              label="ERA5 reference" if historical else "Weather forecast")
    axis.plot(xs, ai, "o-", lw=2.1, ms=4, color=AMBER, label="SunPilot AI")
    axis.plot(xs, yesterday, "--", lw=1.65, color=MUTED,
              label="Yesterday baseline" if historical else "Yesterday proxy")
    axis.axvspan(chosen - .5, chosen + duration - .5, color=TEAL, alpha=.14,
                 label="Selected window")
    axis.set_xlim(7.5, 17.6)
    axis.set_xticks(xs)
    axis.set_xticklabels([f"{x:02d}" for x in xs])
    axis.set_xlabel("Local time · Europe/Stockholm", color=MUTED, labelpad=9)
    axis.set_ylabel("Solar radiation · W/m²", color=MUTED, labelpad=7)
    axis.grid(axis="y", color=BORDER, alpha=.6)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(BORDER)
    legend = axis.legend(loc="upper right", facecolor=SURFACE_2,
                         edgecolor=BORDER, fontsize=8, ncol=2)
    for t in legend.get_texts():
        t.set_color(TEXT)
    figure.subplots_adjust(left=.095, right=.977, bottom=.20, top=.92)
    canvas.draw_idle()


def blank_chart(figure: Figure, canvas: FigureCanvasQTAgg, message: str):
    figure.clear()
    ax = figure.add_subplot(111)
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.5, .5, message, color=MUTED, ha="center", va="center",
            fontsize=12, transform=ax.transAxes, wrap=True)
    figure.tight_layout()
    canvas.draw_idle()


class AdaptiveScrollArea(QScrollArea):
    """Force responsive controls to follow viewport changes, even with old size hints."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        child = self.widget()
        if child is None:
            return
        available = max(0, self.viewport().width() - 55)
        for grid in child.findChildren(ResponsiveFields):
            cols = min(len(grid.fields), 4 if available >= 1090 else 2 if available >= 590 else 1)
            if grid._columns != cols:
                grid._reflow(cols)
        for grid in child.findChildren(ResponsiveMetrics):
            cols = 3 if available >= 950 else 2 if available >= 610 else 1
            if grid._cols != cols:
                grid._reflow(cols)


def make_scroll_page() -> tuple[QScrollArea, QVBoxLayout]:
    scroller = AdaptiveScrollArea()
    scroller.setObjectName("pageScroll")
    scroller.setWidgetResizable(True)
    scroller.setFrameShape(QFrame.Shape.NoFrame)
    scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    page = QWidget()
    page.setObjectName("page")
    page.setMinimumWidth(0)
    page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(6, 18, 13, 24)
    layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
    layout.setSpacing(17)
    scroller.setWidget(page)
    return scroller, layout


def page_heading(title: str, subtitle: str) -> QWidget:
    box = QWidget()
    body = QVBoxLayout(box)
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(5)
    body.addWidget(label(title, "pageTitle"))
    body.addWidget(label(subtitle, "muted"))
    return box


def chart_card(heading: str, caption: str):
    card = Card()
    card.body.addWidget(label(heading, "cardTitle"))
    card.body.addWidget(label(caption, "muted"))
    fig = Figure(figsize=(10, 3.45), dpi=100, facecolor=SURFACE)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setMinimumHeight(310)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    card.body.addWidget(canvas, 1)
    return card, fig, canvas


# ---------- Main window ----------

class SunPilotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SunPilot AI — Solar-aware task planning")
        self.resize(1280, 860)
        self.setMinimumSize(760, 530)

        if not MODEL_FILE.is_file() or not DATA_FILE.is_file():
            raise FileNotFoundError(
                "SunPilot is missing required assets.\n\n"
                f"Model: {MODEL_FILE}\n"
                f"Historical dataset: {DATA_FILE}\n\n"
                "Place this file in the project folder, or rebuild the .exe with --add-data."
            )
        package = joblib.load(MODEL_FILE)
        self.model = package["model"]
        self.features = list(package["features"])
        if self.features != EXPECTED_FEATURES:
            raise ValueError("The installed model has a different feature schema.")
        raw = pd.read_csv(DATA_FILE)
        history = feature_frame(raw).dropna(subset=self.features + ["shortwave_radiation"])
        history = history[history["local_date"].str[:4].isin(["2024", "2025"])].copy()
        if history.empty:
            raise ValueError("No 2024–2025 historical data was found.")
        history["ai_prediction"] = np.maximum(self.model.predict(history[self.features]), 0)
        self.history = history
        self.scheduler_metrics = self._read_json(SCHEDULER_METRICS)
        self.prediction_metrics = self._read_json(PREDICTION_METRICS)
        self.weather_worker: WeatherWorker | None = None
        self.copy_text = ""
        self.pending_plan = None  # Immutable inputs during a network request.
        self.location_worker: LocationWorker | None = None
        self.settings = QSettings("SunPilot", "SunPilot AI")
        try:
            self.location = Location(
                self.settings.value("location/name", DEFAULT_LOCATION.name),
                float(self.settings.value("location/latitude", DEFAULT_LOCATION.latitude)),
                float(self.settings.value("location/longitude", DEFAULT_LOCATION.longitude)),
                self.settings.value("location/timezone", DEFAULT_LOCATION.timezone),
            )
        except (ValueError, TypeError, KeyError):
            self.location = DEFAULT_LOCATION

        self._set_theme()
        self._build_window()
        self.analyze_history()
        self.statusBar().showMessage("Ready  ·  No tasks will run automatically")

    @staticmethod
    def _read_json(path: Path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def _set_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget#root, QWidget#page, QScrollArea#pageScroll {{
                background: {BG}; color: {TEXT};
                font-family: 'Segoe UI'; font-size: 13px;
            }}
            QWidget {{ color: {TEXT}; font-family: 'Segoe UI'; font-size: 13px; }}
            QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 13px; }}
            QLabel#brand {{ font-size: 27px; font-weight: 800; color: {TEXT}; }}
            QLabel#pageTitle {{ font-size: 23px; font-weight: 750; color: {TEXT}; }}
            QLabel#cardTitle {{ font-size: 16px; font-weight: 700; color: {TEXT}; }}
            QLabel#fieldLabel {{ font-size: 12px; font-weight: 650; color: {MUTED}; }}
            QLabel#eyebrow {{ font-size: 11px; font-weight: 700; color: {MUTED}; }}
            QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
            QLabel#bigTime {{ color: {TEAL}; font-size: 37px; font-weight: 800; }}
            QLabel#metricValue {{ font-size: 25px; font-weight: 750; }}
            QLabel#resultLead {{ font-size: 15px; font-weight: 650; }}
            QLabel#notice {{ background:{SURFACE_2}; color:{MUTED}; border-radius:9px;
                             padding:12px 15px; font-size:12px; }}
            QComboBox, QDateEdit, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {SURFACE_2}; color: {TEXT};
                border: 1px solid #40526D; border-radius: 8px;
                padding: 8px 12px; selection-background-color: {AMBER}; }}
            QComboBox:hover, QDateEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {AMBER}; }}
            QComboBox QAbstractItemView {{ background: {SURFACE_2}; color: {TEXT};
                selection-background-color: #39516B; border:1px solid {BORDER}; }}
            QPushButton {{ color:{TEXT}; background:{SURFACE_2};
                border:1px solid #40526D; border-radius:8px; padding:9px 16px;
                font-size:13px; font-weight:650; }}
            QPushButton:hover {{ background:#263951; border-color: {AMBER}; }}
            QPushButton#primary {{ background:{AMBER}; color:{BG};
                border: 1px solid {AMBER}; font-weight:800; }}
            QPushButton#primary:hover {{ background:#FFB06B; border-color:#FFB06B; }}
            QPushButton:disabled {{ color:#718199; background:#233043; border-color:#354359; }}
            QTabWidget::pane {{ border:0; background:{BG}; }}
            QTabBar::tab {{ color:{MUTED}; background:{SURFACE};
                border:1px solid {BORDER}; border-bottom:2px solid transparent;
                padding: 13px 22px; margin-right:5px; }}
            QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {AMBER}; }}
            QTabBar::tab:hover {{ color:{TEXT}; background:{SURFACE_2}; }}
            QTableWidget {{ background:{SURFACE}; alternate-background-color:#1A293D;
                color:{TEXT}; border:1px solid {BORDER}; border-radius:8px;
                outline:none; padding:2px; }}
            QHeaderView::section {{ background:#24344C; color:{MUTED};
                border:0; padding:10px 8px; font-size:12px; font-weight:700; }}
            QTableCornerButton::section {{ background:#24344C; border:0; }}
            QScrollArea {{ border:0; }}
            QScrollBar:vertical {{ background:{BG}; width:10px; margin:0; }}
            QScrollBar::handle:vertical {{ background:#33445D; min-height:30px; border-radius:5px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QStatusBar {{ background:{BG}; color:{MUTED}; font-size:11px; border-top:1px solid {BORDER}; }}
            QToolTip {{ background:{SURFACE_2}; color:{TEXT}; border:1px solid {BORDER}; padding:6px; }}
        """)

    def _build_window(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(23, 18, 23, 11)
        outer.setSpacing(17)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(SunGlyph())
        branding = QVBoxLayout()
        branding.setContentsMargins(0, 0, 0, 0)
        branding.setSpacing(1)
        brand = label("SUNPILOT  AI", "brand")
        brand.setStyleSheet(f"color:{TEXT};")
        branding.addWidget(brand)
        branding.addWidget(label("SOLAR-AWARE COMPUTE SCHEDULING", "muted"))
        top.addLayout(branding)
        top.addStretch(1)
        self.header_location = label(self.location.name.upper(), "eyebrow")
        location = self.header_location
        location.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(location)
        outer.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_live(), "Plan a task")
        self.tabs.addTab(self._build_history(), "Historical simulator")
        self.tabs.addTab(self._build_research(), "Research && results")
        self.tabs.setTabToolTip(0, "Schedule a future computing task using today's weather forecast")
        self.tabs.setTabToolTip(1, "Explore past AI predictions and compare scheduling choices")
        self.tabs.setTabToolTip(2, "See research methods, measured results and limitations")
        self.statusBar().setSizeGripEnabled(False)

    def _build_live(self) -> QScrollArea:
        scroll, page = make_scroll_page()
        page.addWidget(page_heading(
            "Find the best time for your task",
            "Plan a flexible computing job around the solar forecast. No technical knowledge needed.",
        ))

        # This card deliberately never sends an IP lookup unless the user clicks.
        loc_card = Card(spacing=12)
        top = QHBoxLayout()
        top.addWidget(label("LOCATION", "eyebrow"))
        top.addStretch(1)
        self.active_location_label = label(self.location.name, "resultLead")
        top.addWidget(self.active_location_label)
        loc_card.body.addLayout(top)
        loc_card.body.addWidget(label(
            "Choose your city or coordinates. 'Approximate my location' uses a public-IP lookup "
            "only if you click it; this is NOT precise device GPS.", "muted"))
        self.city_edit = QLineEdit()
        self.city_edit.setPlaceholderText("Search a city (e.g. Paris, France)")
        self.city_edit.returnPressed.connect(self.search_city)
        self.city_button = QPushButton("Find city")
        self.city_button.clicked.connect(self.search_city)
        self.ip_button = QPushButton("Approximate my location")
        self.ip_button.setToolTip("Sends your public IP to ipwho.is on your explicit request. Not GPS.")
        self.ip_button.clicked.connect(self.locate_ip)
        loc_grid = QGridLayout()
        loc_grid.setContentsMargins(0, 0, 0, 0)
        loc_grid.setHorizontalSpacing(10)
        loc_grid.setVerticalSpacing(10)
        loc_grid.addWidget(self.city_edit, 0, 0, 1, 3)
        loc_grid.addWidget(self.city_button, 1, 0, 1, 1)
        loc_grid.addWidget(self.ip_button, 1, 1, 1, 2)
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(5)
        self.lat_input.setSingleStep(.01)
        self.lat_input.setValue(self.location.latitude)
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180, 180)
        self.lon_input.setDecimals(5)
        self.lon_input.setSingleStep(.01)
        self.lon_input.setValue(self.location.longitude)
        self.coords_button = QPushButton("Use coordinates")
        self.coords_button.clicked.connect(self.use_coordinates)
        lat_field = Field("LATITUDE", self.lat_input)
        lon_field = Field("LONGITUDE", self.lon_input)
        loc_grid.addWidget(lat_field, 2, 0)
        loc_grid.addWidget(lon_field, 2, 1)
        loc_grid.addWidget(self.coords_button, 2, 2)
        loc_grid.setColumnStretch(0, 1)
        loc_grid.setColumnStretch(1, 1)
        loc_grid.setColumnStretch(2, 1)
        loc_card.body.addLayout(loc_grid)
        page.addWidget(loc_card)

        self.task_kind = QComboBox()
        self.task_kind.addItems(["3D rendering / Blender", "Python script", "Machine learning", "Data processing", "Other workload"])
        self.task_kind.setToolTip("This names your task; it does not change the radiation model.")
        self.target_day = QComboBox()
        self.target_day.addItems(["Today", "Tomorrow", "Next 7 days", "Up to 16 days"])
        # Today/tomorrow are in the selected location's timezone.
        # 'Today' can also schedule an overnight workload.
        self.hours = QSpinBox()
        self.hours.setRange(0, 100000)
        self.hours.setValue(2)
        self.hours.setSuffix(" h")
        self.hours.setToolTip("No arbitrary time limit. Real limit: available forecast horizon.")
        self.minutes = QSpinBox()
        self.minutes.setRange(0, 59)
        self.minutes.setValue(0)
        self.minutes.setSuffix(" min")
        self.minutes.setToolTip("Duration is accurate to one minute; forecasts are hourly estimates.")
        duration_control = QWidget()
        duration_layout = QHBoxLayout(duration_control)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.setSpacing(8)
        duration_layout.addWidget(self.hours, 1)
        duration_layout.addWidget(self.minutes, 1)
        self.method = QComboBox()
        self.method.addItems(["Weather forecast · recommended", "SunPilot AI · experimental"])
        self.method.setToolTip("The second method uses the trained ML model with forecast-derived inputs. It is not yet live-validated.")
        self.plan_button = QPushButton("Find best time")
        self.plan_button.setObjectName("primary")
        self.plan_button.clicked.connect(self.get_live)
        self.plan_controls = ResponsiveFields([
            Field("TASK", self.task_kind),
            Field("WHEN", self.target_day),
            Field("DURATION · HOURS + MINUTES", duration_control),
            Field("SCHEDULING METHOD", self.method),
        ], self.plan_button)
        page.addWidget(self.plan_controls)

        result = Card(spacing=10)
        heading = QHBoxLayout()
        heading.addWidget(label("YOUR RECOMMENDATION", "eyebrow"))
        heading.addStretch(1)
        self.copy_btn = QPushButton("Copy schedule")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self.copy_recommendation)
        heading.addWidget(self.copy_btn)
        result.body.addLayout(heading)
        self.live_time = label("Ready to plan", "bigTime")
        self.live_time.setStyleSheet(f"color:{MUTED};font-size:30px;font-weight:750;")
        result.body.addWidget(self.live_time)
        self.live_lead = label("Choose your task and click Find best time.", "resultLead")
        result.body.addWidget(self.live_lead)
        self.live_detail = label("We will download a current weather forecast and compare it with SunPilot's experimental AI.", "muted")
        result.body.addWidget(self.live_detail)
        page.addWidget(result)

        self.live_chart_card, self.live_fig, self.live_canvas = chart_card(
            "Solar outlook", "Hourly forecast; minute-level scheduling approximates each hour as constant")
        page.addWidget(self.live_chart_card)
        blank_chart(self.live_fig, self.live_canvas, "Your solar forecast will appear here after planning a task")

        compare = Card(spacing=13)
        compare.body.addWidget(label("How the methods compare", "cardTitle"))
        compare.body.addWidget(label("These are forecast-based estimates, not measured energy production.", "muted"))
        self.live_table = data_table()
        compare.body.addWidget(self.live_table)
        page.addWidget(compare)

        page.addWidget(label(
            "EXPERIMENTAL · Forecasts are estimates, not solar-panel electricity or verified CO₂ savings. "
            "The AI was trained on Klippan/ERA5; it has NOT been live-validated or validated in other locations. "
            "Minute precision uses hourly interpolation, NOT minute-by-minute weather. "
            "The planner suggests times but never runs tasks automatically.", "notice"))
        page.addStretch(1)
        return scroll

    def _build_history(self) -> QScrollArea:
        scroll, page = make_scroll_page()
        page.addWidget(page_heading(
            "Explore the historical simulator",
            "Choose a day from 2024–2025 to see what the trained AI would have decided.",
        ))
        self.hist_date = QDateEdit(QDate(2025, 8, 26))
        self.hist_date.setDisplayFormat("yyyy-MM-dd")
        self.hist_date.setCalendarPopup(True)
        self.hist_date.setDateRange(QDate(2024, 1, 1), QDate(2025, 12, 31))
        self.hist_hours = QSpinBox()
        self.hist_hours.setRange(0, 100000)
        self.hist_hours.setValue(2)
        self.hist_hours.setSuffix(" h")
        self.hist_minutes = QSpinBox()
        self.hist_minutes.setRange(0, 59)
        self.hist_minutes.setSuffix(" min")
        hist_duration_control = QWidget()
        hist_duration_layout = QHBoxLayout(hist_duration_control)
        hist_duration_layout.setContentsMargins(0, 0, 0, 0)
        hist_duration_layout.setSpacing(8)
        hist_duration_layout.addWidget(self.hist_hours, 1)
        hist_duration_layout.addWidget(self.hist_minutes, 1)
        self.hist_button = QPushButton("Run simulation")
        self.hist_button.setObjectName("primary")
        self.hist_button.clicked.connect(self.analyze_history)
        page.addWidget(ResponsiveFields([
            Field("HISTORICAL DATE", self.hist_date),
            Field("DURATION · HOURS + MINUTES", hist_duration_control),
        ], self.hist_button))

        result = Card(spacing=9)
        result.body.addWidget(label("WHAT SUNPILOT WOULD HAVE CHOSEN", "eyebrow"))
        self.hist_time = label("—", "bigTime")
        result.body.addWidget(self.hist_time)
        self.hist_lead = label("", "resultLead")
        result.body.addWidget(self.hist_lead)
        self.hist_detail = label("", "muted")
        result.body.addWidget(self.hist_detail)
        page.addWidget(result)

        chart, self.hist_fig, self.hist_canvas = chart_card(
            "Predictions versus historical reference", "Blue: ERA5 reanalysis · amber: SunPilot ML · dashed: previous-day baseline")
        page.addWidget(chart)
        comparison = Card()
        comparison.body.addWidget(label("What the historical day actually delivered", "cardTitle"))
        comparison.body.addWidget(label("Each row is scored using the same ERA5 reference values; perfect knowledge is unattainable.", "muted"))
        self.hist_table = data_table()
        comparison.body.addWidget(self.hist_table)
        page.addWidget(comparison)
        page.addWidget(label(
            "BACKTEST · 2022–2023 training, 2024 validation, 2025 held-out test. "
            "ML features use prior-day or earlier inputs. Reference values come from ERA5 reanalysis, "
            "not a physical solar panel. Changing the task duration in this view does not change "
            "the saved two-hour aggregate benchmark.", "notice"))
        page.addStretch(1)
        return scroll

    def _build_research(self) -> QScrollArea:
        scroll, page = make_scroll_page()
        page.addWidget(page_heading(
            "Research & results", "What we tested, what worked, and where this prototype still needs evidence.",
        ))
        test = (self.scheduler_metrics or {}).get("test_2025", {})
        forecast = (self.prediction_metrics or {}).get("test_2025", {})
        ai_mae = forecast.get("ai", {}).get("mae_all_w_m2")
        baseline_mae = forecast.get("baseline", {}).get("mae_all_w_m2")
        improvement = 100 * (1 - ai_mae / baseline_mae) if ai_mae is not None and baseline_mae else None
        avg = test.get("mean_wh_m2", {})
        ai_delta = test.get("ai_gain_vs_fixed_percent")
        metric_cards = [
            MetricCard("FORECAST ERROR · 2025", f"{improvement:+.2f}%" if improvement is not None else "—",
                       "Lower MAE than yesterday's values" if improvement is not None else "Run training to generate metrics", TEAL),
            MetricCard("VS FIXED NOON · 2025", f"{ai_delta:+.2f}%" if ai_delta is not None else "—",
                       "Solar irradiation for 2-hour tasks", AMBER),
            MetricCard("DAYS EVALUATED", str(test.get("days", "—")),
                       "Held-out 2025 scheduling test", BLUE),
        ]
        page.addWidget(ResponsiveMetrics(metric_cards))
        question = Card()
        question.body.addWidget(label("The question", "cardTitle"))
        question.body.addWidget(label(
            "Can a machine-learning radiation forecast choose better hours for flexible computing "
            "workloads than simple scheduling rules?"))
        page.addWidget(question)

        method = Card()
        method.body.addWidget(label("How we tested it", "cardTitle"))
        method.body.addWidget(label(
            "35,064 hourly records for the Klippan area (2022–2025). Model: HistGradientBoostingRegressor. "
            "Training: 2022–2023 · Validation: 2024 · Test: 2025. "
            "Features include local hour, season, and radiation/weather lags at 24, 48 and 168 hours."))
        page.addWidget(method)

        numbers = Card()
        numbers.body.addWidget(label("Scheduling benchmark · 2025, two-hour tasks", "cardTitle"))
        if avg:
            rows = [
                ("SunPilot AI", f"{avg.get('SunPilot AI', float('nan')):.1f} Wh/m²"),
                ("Fixed noon", f"{avg.get('Fixed 12:00', float('nan')):.1f} Wh/m²"),
                ("Yesterday baseline", f"{avg.get('Yesterday', float('nan')):.1f} Wh/m²"),
                ("Perfect knowledge (upper bound)", f"{avg.get('Perfect knowledge', float('nan')):.1f} Wh/m²"),
            ]
            for name, value in rows:
                row = QHBoxLayout()
                row.addWidget(label(name, "muted"), 2)
                value_label = label(value, "resultLead")
                value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                row.addWidget(value_label, 1)
                numbers.body.addLayout(row)
        else:
            numbers.body.addWidget(label("Saved benchmark data not found. Run src/scheduler.py to generate it."))
        page.addWidget(numbers)

        finding = Card()
        finding.body.addWidget(label("The honest conclusion", "cardTitle"))
        finding.body.addWidget(label(
            "In the historical held-out test, the AI reduced prediction error relative to a "
            "previous-day baseline. It beat yesterday-based scheduling but did not beat a fixed "
            "noon schedule. Better predictions do not necessarily lead to better decisions."))
        finding.body.addWidget(label(
            "The live planner is a separate experimental deployment: its inputs may be forecast-model "
            "estimates rather than the ERA5 data used in training. It has not yet been validated "
            "against archived real-time forecasts.", "muted"))
        page.addWidget(finding)

        sources = Card()
        sources.body.addWidget(label("Data, attribution & limitations", "cardTitle"))
        sources.body.addWidget(label(
            "Weather: Open-Meteo historical ERA5 reanalysis and forecast API. "
            "AI development assistance: ChatGPT. "
            "No measured PV power, financial savings, or verified carbon reduction. "
            "Historical research location: Klippan, Sweden. Live planning supports user-chosen locations. "
            "The historical one-day experiment uses the 08:00–18:00 evaluation window. No automatic execution. "
            "Source code and reproducibility details: github.com/Mivq27/SunPilot-AI."))
        page.addWidget(sources)
        page.addStretch(1)
        return scroll

    def _apply_location(self, location: Location):
        self.location = location
        self.active_location_label.setText(location.name)
        self.header_location.setText(location.name.upper())
        self.lat_input.setValue(location.latitude)
        self.lon_input.setValue(location.longitude)
        self.settings.setValue("location/name", location.name)
        self.settings.setValue("location/latitude", location.latitude)
        self.settings.setValue("location/longitude", location.longitude)
        self.settings.setValue("location/timezone", location.timezone or "")
        self.copy_btn.setEnabled(False)
        self.copy_text = ""
        self.live_time.setText("Ready to plan")
        self.live_lead.setText("Location updated. Click Find best time for a new forecast.")
        self.live_detail.setText(f"Selected area: {location.name}. Your location is saved locally on this PC.")
        blank_chart(self.live_fig, self.live_canvas, "Choose Find best time to refresh this location's forecast")
        self.live_table.setRowCount(0)
        self.statusBar().showMessage(f"Location set: {location.name}")

    def use_coordinates(self):
        lat = self.lat_input.value()
        lon = self.lon_input.value()
        self._apply_location(Location(f"{lat:.4f}, {lon:.4f}", lat, lon))

    def _start_location_lookup(self, mode: str, query: str = ""):
        if self.location_worker is not None and self.location_worker.isRunning():
            return
        if self.weather_worker is not None and self.weather_worker.isRunning():
            return
        self.location_worker = LocationWorker(mode, query)
        self.location_worker.ready.connect(self._location_ready)
        self.location_worker.failed.connect(self._location_failed)
        self.location_worker.finished.connect(self._location_finished)
        self.city_button.setEnabled(False)
        self.ip_button.setEnabled(False)
        self.coords_button.setEnabled(False)
        self.statusBar().showMessage("Looking up location…")
        self.location_worker.start()

    def search_city(self):
        query = self.city_edit.text().strip()
        if len(query) < 2:
            QMessageBox.information(self, "Search a location", "Enter a city name, e.g. Madrid, Spain.")
            return
        self._start_location_lookup("city", query)

    def locate_ip(self):
        # This lookup is opt-in, city-level at best, and not Windows GPS.
        self._start_location_lookup("ip")

    def _location_ready(self, result):
        if isinstance(result, Location):
            self._apply_location(result)
            return
        if not result:
            QMessageBox.information(self, "Location not found", "Try another city name or enter coordinates.")
            return
        if len(result) == 1:
            self._apply_location(result[0])
            return
        choices = [f"{loc.name}  ({loc.latitude:.3f}, {loc.longitude:.3f})" for loc in result]
        choice, accepted = QInputDialog.getItem(self, "Choose your city", "Matching locations:", choices, 0, False)
        if accepted and choice in choices:
            self._apply_location(result[choices.index(choice)])

    def _location_failed(self, error: str):
        self.statusBar().showMessage("Location lookup failed; manual coordinates still work")
        QMessageBox.warning(self, "Location lookup", error)

    def _location_finished(self):
        self.city_button.setEnabled(True)
        self.ip_button.setEnabled(True)
        self.coords_button.setEnabled(True)
        self.location_worker = None

    def analyze_history(self):
        try:
            day = self.hist_date.date().toString("yyyy-MM-dd")
            duration = self.hist_hours.value() * 60 + self.hist_minutes.value()
            if not 1 <= duration <= 600:
                raise ValueError(
                    "Historical research uses one 08:00–18:00 day (maximum 10 hours). "
                    "For long tasks, use Plan a task instead."
                )
            slice_ = self.history[(self.history["local_date"] == day) &
                                  self.history["local_hour"].between(FIRST_HOUR, LAST_HOUR - 1)]
            slice_ = slice_.sort_values("local_hour")
            if slice_["local_hour"].tolist() != list(range(FIRST_HOUR, LAST_HOUR)):
                raise ValueError("This day has incomplete historical solar data.")
            actual = slice_["shortwave_radiation"].to_numpy(dtype=float)
            ai = slice_["ai_prediction"].to_numpy(dtype=float)
            previous = slice_["radiation_lag_24"].to_numpy(dtype=float)
            ai_start, _ = best_historical_start(ai, duration)
            previous_start, _ = best_historical_start(previous, duration)
            oracle_start, _ = best_historical_start(actual, duration)
            # For >6h noon cannot fit, so keep the fixed heuristic feasible.
            fixed_start = min(240, 600 - duration)
            positions = {
                "SunPilot AI": ai_start,
                "Yesterday baseline": previous_start,
                "Fixed near noon": fixed_start,
                "Perfect knowledge *": oracle_start,
            }
            values = {name: historical_energy(actual, start, duration) for name, start in positions.items()}
            chosen = positions["SunPilot AI"]
            self.hist_time.setText(format_hhmm(chosen, duration))
            self.hist_lead.setText(f"AI-selected time on {day} · {human_duration(duration)} · Klippan, Sweden")
            self.hist_detail.setText(
                f"Compared with fixed near noon: {values['SunPilot AI'] - values['Fixed near noon']:+.1f} Wh/m².  "
                f"Compared with yesterday: {values['SunPilot AI'] - values['Yesterday baseline']:+.1f} Wh/m². "
                "Positive means higher incident sunlight for this particular historical day."
            )
            fill_table(self.hist_table, [
                (name, format_hhmm(start, duration), f"{values[name]:.1f}")
                for name, start in positions.items()
            ])
            # Visualization accepts fractional-hour start and duration.
            make_chart(self.hist_fig, self.hist_canvas, actual, ai, previous,
                       FIRST_HOUR + chosen / 60.0, duration / 60.0, historical=True)
            self.statusBar().showMessage(f"Historical simulator · {day} · {human_duration(duration)}")
        except Exception as exc:
            QMessageBox.warning(self, "Historical simulation unavailable", str(exc))

    def get_live(self):
        if self.weather_worker is not None and self.weather_worker.isRunning():
            return
        if self.location_worker is not None and self.location_worker.isRunning():
            QMessageBox.information(self, "Location loading", "Complete the location lookup first.")
            return
        duration = self.hours.value() * 60 + self.minutes.value()
        if duration < 1:
            QMessageBox.warning(self, "Duration required", "Choose at least 1 minute.")
            return
        self.pending_plan = (self.task_kind.currentText(), self.target_day.currentIndex(),
                             duration, self.method.currentIndex(), self.location)
        for control in (self.task_kind, self.target_day, self.hours, self.minutes,
                        self.method, self.city_button, self.ip_button, self.coords_button,
                        self.lat_input, self.lon_input, self.city_edit):
            control.setEnabled(False)
        self.plan_button.setEnabled(False)
        self.plan_button.setText("Checking forecast…")
        self.copy_btn.setEnabled(False)
        self.copy_text = ""
        self.live_time.setText("Checking sunlight…")
        self.live_lead.setText(f"Downloading forecast for {self.location.name}")
        self.live_detail.setText("Hourly forecasts, scheduled in 15-minute start increments.")
        self.statusBar().showMessage("Connecting to Open-Meteo…")
        self.weather_worker = WeatherWorker(self.location)
        self.weather_worker.ready.connect(self.show_live)
        self.weather_worker.failed.connect(self.weather_error)
        self.weather_worker.finished.connect(self.weather_finished)
        self.weather_worker.start()

    def weather_finished(self):
        for control in (self.task_kind, self.target_day, self.hours, self.minutes,
                        self.method, self.city_button, self.ip_button, self.coords_button,
                        self.lat_input, self.lon_input, self.city_edit):
            control.setEnabled(True)
        self.plan_button.setEnabled(True)
        self.plan_button.setText("Find best time")
        self.pending_plan = None
        self.weather_worker = None

    def weather_error(self, message: str):
        self.live_time.setText("Plan unavailable")
        self.live_lead.setText("SunPilot could not calculate a reliable schedule.")
        self.live_detail.setText("Check the forecast connection or choose a different start range. "
                                 "Tasks longer than the forecast horizon cannot be evaluated.")
        self.copy_btn.setEnabled(False)
        self.copy_text = ""
        self.statusBar().showMessage("Planning error · Please retry")
        QMessageBox.warning(self, "SunPilot · Planning", message)

    def _plot_live(self, plan):
        from matplotlib import dates as mdates
        figure = self.live_fig
        figure.clear()
        ax = figure.add_subplot(111)
        ax.set_facecolor(SURFACE)
        local = plan.chart_utc.tz_convert(plan.timezone)
        ax.plot(local, plan.chart_provider, lw=2.5, color=BLUE, label="Weather forecast")
        ax.plot(local, plan.chart_ai, lw=1.9, color=AMBER, alpha=.9,
                label="SunPilot AI (experimental)")
        ax.axvspan(plan.start_utc.astimezone(ZoneInfo(plan.timezone)),
                   plan.end_utc.astimezone(ZoneInfo(plan.timezone)),
                   color=TEAL, alpha=.16, label="Selected task")
        ax.set_ylabel("Solar irradiance (W/m²)", color=MUTED)
        ax.set_xlabel(f"Local time · {plan.timezone}", color=MUTED)
        ax.grid(axis="y", color=BORDER, alpha=.6)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator(), tz=ZoneInfo(plan.timezone)))
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        legend = ax.legend(loc="upper right", facecolor=SURFACE_2, edgecolor=BORDER, fontsize=8)
        for t in legend.get_texts():
            t.set_color(TEXT)
        figure.subplots_adjust(left=.095, right=.975, bottom=.21, top=.92)
        self.live_canvas.draw_idle()

    def show_live(self, forecast: Forecast):
        try:
            task, when, duration, method_index, location = self.pending_plan
            plan = build_plan(forecast, self.model, duration, when, method_index)
            tz = ZoneInfo(plan.timezone)
            start_local = plan.start_utc.astimezone(tz)
            end_local = plan.end_utc.astimezone(tz)
            display_time = f"{start_local:%a %d %b, %H:%M} – {end_local:%a %d %b, %H:%M}"
            self.live_time.setStyleSheet(f"color:{AMBER if method_index == 1 else TEAL};font-size:28px;font-weight:800;")
            self.live_time.setText(display_time)
            self.live_time.setWordWrap(True)
            self.live_lead.setText(f"{task} · {human_duration(duration)} · {location.name}")
            note = " · AI unvalidated in this region" if method_index == 1 else " · provider-recommended"
            self.live_detail.setText(
                f"{plan.method}{note}. Time zone: {plan.timezone}. "
                f"Estimated incident solar irradiation: {plan.estimated_wh_m2:.1f} Wh/m². "
                "It is not measured panel electricity. Minutes are estimated from hourly data."
            )
            self.copy_text = (
                f"SunPilot AI | {task} | {location.name} | "
                f"{start_local:%Y-%m-%d %H:%M} to {end_local:%Y-%m-%d %H:%M} "
                f"({plan.timezone}) | {human_duration(duration)} | {plan.method} | "
                "Weather forecast estimate only; not guaranteed PV production."
            )
            self.copy_btn.setEnabled(True)
            rows = []
            for method, (start_utc, estimate) in plan.comparison.items():
                local_start = start_utc.astimezone(tz)
                rows.append((method, local_start.strftime("%d %b %H:%M"), f"{estimate:.1f}"))
            fill_table(self.live_table, rows, emphasize=1 if method_index == 1 else 0)
            self._plot_live(plan)
            self.statusBar().showMessage(
                f"Forecast received · {location.name} · {plan.timezone} · {human_duration(duration)} · "
                "no automatic execution"
            )
        except Exception as exc:
            self.weather_error(f"{type(exc).__name__}: {exc}")

    def copy_recommendation(self):
        if self.copy_text:
            QApplication.clipboard().setText(self.copy_text)
            self.statusBar().showMessage("Schedule copied to clipboard")

    def closeEvent(self, event):
        # Avoid destroying a running QThread. The HTTP timeout bounds the wait.
        if (self.weather_worker is not None and self.weather_worker.isRunning()) or (self.location_worker is not None and self.location_worker.isRunning()):
            event.ignore()
            self.statusBar().showMessage("Finishing the network request before closing…")
            return
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        window = SunPilotApp()
    except Exception:
        QMessageBox.critical(None, "SunPilot AI could not start", traceback.format_exc())
        raise
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
