"""SunPilot AI Desktop v1.0 release candidate — native Windows desktop application.

Historical research: ERA5 2022–2025, trained model saved in models/.
Practical live planner: Open-Meteo forecast, with opt-in experimental ML.

Run from the SunPilot-AI project folder:
        python sunpilot_desktop_v054.py

No Gradio or browser. Windows programs are scheduled only after user confirmation.
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
from PySide6.QtCore import (
    QDate, QDateTime, QTimer, QThread, Qt, Signal, QSettings, QPropertyAnimation,
    QEasingCurve, QAbstractAnimation,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QDateTimeEdit,
    QFileDialog,
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
    QToolButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Standalone pure-Python planning engine — also used by unit tests.
from src.windows_scheduler import register_launch, list_launches, cancel_launch

from src.solar_time import historical_day

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
ICON_FILE = ROOT / "assets" / "sunpilot.ico"

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
    day = local.dt.day_of_year
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


# ---------- Wheel-safe controls and gentle page scrolling ----------

class NoWheelSpinBox(QSpinBox):
    """Scrolling a page must never alter a task duration by accident.

    Keyboard arrows, typing, and the spin-button arrows keep working normally.
    """

    def wheelEvent(self, event):
        event.ignore()  # Let the parent scroll area handle the wheel.


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDateEdit(QDateEdit):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDateTimeEdit(QDateTimeEdit):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        # Wheel selection is disabled unless the options list is actually open.
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()


class ScheduledLaunchWorker(QThread):
    """Windows PowerShell calls run off the GUI thread."""
    ready = Signal(str, object)
    failed = Signal(str)

    def __init__(self, operation: str, data=None):
        super().__init__()
        self.operation = operation
        self.data = data

    def run(self):
        try:
            if self.operation == "register":
                result = register_launch(self.data["exe"], self.data["when_utc"])
            elif self.operation == "list":
                result = list_launches()
            elif self.operation == "cancel":
                cancel_launch(self.data)
                result = self.data
            else:
                raise ValueError("Unknown schedule operation")
            self.ready.emit(self.operation, result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class NoWheelTable(QTableWidget):
    """A small comparison table should not steal the page's wheel scroll."""

    def wheelEvent(self, event):
        event.ignore()


class ScrollCanvas(FigureCanvasQTAgg):
    """Matplotlib stays interactive but doesn't capture ordinary page scroll."""

    def wheelEvent(self, event):
        event.ignore()


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
    """Responsive input fields with a compact right-aligned primary action."""

    def __init__(self, fields: list[Field], action: QPushButton):
        super().__init__(margins=(22, 20, 22, 20), spacing=15)
        self.fields = fields
        self.action = action
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(14)
        self.body.addLayout(self.grid)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 4, 0, 0)
        actions.addStretch(1)
        action.setMinimumHeight(46)
        action.setMinimumWidth(190)
        actions.addWidget(action)
        self.body.addLayout(actions)
        self._columns = 0
        self._reflow(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = available_page_width(self)
        cols = min(len(self.fields), 3 if width >= 920 else 2 if width >= 590 else 1)
        if cols != self._columns:
            self._reflow(cols)

    def _reflow(self, cols: int):
        self._columns = cols
        for field in self.fields:
            self.grid.removeWidget(field)
        for i in range(self.grid.columnCount()):
            self.grid.setColumnStretch(i, 0)
        for i, field in enumerate(self.fields):
            self.grid.addWidget(field, i // cols, i % cols)
        for j in range(cols):
            self.grid.setColumnStretch(j, 1)


class MethodPicker(QWidget):
    """Accessible, responsive and intentionally prominent planning-method choice.

    The simple currentIndex/currentIndexChanged API keeps the proven scheduling
    engine unchanged. Checkable buttons do not react to mouse-wheel scrolling.
    """
    currentIndexChanged = Signal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName("methodPicker")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons = []
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(10)
        options = [
            (
                "WEATHER FORECAST\nRecommended · Open-Meteo",
                "Uses the weather provider's upcoming solar radiation forecast. "
                "Recommended for real-world planning.",
            ),
            (
                "SUNPILOT AI\nExperimental · research model",
                "Uses our trained machine-learning model. It was trained on "
                "historical data from Klippan and is not live-validated.",
            ),
        ]
        for index, (caption, tooltip) in enumerate(options):
            button = QPushButton(caption)
            button.setObjectName("methodChoice")
            button.setCheckable(True)
            button.setAutoExclusive(False)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(78)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setToolTip(tooltip)
            button.setAccessibleName(caption.replace("\n", ". "))
            self._group.addButton(button, index)
            self._buttons.append(button)
        self._buttons[0].setChecked(True)
        self._group.idClicked.connect(self._selected)
        self._columns = 0
        self._reflow(1)

    def currentIndex(self) -> int:
        return self._group.checkedId()

    def setCurrentIndex(self, index: int) -> None:
        if index not in (0, 1):
            raise ValueError("Unknown planning method")
        if index != self.currentIndex():
            self._buttons[index].setChecked(True)
            self.currentIndexChanged.emit(index)

    def _selected(self, index: int):
        self.currentIndexChanged.emit(index)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cols = 2 if self.width() >= 635 else 1
        if cols != self._columns:
            self._reflow(cols)

    def _reflow(self, cols: int):
        self._columns = cols
        for button in self._buttons:
            self._grid.removeWidget(button)
        for col in range(max(self._grid.columnCount(), 2)):
            self._grid.setColumnStretch(col, 0)
        for i, button in enumerate(self._buttons):
            self._grid.addWidget(button, i // cols, i % cols)
        for col in range(cols):
            self._grid.setColumnStretch(col, 1)


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
    table = NoWheelTable(0, 3)
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
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
    table.setFixedHeight(42 * len(rows) + 50)


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
    """Wheel scroll keeps its place, and controls never hijack it.

    Precision touchpads use native pixel deltas. Mouse wheels get a short,
    interruptible ease-out animation, with the next wheel tick accumulated.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scroll_target = 0
        self._scroll_animation = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._scroll_animation.setDuration(155)
        self._scroll_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.verticalScrollBar().valueChanged.connect(self._remember_position)

    def _remember_position(self, value):
        if self._scroll_animation.state() == QAbstractAnimation.State.Stopped:
            self._scroll_target = value

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.ignore()
            return
        if not event.pixelDelta().isNull():
            # Native high-resolution touchpad scrolling remains smooth.
            self._scroll_animation.stop()
            super().wheelEvent(event)
            self._scroll_target = self.verticalScrollBar().value()
            return
        degrees = event.angleDelta().y()
        if degrees == 0:
            super().wheelEvent(event)
            return
        bar = self.verticalScrollBar()
        if bar.maximum() <= bar.minimum():
            event.ignore()
            return
        if self._scroll_animation.state() == QAbstractAnimation.State.Stopped:
            self._scroll_target = bar.value()
        delta = round(-degrees * 0.65)  # One common wheel notch -> 78 px.
        self._scroll_target = max(bar.minimum(), min(bar.maximum(), self._scroll_target + delta))
        self._scroll_animation.stop()
        self._scroll_animation.setStartValue(bar.value())
        self._scroll_animation.setEndValue(self._scroll_target)
        self._scroll_animation.start()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        child = self.widget()
        if child is None:
            return
        available = max(0, self.viewport().width() - 40)
        for grid in child.findChildren(ResponsiveFields):
            cols = min(len(grid.fields), 3 if available >= 920 else 2 if available >= 590 else 1)
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
    scroller.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroller.verticalScrollBar().setSingleStep(24)
    page = QWidget()
    page.setObjectName("page")
    page.setMinimumWidth(0)
    page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(9, 18, 17, 28)
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
    canvas = ScrollCanvas(fig)
    canvas.setMinimumHeight(310)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    card.body.addWidget(canvas, 1)
    return card, fig, canvas


# ---------- Main window ----------

class SunPilotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SunPilot AI  ·  Solar-aware scheduling")
        if ICON_FILE.is_file():
            self.setWindowIcon(QIcon(str(ICON_FILE)))
        self.resize(1210, 850)
        self.setMinimumSize(590, 470)

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
        self.last_plan = None
        self.selected_exe = ""
        self.scheduler_worker = None
        self._scheduler_refresh_pending = False
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
        self.statusBar().showMessage("Ready  ·  Plan a task or explore the historical simulator")

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
            QFrame#card {{ background: {SURFACE}; border: 1px solid #293A51; border-radius: 12px; }}
            QLabel#brand {{ font-size: 27px; font-weight: 800; color: {TEXT}; }}
            QLabel#pageTitle {{ font-size: 25px; font-weight: 750; color: {TEXT}; }}
            QLabel#cardTitle {{ font-size: 17px; font-weight: 700; color: {TEXT}; }}
            QLabel#fieldLabel {{ font-size: 12px; font-weight: 650; color: {MUTED}; }}
            QLabel#eyebrow {{ font-size: 11px; font-weight: 700; color: {MUTED}; }}
            QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
            QLabel#bigTime {{ color: {TEAL}; font-size: 35px; font-weight: 800; }}
            QLabel#metricValue {{ font-size: 25px; font-weight: 750; }}
            QLabel#resultLead {{ font-size: 15px; font-weight: 650; }}
            QLabel#notice {{ background:{SURFACE_2}; color:{MUTED}; border-radius:9px;
                             padding:12px 15px; font-size:12px; }}
            QComboBox, QDateEdit, QDateTimeEdit, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {SURFACE_2}; color: {TEXT};
                border: 1px solid #40526D; border-radius: 8px;
                padding: 8px 12px; selection-background-color: {AMBER}; }}
            QComboBox:hover, QDateEdit:hover, QDateTimeEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {AMBER}; }}
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
            QFrame#methodPanel {{ background: #18283A; border:1px solid #354B63; border-radius: 10px; }}
            QPushButton#methodChoice {{ background: #1B2A40; color: {TEXT};
                border: 1px solid #3B526C; border-radius: 10px;
                padding: 13px 12px; font-size: 13px; font-weight: 650; text-align: left; }}
            QPushButton#methodChoice:hover {{ border-color: {BLUE}; background:#23354D; }}
            QPushButton#methodChoice:checked {{ background:#143A3B; color: #E9FFFB;
                border: 2px solid {TEAL}; font-weight: 800; }}
            QPushButton#methodChoice:checked:hover {{ background:#164345; }}
            QPushButton#methodChoice:disabled {{ background:#182235; color:#718199; border-color:#354359; }}
            QCheckBox {{ spacing:10px; color:{TEXT}; }}
            QCheckBox::indicator {{ width:18px; height:18px; }}
            QTabWidget::pane {{ border:0; background:{BG}; }}
            QTabBar::tab {{ color:{MUTED}; background:transparent;
                border:0; border-bottom:3px solid transparent;
                padding: 13px 23px; margin-right:6px; border-radius:0; }}
            QTabBar::tab:selected {{ color:{TEXT}; border-bottom:3px solid {AMBER}; font-weight:800; }}
            QToolButton#linkButton {{ background:transparent; color:{AMBER};
                border:none; padding:5px 6px; font-size:12px; font-weight:700; }}
            QToolButton#linkButton:hover {{ color:#FFC185; }}
            QToolButton#linkButton:checked {{ color:{TEAL}; }}
            QTabBar::tab:hover {{ color:{TEXT}; background:{SURFACE_2}; }}
            QTableWidget {{ background:{SURFACE}; alternate-background-color:#1A293D;
                color:{TEXT}; border:1px solid {BORDER}; border-radius:8px;
                outline:none; padding:2px; }}
            QHeaderView::section {{ background:#24344C; color:{MUTED};
                border:0; padding:10px 8px; font-size:12px; font-weight:700; }}
            QTableCornerButton::section {{ background:#24344C; border:0; }}
            QScrollArea {{ border:0; }}
            QScrollBar:vertical {{ background:{BG}; width:10px; margin:0; }}
            QScrollBar::handle:vertical {{ background:#455A75; min-height:32px; border-radius:5px; }}
            QScrollBar::handle:vertical:hover {{ background:#6785A5; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QStatusBar {{ background:{BG}; color:{MUTED}; font-size:11px; border-top:1px solid {BORDER}; }}
            QToolTip {{ background:{SURFACE_2}; color:{TEXT}; border:1px solid {BORDER}; padding:6px; }}
        """)

    def _build_window(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 16, 22, 10)
        outer.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(13)
        top.addWidget(SunGlyph())
        branding = QVBoxLayout()
        branding.setContentsMargins(0, 0, 0, 0)
        branding.setSpacing(3)
        branding.addWidget(label("SunPilot AI", "brand"))
        branding.addWidget(label("Smarter timing for flexible computing", "muted"))
        top.addLayout(branding)
        top.addStretch(1)
        self.header_location = label(self.location.name, "eyebrow")
        self.header_location.setMaximumWidth(210)
        self.header_location.setToolTip(self.location.name)
        self.header_location.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.header_location)
        self.header_version = label("v1.0", "eyebrow")
        self.header_version.setToolTip("SunPilot AI v1.0 release candidate")
        top.addSpacing(8)
        top.addWidget(self.header_version)
        outer.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_live(), "Plan")
        self.tabs.addTab(self._build_automation(), "Automate")
        self.tabs.addTab(self._build_history(), "Simulator")
        self.tabs.addTab(self._build_research(), "Research")
        self.tabs.setTabToolTip(0, "Get a forecast-based recommended time")
        self.tabs.setTabToolTip(1, "Choose an .exe and register a Windows launch")
        self.tabs.setTabToolTip(2, "Examine 2024–2025 historical AI predictions")
        self.tabs.setTabToolTip(3, "Methods, benchmarks and limitations")
        self.statusBar().setSizeGripEnabled(False)

    def _go_to_automation(self):
        """Move to the explicit opt-in Windows launcher; never schedule automatically."""
        self.use_suggestion.setChecked(True)
        self._update_automation_controls()
        self.tabs.setCurrentIndex(1)

    def _go_to_plan(self):
        self.tabs.setCurrentIndex(0)

    def _toggle_forecast_details(self, shown: bool):
        self.forecast_details.setVisible(shown)
        self.forecast_detail_toggle.setText(
            "Hide solar chart and comparison  ▴" if shown
            else "View solar chart and method comparison  ▾"
        )

    def _build_live(self) -> QScrollArea:
        scroll, page = make_scroll_page()
        page.addWidget(page_heading(
            "Plan your next task",
            "Find a time with stronger expected sunlight. Scheduling a program is always optional.",
        ))

        # Compact location summary. Edit controls only show if requested.
        loc_card = Card(margins=(20, 14, 20, 14), spacing=10)
        loc_top = QHBoxLayout()
        loc_top.setSpacing(9)
        location_info = QVBoxLayout()
        location_info.setContentsMargins(0, 0, 0, 0)
        location_info.setSpacing(4)
        location_info.addWidget(label("PLANNING LOCATION", "eyebrow"))
        self.active_location_label = label(self.location.name, "resultLead")
        location_info.addWidget(self.active_location_label)
        loc_top.addLayout(location_info, 1)
        self.location_toggle = QToolButton()
        self.location_toggle.setObjectName("linkButton")
        self.location_toggle.setText("Change location  ▾")
        self.location_toggle.setCheckable(True)
        self.location_toggle.toggled.connect(self._toggle_location_editor)
        loc_top.addWidget(self.location_toggle, 0, Qt.AlignmentFlag.AlignVCenter)
        loc_card.body.addLayout(loc_top)

        self.location_editor = QWidget()
        location_editor_layout = QVBoxLayout(self.location_editor)
        location_editor_layout.setContentsMargins(0, 9, 0, 0)
        location_editor_layout.setSpacing(11)
        location_editor_layout.addWidget(label(
            "Find a city or use an approximate IP location. No location is read automatically.", "muted"))
        city_row = QHBoxLayout()
        city_row.setSpacing(8)
        self.city_edit = QLineEdit()
        self.city_edit.setPlaceholderText("City, country (e.g. Stockholm, Sweden)")
        self.city_edit.setClearButtonEnabled(True)
        self.city_edit.returnPressed.connect(self.search_city)
        city_row.addWidget(self.city_edit, 1)
        self.city_button = QPushButton("Search")
        self.city_button.clicked.connect(self.search_city)
        city_row.addWidget(self.city_button)
        location_editor_layout.addLayout(city_row)
        self.ip_button = QPushButton("Use approximate location from IP")
        self.ip_button.setToolTip("Only when clicked: a request is made to ipwho.is using your public IP. Not GPS.")
        self.ip_button.clicked.connect(self.locate_ip)
        location_editor_layout.addWidget(self.ip_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.coordinates_toggle = QToolButton()
        self.coordinates_toggle.setObjectName("linkButton")
        self.coordinates_toggle.setCheckable(True)
        self.coordinates_toggle.setText("Enter coordinates instead  ▾")
        location_editor_layout.addWidget(self.coordinates_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.coordinates_editor = QWidget()
        coord_row = QHBoxLayout(self.coordinates_editor)
        coord_row.setContentsMargins(0, 0, 0, 0)
        coord_row.setSpacing(10)
        self.lat_input = NoWheelDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(5)
        self.lat_input.setSingleStep(.01)
        self.lat_input.setValue(self.location.latitude)
        self.lon_input = NoWheelDoubleSpinBox()
        self.lon_input.setRange(-180, 180)
        self.lon_input.setDecimals(5)
        self.lon_input.setSingleStep(.01)
        self.lon_input.setValue(self.location.longitude)
        self.coords_button = QPushButton("Apply")
        self.coords_button.clicked.connect(self.use_coordinates)
        coord_row.addWidget(Field("LATITUDE", self.lat_input), 1)
        coord_row.addWidget(Field("LONGITUDE", self.lon_input), 1)
        coord_row.addWidget(self.coords_button, 0, Qt.AlignmentFlag.AlignBottom)
        self.coordinates_editor.setVisible(False)
        self.coordinates_toggle.toggled.connect(self.coordinates_editor.setVisible)
        location_editor_layout.addWidget(self.coordinates_editor)
        self.location_editor.setVisible(False)
        loc_card.body.addWidget(self.location_editor)
        page.addWidget(loc_card)

        # Three uncomplicated controls. The scientific AI option is tucked
        # away but still accessible without losing the capability.
        self.target_day = NoWheelComboBox()
        self.target_day.addItems(["Today", "Tomorrow", "Next 7 days", "Up to 16 days"])
        self.hours = NoWheelSpinBox()
        self.hours.setRange(0, 100000)
        self.hours.setValue(2)
        self.hours.setSuffix(" h")
        self.hours.setToolTip("No arbitrary duration limit; real limit is forecast coverage")
        self.minutes = NoWheelSpinBox()
        self.minutes.setRange(0, 59)
        self.minutes.setSuffix(" min")
        self.minutes.setToolTip("Minute resolution is estimated from hourly forecast data")
        duration_control = QWidget()
        duration_layout = QHBoxLayout(duration_control)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.setSpacing(8)
        duration_layout.addWidget(self.hours, 1)
        duration_layout.addWidget(self.minutes, 1)
        self.method = MethodPicker()
        self.method.setToolTip("Choose how your task time will be selected")
        self.plan_button = QPushButton("Find best time  →")
        self.plan_button.setObjectName("primary")
        self.plan_button.clicked.connect(self.get_live)
        self.plan_controls = ResponsiveFields([
            Field("START DATE", self.target_day),
            Field("DURATION", duration_control),
        ], self.plan_button)
        # The method decision is central to SunPilot, so it must be visible.
        # Place it between the duration fields and the Find best time button.
        self.method_panel = QFrame()
        self.method_panel.setObjectName("methodPanel")
        method_layout = QVBoxLayout(self.method_panel)
        method_layout.setContentsMargins(16, 15, 16, 14)
        method_layout.setSpacing(9)
        method_layout.addWidget(label("HOW SHOULD WE PICK THE TIME?", "eyebrow"))
        method_layout.addWidget(self.method)
        self.method_explanation = label(
            "Recommended for everyday use. Uses Open-Meteo's upcoming hourly sunlight forecast.", "muted")
        method_layout.addWidget(self.method_explanation)
        self.plan_controls.body.insertWidget(1, self.method_panel)
        page.addWidget(self.plan_controls)

        result = Card(margins=(22, 18, 22, 18), spacing=9)
        heading = QHBoxLayout()
        heading.addWidget(label("SUGGESTED TIME", "eyebrow"))
        heading.addStretch(1)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self.copy_recommendation)
        self.copy_btn.hide()  # Reveal only when a recommendation exists.
        heading.addWidget(self.copy_btn)
        self.quick_schedule_btn = QPushButton("Schedule a program  →")
        self.quick_schedule_btn.setObjectName("primary")
        self.quick_schedule_btn.setToolTip("Open the Automate tab; nothing launches without confirmation")
        self.quick_schedule_btn.clicked.connect(self._go_to_automation)
        self.quick_schedule_btn.hide()
        heading.addWidget(self.quick_schedule_btn)
        result.body.addLayout(heading)
        self.live_time = label("Your recommendation appears here", "bigTime")
        self.live_time.setStyleSheet(f"color:{MUTED}; font-size:26px; font-weight:750;")
        self.live_time.setWordWrap(True)
        result.body.addWidget(self.live_time)
        self.live_lead = label("Choose when, how long and a planning method, then find your time.", "resultLead")
        result.body.addWidget(self.live_lead)
        self.live_detail = label("No program will launch until you separately confirm it in Automate.", "muted")
        result.body.addWidget(self.live_detail)
        page.addWidget(result)

        # Scheduling and managing programs live on a dedicated tab.
        # No executable is launched by the planning action.

        # Any change to forecast inputs invalidates the previous suggestion.
        self.target_day.currentIndexChanged.connect(self._invalidate_plan)
        self.hours.valueChanged.connect(self._invalidate_plan)
        self.minutes.valueChanged.connect(self._invalidate_plan)
        self.method.currentIndexChanged.connect(self._method_changed)

        self.forecast_detail_toggle = QToolButton()
        self.forecast_detail_toggle.setObjectName("linkButton")
        self.forecast_detail_toggle.setText("View solar chart and method comparison  ▾")
        self.forecast_detail_toggle.setCheckable(True)
        self.forecast_detail_toggle.toggled.connect(self._toggle_forecast_details)
        page.addWidget(self.forecast_detail_toggle, alignment=Qt.AlignmentFlag.AlignLeft)

        self.forecast_details = QWidget()
        details_layout = QVBoxLayout(self.forecast_details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(16)
        self.live_chart_card, self.live_fig, self.live_canvas = chart_card(
            "Solar outlook", "Estimated sunlight over time; the selected task is highlighted")
        details_layout.addWidget(self.live_chart_card)
        blank_chart(self.live_fig, self.live_canvas, "Your solar forecast will appear here")
        compare = Card(spacing=10)
        compare.body.addWidget(label("Compare scheduling methods", "cardTitle"))
        compare.body.addWidget(label("Estimated sunlight falling on 1 m²; not electricity generation.", "muted"))
        self.live_table = data_table()
        compare.body.addWidget(self.live_table)
        details_layout.addWidget(compare)
        self.forecast_details.hide()
        page.addWidget(self.forecast_details)
        page.addWidget(label(
            "Weather-based recommendation, not guaranteed solar-panel output. "
            "The SunPilot ML option remains experimental and is not live-validated.", "notice"))
        page.addStretch(1)
        return scroll

    def _toggle_location_editor(self, shown: bool):
        self.location_editor.setVisible(shown)
        self.location_toggle.setText("Close  ▴" if shown else "Change location  ▾")


    def _build_automation(self) -> QScrollArea:
        """Dedicated opt-in Windows automation screen; no executable is run here directly."""
        scroll, page = make_scroll_page()
        page.addWidget(page_heading(
            "Automate a program",
            "Choose a Windows program and schedule one launch. Nothing is executed without your confirmation."
        ))
        guide = Card(margins=(20, 17, 20, 17), spacing=9)
        guide.body.addWidget(label("HOW IT WORKS", "eyebrow"))
        guide.body.addWidget(label(
            "Choose a trusted .exe, select its start time and confirm. "
            "Windows handles the launch even if SunPilot is closed.", "resultLead"))
        page.addWidget(guide)
        card = Card(margins=(20, 18, 20, 18), spacing=12)
        card.body.addWidget(label("Choose your program", "cardTitle"))
        card.body.addWidget(label(
            "Select an .exe you trust. Windows will open it once at the scheduled time.", "muted"))

        program = QHBoxLayout()
        self.executable_label = label("No executable selected", "muted")
        self.executable_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        program.addWidget(self.executable_label, 1)
        self.browse_exe_button = QPushButton("Choose .exe")
        self.browse_exe_button.clicked.connect(self.browse_executable)
        program.addWidget(self.browse_exe_button)
        card.body.addLayout(program)

        self.use_suggestion = QCheckBox("Use my solar recommendation from Plan")
        # Manual scheduling must work even when the user has not used Plan.
        self.use_suggestion.setChecked(False)
        self.use_suggestion.toggled.connect(self._update_automation_controls)
        card.body.addWidget(self.use_suggestion)
        self.suggestion_hint = label("No recommendation yet — go to Plan, or choose a manual start time below.", "muted")
        card.body.addWidget(self.suggestion_hint)

        manual = QHBoxLayout()
        manual.addWidget(label("Manual start (this PC):", "fieldLabel"))
        self.manual_start = NoWheelDateTimeEdit()
        self.manual_start.setDisplayFormat("yyyy-MM-dd  HH:mm")
        self.manual_start.setCalendarPopup(True)
        self.manual_start.setDateTime(QDateTime.currentDateTime().addSecs(10 * 60))
        self.manual_start.setMinimumWidth(210)
        self.manual_start.setToolTip("Windows local date and time, not the planning location's time zone")
        manual.addWidget(self.manual_start)
        manual.addStretch(1)
        card.body.addLayout(manual)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.schedule_exe_button = QPushButton("Confirm scheduled launch")
        self.schedule_exe_button.setObjectName("primary")
        self.schedule_exe_button.clicked.connect(self.schedule_executable)
        actions.addWidget(self.schedule_exe_button)
        card.body.addLayout(actions)
        self.launch_status = label(
            "The PC must be awake and you must be signed in when the task starts. "
            "The duration is used for solar planning; SunPilot will NOT stop the program after that duration.",
            "muted")
        card.body.addWidget(self.launch_status)

        self.launch_list_toggle = QToolButton()
        self.launch_list_toggle.setObjectName("linkButton")
        self.launch_list_toggle.setText("View scheduled launches  ▾")
        self.launch_list_toggle.setCheckable(True)
        self.launch_list_toggle.toggled.connect(self._toggle_schedule_list)
        card.body.addWidget(self.launch_list_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.launch_list_panel = QWidget()
        list_layout = QVBoxLayout(self.launch_list_panel)
        list_layout.setContentsMargins(0, 5, 0, 0)
        list_layout.setSpacing(9)
        self.launch_table = NoWheelTable(0, 3)
        self.launch_table.setHorizontalHeaderLabels(["Program", "Start · PC time", "Windows status"])
        self.launch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.launch_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.launch_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.launch_table.verticalHeader().hide()
        self.launch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.launch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.launch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.launch_table.setMinimumHeight(155)
        list_layout.addWidget(self.launch_table)
        buttons = QHBoxLayout()
        self.refresh_launches_button = QPushButton("Refresh")
        self.refresh_launches_button.clicked.connect(self.refresh_schedule_list)
        buttons.addWidget(self.refresh_launches_button)
        self.cancel_launch_button = QPushButton("Cancel selected")
        self.cancel_launch_button.clicked.connect(self.cancel_selected_launch)
        buttons.addWidget(self.cancel_launch_button)
        buttons.addStretch(1)
        list_layout.addLayout(buttons)
        self.launch_list_panel.hide()
        card.body.addWidget(self.launch_list_panel)
        page.addWidget(card)
        back = QToolButton()
        back.setObjectName("linkButton")
        back.setText("← Back to solar planning")
        back.clicked.connect(self._go_to_plan)
        page.addWidget(back, alignment=Qt.AlignmentFlag.AlignLeft)
        page.addStretch(1)
        self._update_automation_controls()
        return scroll

    def _method_changed(self, index: int):
        if index == 1:
            self.method_explanation.setText(
                "Experimental: uses our trained AI model. Its live performance has not been "
                "validated outside the historical Klippan experiment."
            )
            self.method_explanation.setStyleSheet(f"color:{AMBER};")
        else:
            self.method_explanation.setText(
                "Recommended for everyday use. Uses Open-Meteo's upcoming hourly sunlight forecast."
            )
            self.method_explanation.setStyleSheet(f"color:{MUTED};")
        self._invalidate_plan()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "header_location"):
            self.header_location.setVisible(self.width() >= 890)
        if hasattr(self, "header_version"):
            self.header_version.setVisible(self.width() >= 660)

    def _invalidate_plan(self, *args):
        self.last_plan = None
        if hasattr(self, "quick_schedule_btn"):
            self.quick_schedule_btn.hide()
        if hasattr(self, "use_suggestion") and self.use_suggestion.isChecked():
            self.use_suggestion.setChecked(False)
        if hasattr(self, "suggestion_hint"):
            self.suggestion_hint.setText("No active recommendation — use Plan, or select a manual start time.")
            self._update_automation_controls()

    def _update_automation_controls(self, *args):
        if not hasattr(self, "schedule_exe_button"):
            return
        self.use_suggestion.setEnabled(self.last_plan is not None)
        suggested = self.use_suggestion.isChecked()
        self.manual_start.setEnabled(not suggested)
        ready = bool(self.selected_exe) and (not suggested or self.last_plan is not None)
        self.schedule_exe_button.setEnabled(ready and not (
            self.scheduler_worker is not None and self.scheduler_worker.isRunning()
        ))
        if self.last_plan is not None:
            when_pc = self.last_plan.start_utc.astimezone()
            self.suggestion_hint.setText(
                f"Solar recommendation: {when_pc:%a %d %b %Y, %H:%M} (this PC's local time)."
            )
        else:
            self.suggestion_hint.setText(
                "Optional: find a solar window in Plan to unlock this option; manual time works now."
            )

    def browse_executable(self):
        source = str(Path(self.selected_exe).parent) if self.selected_exe else str(Path.home())
        filepath, _ = QFileDialog.getOpenFileName(self, "Choose a trusted Windows program", source,
                                                  "Windows applications (*.exe)")
        if not filepath:
            return
        self.selected_exe = filepath
        self.executable_label.setText(Path(filepath).name)
        self.executable_label.setToolTip(filepath)
        self._update_automation_controls()

    def schedule_executable(self):
        if not self.selected_exe:
            QMessageBox.information(self, "Select a program", "Choose a .exe before scheduling it.")
            return
        if self.use_suggestion.isChecked():
            if self.last_plan is None:
                QMessageBox.information(self, "Find a recommended time", "Run Find best time first.")
                return
            when_utc = self.last_plan.start_utc
            source = "SunPilot forecast recommendation"
        else:
            # Qt converts local time to Unix milliseconds with correct DST rules.
            when_utc = datetime.fromtimestamp(self.manual_start.dateTime().toMSecsSinceEpoch() / 1000,
                                              tz=timezone.utc)
            source = "Manually selected time"
        try:
            from src.windows_scheduler import validate_request
            path, when_utc = validate_request(self.selected_exe, when_utc)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "Cannot schedule", str(exc))
            return
        pc_time = when_utc.astimezone().strftime("%A %d %B %Y at %H:%M")
        confirm = QMessageBox.question(self, "Confirm executable launch",
            f"Program: {path}\n\nStart: {pc_time} (this computer)\n"
            f"Source: {source}\n\n"
            "Windows Task Scheduler will run this .exe ONCE at that time.\n"
            "Only schedule programs that you trust. Your Windows account must be logged in, "
            "and this computer must be awake. The .exe must stay at its current path.\n\n"
            "Create this task?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._start_scheduler("register", {"exe": str(path), "when_utc": when_utc})

    def _toggle_schedule_list(self, expanded: bool):
        self.launch_list_panel.setVisible(expanded)
        self.launch_list_toggle.setText("Hide scheduled launches  ▴" if expanded else "View scheduled launches  ▾")
        if expanded:
            self.refresh_schedule_list()

    def refresh_schedule_list(self):
        self._start_scheduler("list")

    def cancel_selected_launch(self):
        row = self.launch_table.currentRow()
        item = self.launch_table.item(row, 0) if row >= 0 else None
        task_name = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not task_name:
            QMessageBox.information(self, "Choose a launch", "Select a scheduled launch to cancel.")
            return
        confirm = QMessageBox.question(self, "Cancel this scheduled launch",
                                       f"Remove Windows task {task_name}?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            self._start_scheduler("cancel", task_name)

    def _start_scheduler(self, action: str, data=None):
        if self.scheduler_worker is not None and self.scheduler_worker.isRunning():
            return
        self.scheduler_worker = ScheduledLaunchWorker(action, data)
        self.scheduler_worker.ready.connect(self._scheduler_result)
        self.scheduler_worker.failed.connect(self._scheduler_error)
        self.scheduler_worker.finished.connect(self._scheduler_finished)
        self.scheduler_worker.start()
        self.schedule_exe_button.setEnabled(False)
        self.refresh_launches_button.setEnabled(False)
        self.cancel_launch_button.setEnabled(False)
        self.statusBar().showMessage("Contacting Windows Task Scheduler…")

    def _scheduler_result(self, operation: str, data):
        if operation == "register":
            self.launch_status.setText(
                f"Scheduled successfully · {data['run_local'].replace('T', ' ')} (PC local time) "
                f"· {Path(data['exe']).name}. You can close SunPilot."
            )
            self.statusBar().showMessage("Windows confirmed the scheduled program launch")
            self._scheduler_refresh_pending = self.launch_list_panel.isVisible()
        elif operation == "cancel":
            self.launch_status.setText("The selected launch has been cancelled in Windows.")
            self._scheduler_refresh_pending = True
        elif operation == "list":
            rows = data or []
            self.launch_table.setRowCount(len(rows))
            for i, info in enumerate(rows):
                filename = Path(info.get("exe", "")).name
                scheduled = str(info.get("run_local", "")).replace("T", " ")[:16]
                for j, value in enumerate((filename, scheduled, info.get("status", "?"))):
                    cell = QTableWidgetItem(value)
                    cell.setToolTip(info.get("exe", ""))
                    if j == 0:
                        cell.setData(Qt.ItemDataRole.UserRole, info.get("name"))
                    self.launch_table.setItem(i, j, cell)
                self.launch_table.setRowHeight(i, 36)
            self.launch_table.setMinimumHeight(min(300, max(155, 42 + len(rows) * 37)))
            self.launch_status.setText(f"{len(rows)} SunPilot launch(es) registered in Windows.")

    def _scheduler_error(self, message: str):
        self.statusBar().showMessage("Windows could not complete the operation")
        QMessageBox.warning(self, "Windows Task Scheduler", message +
                            "\n\nCheck that Task Scheduler is available and your Windows account "
                            "has permission to register tasks.")

    def _scheduler_finished(self):
        self.scheduler_worker = None
        self.refresh_launches_button.setEnabled(True)
        self.cancel_launch_button.setEnabled(True)
        self._update_automation_controls()
        if self._scheduler_refresh_pending:
            self._scheduler_refresh_pending = False
            QTimer.singleShot(0, self.refresh_schedule_list)

    def _build_history(self) -> QScrollArea:
        scroll, page = make_scroll_page()
        page.addWidget(page_heading(
            "Explore the historical simulator",
            "Choose a day from 2024–2025 to see what the trained AI would have decided.",
        ))
        # Default to the verified worst case once corrected 2025 metrics exist.
        worst_date = ((self.scheduler_metrics or {}).get("test_2025", {})
                      .get("worst_day_vs_yesterday"))
        worst_qdate = QDate.fromString(worst_date or "", "yyyy-MM-dd")
        default_day = worst_qdate if worst_qdate.isValid() and (self.scheduler_metrics or {}).get("evaluation_version") == 2 else QDate(2025, 8, 26)
        self.hist_date = NoWheelDateEdit(default_day)
        self.hist_date.setDisplayFormat("yyyy-MM-dd")
        self.hist_date.setCalendarPopup(True)
        self.hist_date.setDateRange(QDate(2024, 1, 1), QDate(2025, 12, 31))
        self.hist_hours = NoWheelSpinBox()
        self.hist_hours.setRange(0, 100000)
        self.hist_hours.setValue(2)
        self.hist_hours.setSuffix(" h")
        self.hist_minutes = NoWheelSpinBox()
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
        comparison.body.addWidget(label("Historical ERA5 reference results", "cardTitle"))
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
        # Old v1.0 metrics used 08–17 hour-end labels. Never silently show
        # pre-correction values after installing the new evaluator.
        is_recalculated = ((self.scheduler_metrics or {}).get("hour_label_semantics") == "hour_end"
                           and (self.scheduler_metrics or {}).get("evaluation_version") == 2)
        test = (self.scheduler_metrics or {}).get("test_2025", {}) if is_recalculated else {}
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
        if not is_recalculated:
            page.addWidget(label(
                "Scheduling benchmark pending corrected evaluation: run python src/scheduler.py "
                "before quoting any historical scheduling gains. The previous values used a "
                "one-hour misalignment.", "notice"))
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

        # A measured carbon benefit cannot be inferred from irradiance. Show a
        # hypothetical rooftop-PV user scenario ONLY after a corrected audit.
        pv_scenario = test.get("pv_scenario_mean_kwh", {}) if is_recalculated else {}
        if pv_scenario:
            scenario = Card()
            scenario.body.addWidget(label("Illustrative rooftop PV scenario · NOT measured", "cardTitle"))
            scenario.body.addWidget(label(
                "Assumptions: 4 m² horizontal panels, nominal efficiency 20%, "
                "300 W computer, no batteries, shading or other loads. "
                "Potential solar-powered share of a 2-hour workload; "
                "not PV meter data or avoided CO₂.", "muted"))
            for name in ("SunPilot AI", "Yesterday", "Fixed 12:00"):
                if name in pv_scenario:
                    line = QHBoxLayout()
                    line.addWidget(label(name, "muted"), 2)
                    result = label(f"{pv_scenario[name]:.3f} kWh / task", "resultLead")
                    result.setAlignment(Qt.AlignmentFlag.AlignRight)
                    line.addWidget(result, 1)
                    scenario.body.addLayout(line)
            page.addWidget(scenario)

        finding = Card()
        finding.body.addWidget(label("The honest conclusion", "cardTitle"))
        finding.body.addWidget(label(
            "In the historical held-out test, the AI reduced prediction error relative to a "
            "previous-day baseline. It beat yesterday-based scheduling but did not beat a fixed "
            "noon schedule. Better predictions do not necessarily lead to better decisions."))
        finding.body.addWidget(label(
            "The historical evaluation now aligns hour-end radiation labels with real task start "
            "times (e.g. 09:00 radiation describes 08:00–09:00). Results must be "
            "regenerated with the corrected evaluator before interpretation.", "muted"))
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
            "The intended user is someone with rooftop solar and a flexible computing job. "
            "Historical solar irradiation is a proxy, not proof of solar self-consumption. "
            "No measured PV power, financial savings, or verified carbon reduction. "
            "Historical research location: Klippan, Sweden. Live planning supports user-chosen locations. "
            "The historical one-day experiment uses the 08:00–18:00 evaluation window. "
            "Windows program launches require a separate opt-in and are not measured carbon reductions. "
            "Source code and reproducibility details: github.com/Mivq27/SunPilot-AI."))
        page.addWidget(sources)
        page.addStretch(1)
        return scroll

    def _apply_location(self, location: Location):
        self.location = location
        self.active_location_label.setText(location.name)
        self.header_location.setText(location.name)
        self.header_location.setToolTip(location.name)
        self.lat_input.setValue(location.latitude)
        self.lon_input.setValue(location.longitude)
        self.settings.setValue("location/name", location.name)
        self.settings.setValue("location/latitude", location.latitude)
        self.settings.setValue("location/longitude", location.longitude)
        self.settings.setValue("location/timezone", location.timezone or "")
        self.copy_btn.setEnabled(False)
        self.copy_btn.hide()
        self.location_toggle.setChecked(False)
        self.copy_text = ""
        self._invalidate_plan()
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
            # Hourly shortwave_radiation is averaged over the preceding hour.
            # To evaluate a true 08:00–18:00 job window, use end-hour labels
            # 09:00–18:00. Identical selection is used in src/scheduler.py.
            slice_ = historical_day(self.history, day)
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
        self._invalidate_plan()
        self.pending_plan = (self.target_day.currentIndex(),
                             duration, self.method.currentIndex(), self.location)
        for control in (self.target_day, self.hours, self.minutes,
                        self.method, self.city_button, self.ip_button, self.coords_button,
                        self.lat_input, self.lon_input, self.city_edit,
                        self.location_toggle, self.coordinates_toggle):
            control.setEnabled(False)
        self.plan_button.setEnabled(False)
        self.plan_button.setText("Checking forecast…")
        self.copy_btn.setEnabled(False)
        self.copy_btn.hide()
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
        for control in (self.target_day, self.hours, self.minutes,
                        self.method, self.city_button, self.ip_button, self.coords_button,
                        self.lat_input, self.lon_input, self.city_edit,
                        self.location_toggle, self.coordinates_toggle):
            control.setEnabled(True)
        self.plan_button.setEnabled(True)
        self.plan_button.setText("Find best time  →")
        self.pending_plan = None
        self.weather_worker = None

    def weather_error(self, message: str):
        self.live_time.setText("Plan unavailable")
        self.live_lead.setText("SunPilot could not calculate a reliable schedule.")
        self.live_detail.setText("Check the forecast connection or choose a different start range. "
                                 "Tasks longer than the forecast horizon cannot be evaluated.")
        self.copy_btn.setEnabled(False)
        self.copy_btn.hide()
        self.copy_text = ""
        self._invalidate_plan()
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
            when, duration, method_index, location = self.pending_plan
            plan = build_plan(forecast, self.model, duration, when, method_index)
            tz = ZoneInfo(plan.timezone)
            start_local = plan.start_utc.astimezone(tz)
            end_local = plan.end_utc.astimezone(tz)
            display_time = f"{start_local:%a %d %b, %H:%M} – {end_local:%a %d %b, %H:%M}"
            self.live_time.setStyleSheet(f"color:{AMBER if method_index == 1 else TEAL};font-size:28px;font-weight:800;")
            self.live_time.setText(display_time)
            self.live_time.setWordWrap(True)
            source_label = "SunPilot AI (experimental)" if method_index == 1 else "Weather forecast (recommended)"
            self.live_lead.setText(f"{source_label}  ·  {human_duration(duration)}  ·  {location.name}")
            note = " · experimental, not live-validated" if method_index == 1 else " · recommended for everyday use"
            self.live_detail.setText(
                f"{plan.method}{note}. Time zone: {plan.timezone}. "
                f"Estimated incident solar irradiation: {plan.estimated_wh_m2:.1f} Wh/m². "
                "It is not measured panel electricity. Minutes are estimated from hourly data."
            )
            self.copy_text = (
                f"SunPilot AI | {location.name} | "
                f"{start_local:%Y-%m-%d %H:%M} to {end_local:%Y-%m-%d %H:%M} "
                f"({plan.timezone}) | {human_duration(duration)} | {plan.method} | "
                "Weather forecast estimate only; not guaranteed PV production."
            )
            self.copy_btn.setEnabled(True)
            self.copy_btn.show()
            self.quick_schedule_btn.show()
            self.last_plan = plan
            self._update_automation_controls()
            rows = []
            for method, (start_utc, estimate) in plan.comparison.items():
                local_start = start_utc.astimezone(tz)
                rows.append((method, local_start.strftime("%d %b %H:%M"), f"{estimate:.1f}"))
            fill_table(self.live_table, rows, emphasize=1 if method_index == 1 else 0)
            self._plot_live(plan)
            self.statusBar().showMessage(
                f"Forecast received · {location.name} · {plan.timezone} · {human_duration(duration)} · "
                "program launch available with confirmation"
            )
        except Exception as exc:
            self.weather_error(f"{type(exc).__name__}: {exc}")

    def copy_recommendation(self):
        if self.copy_text:
            QApplication.clipboard().setText(self.copy_text)
            self.statusBar().showMessage("Schedule copied to clipboard")

    def closeEvent(self, event):
        # Avoid destroying a running QThread. The HTTP timeout bounds the wait.
        if (self.weather_worker is not None and self.weather_worker.isRunning()) or (self.location_worker is not None and self.location_worker.isRunning()) or (self.scheduler_worker is not None and self.scheduler_worker.isRunning()):
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
