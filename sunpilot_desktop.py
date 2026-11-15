"""SunPilot AI native desktop prototype (PySide6 + Matplotlib).

Uses the existing SunPilot project artifacts. This is a historical backtest,
not a live solar forecast or a program that executes third-party applications.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# In a PyInstaller bundle, files passed via --add-data live under _MEIPASS.
# Set the working directory *before* importing src.train, which uses relative paths.
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
os.chdir(ROOT)
from src.train import FEATURES, load_data  # noqa: E402

START_HOUR = 8
END_HOUR = 18
FIXED_START = 12
BG = "#0b1020"
SURFACE = "#151c2f"
TEXT = "#edf2fc"
MUTED = "#9ca9bf"
ORANGE = "#f97316"
BLUE = "#38bdf8"
GREEN = "#34d399"


class SunPilotWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SunPilot AI | Solar-Aware Compute Scheduling")
        self.resize(1210, 875)
        self.setMinimumSize(980, 720)

        model_file = ROOT / "models" / "sunpilot_v1.joblib"
        if not model_file.is_file():
            raise FileNotFoundError(f"Model not found: {model_file}")
        self.model = joblib.load(model_file)["model"]

        df = load_data()
        df = df[df["local_date"].str[:4].isin(["2024", "2025"])].copy()
        if df.empty:
            raise ValueError("No 2024/2025 dates available in the dataset")
        df["ai_prediction"] = np.maximum(self.model.predict(df[FEATURES]), 0)
        self.data = df

        container = QWidget()
        self.setCentralWidget(container)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(25, 21, 25, 22)
        outer.setSpacing(14)

        title = QLabel("SUNPILOT  <span style='color:#f97316'>AI</span>")
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setObjectName("title")
        outer.addWidget(title)
        subtitle = QLabel("Solar-Aware Compute Scheduling  /  Historical research prototype  /  Klippan, Sweden")
        subtitle.setObjectName("muted")
        outer.addWidget(subtitle)

        controls_frame = QFrame()
        controls_frame.setObjectName("panel")
        controls = QHBoxLayout(controls_frame)
        controls.setContentsMargins(16, 14, 16, 14)
        controls.setSpacing(17)

        controls.addWidget(QLabel("Historical day"))
        self.date = QDateEdit(QDate(2025, 8, 26))
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.date.setCalendarPopup(True)
        self.date.setDateRange(QDate(2024, 1, 1), QDate(2025, 12, 31))
        controls.addWidget(self.date)

        controls.addWidget(QLabel("Render duration"))
        self.duration = QSpinBox()
        self.duration.setRange(1, 4)
        self.duration.setSuffix(" h")
        self.duration.setValue(2)
        controls.addWidget(self.duration)

        self.analyze_button = QPushButton("Analyze with AI")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self.analyze)
        controls.addWidget(self.analyze_button)
        controls.addStretch(1)
        outer.addWidget(controls_frame)

        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setWordWrap(True)
        self.summary.setObjectName("summary")
        outer.addWidget(self.summary)

        self.figure = Figure(figsize=(11.5, 5.6), facecolor=BG)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(330)
        outer.addWidget(self.canvas, 1)

        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["Strategy", "Scheduled time", "Observed irradiation (Wh/m²)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setFixedHeight(157)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 240)
        self.table.setColumnWidth(1, 220)
        outer.addWidget(self.table)

        foot = QLabel(
            "Research backtest only  •  Actual observations are shown for evaluation, "
            "never supplied as future model inputs  •  No applications are executed"
        )
        foot.setObjectName("muted")
        foot.setWordWrap(True)
        outer.addWidget(foot)

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
            QLabel#title {{ font-size: 27px; font-weight: 800; }}
            QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
            QLabel#summary {{ background: {SURFACE}; padding: 13px 18px;
                              border: 1px solid #283149; border-radius: 9px; }}
            QFrame#panel {{ background: {SURFACE}; border: 1px solid #283149; border-radius: 9px; }}
            QDateEdit, QSpinBox {{ background: #212b42; color: {TEXT}; border: 1px solid #38445f;
                                  border-radius: 6px; padding: 7px 10px; min-width: 110px; }}
            QPushButton#primaryButton {{ color: #fff; background: {ORANGE}; font-weight: bold;
                                       border: 0px; padding: 10px 20px; border-radius: 6px; }}
            QPushButton#primaryButton:hover {{ background: #ea580c; }}
            QTableWidget {{ background: {SURFACE}; alternate-background-color: #1b2540;
                            color: {TEXT}; gridline-color: #2a3650;
                            border: 1px solid #283149; }}
            QHeaderView::section {{ background: #222b42; color: {TEXT}; border: 0;
                                    padding: 8px; font-weight: bold; }}
        """)
        self.analyze()

    @staticmethod
    def best_start(values: np.ndarray, duration: int) -> int:
        scores = np.asarray([
            float(np.sum(values[i:i + duration]))
            for i in range(END_HOUR - START_HOUR - duration + 1)
        ])
        indices = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-8))
        # Exact tie-breaking matches src/scheduler.py.
        return START_HOUR + int(min(indices, key=lambda i: abs(START_HOUR + int(i) - 12)))

    @staticmethod
    def observed_energy(observed: np.ndarray, start: int, duration: int) -> float:
        offset = start - START_HOUR
        return float(np.sum(observed[offset:offset + duration]))

    def analyze(self) -> None:
        try:
            date_str = self.date.date().toString("yyyy-MM-dd")
            duration = int(self.duration.value())
            selected = self.data[
                (self.data["local_date"] == date_str)
                & self.data["local_hour"].between(START_HOUR, END_HOUR - 1)
            ].sort_values("local_hour")
            if selected["local_hour"].tolist() != list(range(START_HOUR, END_HOUR)):
                raise ValueError(f"Missing hours in {date_str}")

            observed = selected["shortwave_radiation"].to_numpy(dtype=float)
            predicted = selected["ai_prediction"].to_numpy(dtype=float)
            yesterday = selected["radiation_lag_24"].to_numpy(dtype=float)
            hours = selected["local_hour"].to_numpy(dtype=int)

            starts = {
                "SunPilot AI": self.best_start(predicted, duration),
                "Yesterday baseline": self.best_start(yesterday, duration),
                "Fixed noon": FIXED_START,
                "Perfect knowledge *": self.best_start(observed, duration),
            }
            results = {
                name: self.observed_energy(observed, start, duration)
                for name, start in starts.items()
            }
            gain_fixed = results["SunPilot AI"] - results["Fixed noon"]
            gain_yesterday = results["SunPilot AI"] - results["Yesterday baseline"]
            choice = starts["SunPilot AI"]
            self.summary.setText(
                f"<span style='color:{MUTED}'>AI recommendation</span> &nbsp; "
                f"<span style='font-size:19px; font-weight:bold'>{choice:02d}:00–{choice + duration:02d}:00</span>"
                f" &nbsp;&nbsp; <span style='color:{MUTED}'>vs fixed</span> {gain_fixed:+.1f} Wh/m²"
                f" &nbsp;&nbsp; <span style='color:{MUTED}'>vs yesterday</span> {gain_yesterday:+.1f} Wh/m²"
            )
            self.render_charts(hours, observed, predicted, yesterday, starts, results, duration, date_str)
            for row, (name, start) in enumerate(starts.items()):
                values = [name, f"{start:02d}:00–{start + duration:02d}:00", f"{results[name]:.1f}"]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 2:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(row, column, item)
            self.statusBar().showMessage(f"Historical simulation: {date_str} | {duration}h task | 08:00–18:00 local")
        except Exception as exc:
            QMessageBox.critical(self, "SunPilot AI — Analysis error", str(exc))

    def render_charts(
        self, hours: np.ndarray, observed: np.ndarray,
        predicted: np.ndarray, yesterday: np.ndarray,
        starts: dict[str, int], results: dict[str, float],
        duration: int, date_str: str,
    ) -> None:
        self.figure.clear()
        ax1, ax2 = self.figure.subplots(2, 1, gridspec_kw={"height_ratios": [2.15, 1.05]})
        self.figure.subplots_adjust(left=0.19, right=0.97, top=0.91, bottom=0.09, hspace=0.52)
        for ax in (ax1, ax2):
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.grid(axis="y", color="#34415a", alpha=0.35)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#34415a")
        ax1.plot(hours, observed, marker="o", ms=4, color=BLUE, linewidth=2.4, label="Observed (evaluation)")
        ax1.plot(hours, predicted, marker=".", ms=6, color=ORANGE, linewidth=2.2, label="SunPilot AI")
        ax1.plot(hours, yesterday, color=MUTED, linewidth=1.6, linestyle="--", label="Yesterday baseline")
        chosen = starts["SunPilot AI"]
        ax1.axvspan(chosen - 0.5, chosen + duration - 0.5, color=ORANGE, alpha=0.15)
        ax1.set_xlim(7.5, 17.5)
        ax1.set_xticks(range(8, 18))
        ax1.set_ylabel("Radiation (W/m²)", color=MUTED)
        ax1.set_title(f"Hourly solar radiation  •  {date_str}", color=TEXT, loc="left", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right", facecolor=SURFACE, edgecolor="#34415a", labelcolor=TEXT, fontsize=8)

        names = list(results)
        vals = list(results.values())
        ax2.barh(names, vals, color=[ORANGE, MUTED, "#64748b", BLUE], height=0.6)
        ax2.invert_yaxis()
        ax2.set_xlabel("Observed irradiation during the selected task (Wh/m²)", color=MUTED)
        ax2.set_xlim(0, max(vals + [1]) * 1.18)
        for idx, v in enumerate(vals):
            ax2.text(v + max(vals + [1]) * 0.012, idx, f"{v:.1f}", color=TEXT, va="center", fontsize=9)
        self.canvas.draw_idle()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        window = SunPilotWindow()
    except Exception:
        QMessageBox.critical(None, "SunPilot AI — Startup error", traceback.format_exc())
        raise
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
