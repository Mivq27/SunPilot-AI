from pathlib import Path
import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.train import load_data, FEATURES

ROOT = Path(__file__).resolve().parent
START_HOUR = 8
END_HOUR = 18

# Load our actual trained machine-learning model.
artifact = joblib.load(
    ROOT / "models/sunpilot_v1.joblib"
)
model = artifact["model"]

# Prepare historical features.
data = load_data()
data = data[
    data["local_date"].str.startswith(
        ("2024-", "2025-")
    )
].copy()

AVAILABLE_DATES = sorted(
    data["local_date"].unique().tolist(),
    reverse=True,
)

metrics = json.loads(
    (ROOT / "results/scheduler_metrics.json").read_text(
        encoding="utf-8"
    )
)


def choose_slot(values, duration):
    """Select consecutive hours maximizing predicted irradiation."""
    scores = np.array([
        np.sum(values[i:i + duration])
        for i in range(len(values) - duration + 1)
    ])

    best = np.flatnonzero(
        np.isclose(scores, scores.max(), atol=1e-8)
    )

    index = min(
        best,
        key=lambda i: abs(START_HOUR + i - 12),
    )

    return START_HOUR + int(index)


def render(selected_date, duration):
    duration = int(duration)

    day = data[
        (data["local_date"] == selected_date)
        & (data["local_hour"] >= START_HOUR)
        & (data["local_hour"] < END_HOUR)
    ].sort_values("local_hour")

    if day["local_hour"].tolist() != list(
        range(START_HOUR, END_HOUR)
    ):
        raise gr.Error("Incomplete historical day.")

    hours = day["local_hour"].to_numpy()
    actual = day["shortwave_radiation"].to_numpy()
    baseline = day["radiation_lag_24"].to_numpy()

    # Real model inference, not precomputed predictions.
    predictions = np.maximum(
        model.predict(day[FEATURES]),
        0,
    )

    ai_start = choose_slot(predictions, duration)
    baseline_start = choose_slot(baseline, duration)
    oracle_start = choose_slot(actual, duration)
    fixed_start = 12

    def actual_energy(start):
        index = start - START_HOUR
        return float(
            np.sum(actual[index:index + duration])
        )

    strategies = [
        ("SunPilot AI", ai_start),
        ("Yesterday baseline", baseline_start),
        ("Fixed 12:00", fixed_start),
        ("Perfect knowledge", oracle_start),
    ]

    oracle_energy = actual_energy(oracle_start)

    rows = []
    for name, start in strategies:
        energy = actual_energy(start)
        rows.append({
            "Strategy": name,
            "Time": (
                f"{start:02d}:00–"
                f"{start + duration:02d}:00"
            ),
            "Actual Wh/m²": round(energy, 1),
            "Gap to ideal Wh/m²": round(
                oracle_energy - energy, 1
            ),
        })

    comparison = pd.DataFrame(rows)

    # Hourly prediction graph.
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hours,
        y=actual,
        name="Observed solar radiation",
        mode="lines+markers",
        line=dict(color="#38BDF8", width=3),
    ))

    fig.add_trace(go.Scatter(
        x=hours,
        y=predictions,
        name="SunPilot AI prediction",
        mode="lines+markers",
        line=dict(color="#F97316", width=3),
    ))

    fig.add_trace(go.Scatter(
        x=hours,
        y=baseline,
        name="Yesterday baseline",
        mode="lines",
        line=dict(
            color="#94A3B8",
            width=2,
            dash="dash",
        ),
    ))

    fig.add_vrect(
        x0=ai_start,
        x1=ai_start + duration,
        fillcolor="rgba(249,115,22,0.16)",
        line_width=0,
        annotation_text="AI-selected window",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"Solar radiation — {selected_date}",
        xaxis_title="Local hour (Europe/Stockholm)",
        yaxis_title="Solar radiation (W/m²)",
        xaxis=dict(dtick=1),
        height=430,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=45, r=25, t=65, b=45),
    )

    # Comparison chart.
    bar = go.Figure(go.Bar(
        x=comparison["Strategy"],
        y=comparison["Actual Wh/m²"],
        marker_color=[
            "#F97316",
            "#94A3B8",
            "#64748B",
            "#38BDF8",
        ],
        text=comparison["Actual Wh/m²"],
        textposition="outside",
    ))

    bar.update_layout(
        title="Observed irradiation by strategy",
        yaxis_title="Wh/m² per task",
        height=370,
        template="plotly_dark",
        margin=dict(l=45, r=25, t=65, b=65),
    )

    ai_energy = actual_energy(ai_start)
    fixed_energy = actual_energy(fixed_start)
    yesterday_energy = actual_energy(baseline_start)

    delta_fixed = ai_energy - fixed_energy
    delta_yesterday = ai_energy - yesterday_energy

    year = selected_date[:4]

    summary = f"""
### Decision summary

**Selected date:** {selected_date}  
**Evaluation set:** {year}  
**Task duration:** {duration} hour(s)  
**AI-recommended start:** {ai_start:02d}:00

**SunPilot AI:** {ai_energy:.1f} Wh/m²  
**Compared with fixed schedule:** {delta_fixed:+.1f} Wh/m²  
**Compared with yesterday baseline:** {delta_yesterday:+.1f} Wh/m²

The blue line shows historical observations, which were
**not provided to the model as inputs for the predicted day**.

The perfect-knowledge strategy is an unattainable reference
using the actual observations. It is not a real forecast.

This is a historical simulation, not a live solar forecast.
"""

    return fig, bar, comparison, summary


test = metrics["test_2025"]

with gr.Blocks(
    title="SunPilot AI",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
# ☀️ SunPilot AI

### Solar-Aware Compute Scheduling

**An experimental machine-learning system that predicts
solar radiation and schedules flexible computing workloads.**

Veritas AI Hackathon 2026 · AI + Climate · Klippan, Sweden
""")

    gr.Markdown(
        "**2025 backtest:** SunPilot AI achieved "
        f"**{test['ai_gain_vs_yesterday_percent']:+.2f}%** "
        "versus yesterday-based scheduling and "
        f"**{test['ai_gain_vs_fixed_percent']:+.2f}%** "
        "versus a fixed noon schedule for two-hour tasks."
    )

    with gr.Row():
        date_input = gr.Dropdown(
            choices=AVAILABLE_DATES,
            value="2025-08-26",
            label="Historical simulation date",
        )

        duration_input = gr.Slider(
            minimum=1,
            maximum=4,
            value=2,
            step=1,
            label="Computing task duration (hours)",
        )

    run_button = gr.Button(
        "Analyze and schedule",
        variant="primary",
    )

    summary_output = gr.Markdown()

    prediction_plot = gr.Plot(
        label="Solar prediction"
    )

    comparison_plot = gr.Plot(
        label="Scheduling performance"
    )

    comparison_table = gr.Dataframe(
        label="Strategy comparison",
        interactive=False,
    )

    outputs = [
        prediction_plot,
        comparison_plot,
        comparison_table,
        summary_output,
    ]

    inputs = [date_input, duration_input]

    run_button.click(
        fn=render,
        inputs=inputs,
        outputs=outputs,
    )

    demo.load(
        fn=render,
        inputs=inputs,
        outputs=outputs,
    )

    gr.Markdown("""
---
**Research limitations**

- Historical ERA5 weather reanalysis, not direct PV measurements.
- Solar irradiation does not directly equal electricity produced.
- Higher irradiation alone does not establish lower CO₂ emissions.
- Models were trained on 2022–2023 data.
- The 2024 validation set and 2025 test set are separate.
- Results may worsen on cloudy or highly variable days.
""")

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
