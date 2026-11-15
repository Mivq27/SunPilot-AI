import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
START_HOUR = 8
END_HOUR = 18
TASK_DURATION = 2
FIXED_START = 12


def choose_best(scores):
    """Select the best slot. Break ties near noon."""
    scores = np.asarray(scores, dtype=float)
    best = np.flatnonzero(
        np.isclose(scores, scores.max(), atol=1e-8)
    )
    return int(
        min(best, key=lambda i: abs(START_HOUR + i - 12))
    )


def window_scores(values):
    """Calculate total irradiation for each 2-hour slot."""
    return np.array([
        np.sum(values[i:i + TASK_DURATION])
        for i in range(
            END_HOUR - START_HOUR - TASK_DURATION + 1
        )
    ])


def evaluate_year(year):
    path = RESULTS / (
        f"validation_predictions_{year}.csv"
        if year == 2024
        else f"test_predictions_{year}.csv"
    )

    df = pd.read_csv(path)
    records = []

    for date, group in df.groupby("local_date"):

        # Ignore local dates outside the evaluation year.
        # UTC year boundaries can include hours from the next
        # calendar year in Sweden.
        if not date.startswith(f"{year}-"):
            continue
        window = group[
            group["local_hour"].between(
                START_HOUR, END_HOUR - 1
            )
        ].copy()

        window = window.sort_values("local_hour")

        expected_hours = list(range(START_HOUR, END_HOUR))

        if window["local_hour"].tolist() != expected_hours:
            raise ValueError(
                f"Incomplete operating window: {date}"
            )

        actual = window["actual"].to_numpy()
        baseline = window["baseline"].to_numpy()
        ai = window["ai_prediction"].to_numpy()

        actual_slots = window_scores(actual)
        baseline_slots = window_scores(baseline)
        ai_slots = window_scores(ai)

        ai_index = choose_best(ai_slots)
        baseline_index = choose_best(baseline_slots)
        oracle_index = choose_best(actual_slots)
        fixed_index = FIXED_START - START_HOUR

        records.append({
            "date": date,
            "ai_start": START_HOUR + ai_index,
            "baseline_start": START_HOUR + baseline_index,
            "fixed_start": FIXED_START,
            "ai_actual_wh_m2": actual_slots[ai_index],
            "baseline_actual_wh_m2": (
                actual_slots[baseline_index]
            ),
            "fixed_actual_wh_m2": (
                actual_slots[fixed_index]
            ),
            "oracle_actual_wh_m2": (
                actual_slots[oracle_index]
            ),
            "oracle_start": START_HOUR + oracle_index,
        })

    result = pd.DataFrame(records)

    result.to_csv(
        RESULTS / f"schedule_evaluation_{year}.csv",
        index=False,
    )

    columns = {
        "Fixed 12:00": "fixed_actual_wh_m2",
        "Yesterday": "baseline_actual_wh_m2",
        "SunPilot AI": "ai_actual_wh_m2",
        "Perfect knowledge": "oracle_actual_wh_m2",
    }

    averages = {
        name: float(result[col].mean())
        for name, col in columns.items()
    }

    ai_total = result["ai_actual_wh_m2"].sum()
    fixed_total = result["fixed_actual_wh_m2"].sum()
    baseline_total = result[
        "baseline_actual_wh_m2"
    ].sum()

    gain_fixed = 100 * (
        ai_total / fixed_total - 1
    )

    gain_baseline = 100 * (
        ai_total / baseline_total - 1
    )

    wins_fixed = 100 * (
        result["ai_actual_wh_m2"]
        > result["fixed_actual_wh_m2"]
    ).mean()

    wins_baseline = 100 * (
        result["ai_actual_wh_m2"]
        > result["baseline_actual_wh_m2"]
    ).mean()

    print(f"\n=== SCHEDULING EVALUATION {year} ===")
    print(f"Days evaluated: {len(result)}")
    print("\nMean solar irradiation per task (Wh/m2):")

    for name, value in averages.items():
        print(f"  {name:<20}: {value:.2f}")

    print("\nSunPilot improvements:")
    print(f"  vs fixed schedule : {gain_fixed:+.2f}%")
    print(f"  vs yesterday      : {gain_baseline:+.2f}%")

    print("\nPercentage of days with strict improvement:")
    print(f"  vs fixed schedule : {wins_fixed:.2f}%")
    print(f"  vs yesterday      : {wins_baseline:.2f}%")

    result["difference_vs_baseline"] = (
        result["ai_actual_wh_m2"]
        - result["baseline_actual_wh_m2"]
    )

    worst = result.loc[
        result["difference_vs_baseline"].idxmin()
    ]

    print("\nWorst AI day vs yesterday:")
    print(f"  Date: {worst['date']}")
    print(f"  AI start: {int(worst['ai_start']):02d}:00")
    print(
        f"  Baseline start: "
        f"{int(worst['baseline_start']):02d}:00"
    )
    print(
        f"  Difference: "
        f"{worst['difference_vs_baseline']:.2f} Wh/m2"
    )

    # Perfect knowledge is an upper bound.
    assert (
        result["oracle_actual_wh_m2"]
        >= result["ai_actual_wh_m2"] - 1e-8
    ).all()

    return {
        "days": len(result),
        "mean_wh_m2": {
            k: round(v, 3) for k, v in averages.items()
        },
        "ai_gain_vs_fixed_percent": round(
            float(gain_fixed), 3
        ),
        "ai_gain_vs_yesterday_percent": round(
            float(gain_baseline), 3
        ),
        "ai_wins_vs_fixed_percent": round(
            float(wins_fixed), 3
        ),
        "ai_wins_vs_yesterday_percent": round(
            float(wins_baseline), 3
        ),
        "worst_day_vs_yesterday": str(worst["date"]),
    }


def create_plot(metrics):
    values = metrics["mean_wh_m2"]

    labels = [
        "Fixed 12:00",
        "Yesterday",
        "SunPilot AI",
        "Perfect knowledge",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        labels,
        [values[label] for label in labels],
    )

    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("Mean irradiation (Wh/m2 per 2h task)")
    ax.set_title("SunPilot AI - Scheduling Backtest 2025")
    ax.tick_params(axis="x", rotation=12)

    fig.tight_layout()
    fig.savefig(
        RESULTS / "schedule_comparison_2025.png",
        dpi=180,
    )
    plt.close(fig)


def main():
    print("=== SUNPILOT AI / INTELLIGENT SCHEDULER ===")

    val_metrics = evaluate_year(2024)
    test_metrics = evaluate_year(2025)

    all_metrics = {
        "validation_2024": val_metrics,
        "test_2025": test_metrics,
        "task_duration_hours": TASK_DURATION,
        "operating_window": "08:00-18:00 Europe/Stockholm",
        "note": (
            "Historical backtest using observed irradiation. "
            "This does not measure actual PV energy or CO2 savings."
        ),
    }

    (RESULTS / "scheduler_metrics.json").write_text(
        json.dumps(all_metrics, indent=2),
        encoding="utf-8",
    )

    create_plot(test_metrics)

    print("\nSaved results:")
    print("  results/scheduler_metrics.json")
    print("  results/schedule_evaluation_2024.csv")
    print("  results/schedule_evaluation_2025.csv")
    print("  results/schedule_comparison_2025.png")

    print("\nSUNPILOT SCHEDULER COMPLETE")


if __name__ == "__main__":
    main()


