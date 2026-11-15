import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/klippan_hourly_2022_2025.csv")
MODEL_PATH = Path("models/sunpilot_v1.joblib")
RESULTS_DIR = Path("results")

FEATURES = [
    "hour_sin",
    "hour_cos",
    "season_sin",
    "season_cos",
    "radiation_lag_24",
    "radiation_lag_48",
    "radiation_lag_168",
    "cloud_lag_24",
    "temperature_lag_24",
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["time_utc"] = pd.to_datetime(
        df["time_utc"], utc=True
    )
    df = df.sort_values("time_utc").reset_index(drop=True)

    # Local calendar features
    local_time = df["time_utc"].dt.tz_convert(
        "Europe/Stockholm"
    )

    df["local_date"] = local_time.dt.strftime("%Y-%m-%d")
    df["local_hour"] = local_time.dt.hour

    hour = local_time.dt.hour
    day = local_time.dt.dayofyear

    # Cyclical time features
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["season_sin"] = np.sin(
        2 * np.pi * day / 365.25
    )
    df["season_cos"] = np.cos(
        2 * np.pi * day / 365.25
    )

    # Historical information only.
    # For a target hour tomorrow, lag_24 is from
    # the same UTC hour on the preceding day.
    df["radiation_lag_24"] = (
        df["shortwave_radiation"].shift(24)
    )
    df["radiation_lag_48"] = (
        df["shortwave_radiation"].shift(48)
    )
    df["radiation_lag_168"] = (
        df["shortwave_radiation"].shift(168)
    )

    df["cloud_lag_24"] = df["cloud_cover"].shift(24)
    df["temperature_lag_24"] = (
        df["temperature_2m"].shift(24)
    )

    df = df.dropna(
        subset=FEATURES + ["shortwave_radiation"]
    ).reset_index(drop=True)

    return df


def evaluate(name, df, predictions):
    actual = df["shortwave_radiation"].to_numpy()

    predictions = np.maximum(
        np.asarray(predictions), 0
    )

    # Overall prediction error
    mae_all = mean_absolute_error(actual, predictions)

    # Operational window: local 08:00–18:00
    mask = (
        (df["local_hour"] >= 8)
        & (df["local_hour"] < 18)
    ).to_numpy()

    mae_window = mean_absolute_error(
        actual[mask], predictions[mask]
    )

    print(
        f"{name:<22} "
        f"MAE all: {mae_all:8.2f} | "
        f"MAE 08-18: {mae_window:8.2f}"
    )

    return {
        "mae_all_w_m2": round(float(mae_all), 3),
        "mae_08_18_w_m2": round(float(mae_window), 3),
    }


def save_predictions(df, ai_predictions, filename):
    output = df[
        [
            "time_utc",
            "local_date",
            "local_hour",
            "shortwave_radiation",
            "radiation_lag_24",
        ]
    ].copy()

    output = output.rename(
        columns={
            "shortwave_radiation": "actual",
            "radiation_lag_24": "baseline",
        }
    )

    output["ai_prediction"] = np.maximum(
        ai_predictions, 0
    )

    output.to_csv(
        RESULTS_DIR / filename,
        index=False,
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== SUNPILOT AI / TRAINING ===")

    df = load_data()

    years = df["time_utc"].dt.year

    train = df[years.isin([2022, 2023])].copy()
    validation = df[years == 2024].copy()
    test = df[years == 2025].copy()

    print(f"Training samples:   {len(train):,}")
    print(f"Validation samples: {len(validation):,}")
    print(f"Test samples:       {len(test):,}")

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One dataset split is empty")

    # Baseline: same UTC hour yesterday.
    baseline_val = validation[
        "radiation_lag_24"
    ].to_numpy()

    baseline_test = test[
        "radiation_lag_24"
    ].to_numpy()

    print("\nTraining machine learning model...")

    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.07,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        train[FEATURES],
        train["shortwave_radiation"],
    )

    ai_val = np.maximum(
        model.predict(validation[FEATURES]), 0
    )

    ai_test = np.maximum(
        model.predict(test[FEATURES]), 0
    )

    print("\n=== VALIDATION / 2024 ===")

    metrics = {"validation_2024": {}, "test_2025": {}}

    metrics["validation_2024"]["baseline"] = evaluate(
        "Yesterday baseline",
        validation,
        baseline_val,
    )

    metrics["validation_2024"]["ai"] = evaluate(
        "SunPilot AI",
        validation,
        ai_val,
    )

    print("\n=== UNSEEN TEST / 2025 ===")

    metrics["test_2025"]["baseline"] = evaluate(
        "Yesterday baseline",
        test,
        baseline_test,
    )

    metrics["test_2025"]["ai"] = evaluate(
        "SunPilot AI",
        test,
        ai_test,
    )

    # Save trained model and feature specification
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "training_years": [2022, 2023],
        },
        MODEL_PATH,
    )

    save_predictions(
        validation,
        ai_val,
        "validation_predictions_2024.csv",
    )

    save_predictions(
        test,
        ai_test,
        "test_predictions_2025.csv",
    )

    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved trained model:", MODEL_PATH)
    print("Saved metrics: results/metrics.json")
    print("Saved prediction CSV files")
    print("\nSUNPILOT TRAINING COMPLETE")


if __name__ == "__main__":
    main()
