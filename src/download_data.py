from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
import requests

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT = Path("data/klippan_hourly_2022_2025.csv")

PARAMS = {
    "latitude": 56.1356,
    "longitude": 13.1309,
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
    "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
    "models": "era5",
    "timezone": "UTC",
}

FIELDS = [
    "shortwave_radiation",
    "cloud_cover",
    "temperature_2m",
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("=== SUNPILOT AI / DATA DOWNLOAD ===")
    print("Location: Klippan, Sweden")
    print("Period: 2022-01-01 to 2025-12-31")
    print("Source: Open-Meteo / ERA5")
    print("Downloading...")

    response = requests.get(
        BASE_URL,
        params=PARAMS,
        timeout=(15, 120),
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "API error"))

    hourly = payload.get("hourly", {})
    required = ["time", *FIELDS]

    for name in required:
        if name not in hourly:
            raise ValueError(f"Missing API field: {name}")

    df = pd.DataFrame({
        "time_utc": hourly["time"],
        **{name: hourly[name] for name in FIELDS},
    })

    df["time_utc"] = pd.to_datetime(
        df["time_utc"],
        utc=True,
    )

    df = df.sort_values("time_utc").reset_index(drop=True)

    if df["time_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    expected = pd.date_range(
        "2022-01-01",
        "2025-12-31 23:00",
        freq="h",
        tz="UTC",
    )

    if not pd.DatetimeIndex(df["time_utc"]).equals(expected):
        raise ValueError(
            "Incomplete or irregular hourly time series"
        )

    df.to_csv(OUTPUT, index=False)

    metadata = {
        "source": "Open-Meteo Historical Weather API",
        "model": "ERA5",
        "requested_coordinates": {
            "latitude": PARAMS["latitude"],
            "longitude": PARAMS["longitude"],
        },
        "retrieved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_url": response.url,
        "hourly_units": payload.get("hourly_units", {}),
        "rows": len(df),
    }

    Path("data/source_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"First timestamp: {df['time_utc'].min()}")
    print(f"Last timestamp: {df['time_utc'].max()}")

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nRows per year:")
    print(df.groupby(df["time_utc"].dt.year).size().to_string())

    print("\nRadiation statistics (W/m2):")
    print(df["shortwave_radiation"].describe().round(2))

    print(f"\nSaved: {OUTPUT}")
    print("SUNPILOT DATA DOWNLOAD COMPLETE")


if __name__ == "__main__":
    main()
