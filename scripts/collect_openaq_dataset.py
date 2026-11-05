"""Collect, clean (EDA), and save the OpenAQ (Trạm 4946813 - ĐHBK) dataset to CSV & SQLite.

Follows the dataset collection & EDA pipeline in `FAIR2023_paper_3434.pdf` and
`[Slide] Thesis - AQI.pdf` (Slide 24):
  1. Fetch hourly PM2.5 & PM10 from OpenAQ v3 (Location 4946813: Số 1 Giải Phóng - ĐHBK)
     and hourly meteorological features (Temperature, Relative Humidity, Pressure).
  2. Compute reference AQI = max(f(PM2.5), AQI_PM10).
  3. Apply 3-step EDA cleaning:
     - Reform and equidistant timestamps (`asfreq('h')`)
     - Remove outlier data (`AQI > 400`, invalid negative readings)
     - Handle missing data (causal forward-fill <= 1h)
  4. Persist to:
     - SQLite (`results/realtime_aqi.db`)
     - Master CSV (`Dataset/openaq_dhbk_collected.csv`)
     - Daily CSV snapshots (`Dataset/openaq_daily/YYYY-MM-DD.csv`)

Usage:
    .venv/bin/python scripts/collect_openaq_dataset.py --hours 168
"""

import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.serving.ingestor import RealtimeDataStore


def main():
    parser = argparse.ArgumentParser(
        description="Collect & clean OpenAQ (Location 4946813 - ĐHBK) dataset to CSV and SQLite."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=168,
        help="Number of past hours to synchronize from OpenAQ & Open-Meteo (default: 168)",
    )
    args = parser.parse_args()

    store = RealtimeDataStore()
    print(f"Synchronizing last {args.hours} hours from OpenAQ (Location {store.openaq_location_id})...")
    sync_summary = store.sync_openaq_and_weather(hours=args.hours)
    csv_info = store.export_collected_csv()

    print("\n" + "=" * 72)
    print(" OPENAQ DATASET COLLECTION & EDA SUMMARY (Slide 24 / FAIR2023 Pipeline)")
    print("=" * 72)
    print(f"  Station         : {sync_summary['openaq_station']} (ID: {sync_summary['openaq_location_id']})")
    print(f"  OpenAQ Fetched  : {sync_summary['openaq_records_fetched']} hourly records")
    print(f"  Weather Fetched : {sync_summary['weather_records_fetched']} hourly records")
    print(f"  Total Cleaned   : {csv_info['saved_rows']} equidistant hourly rows saved")
    print(f"  Master CSV      : {csv_info['master_csv']}")
    if "daily_csv" in csv_info:
        print(f"  Latest Daily CSV: {csv_info['daily_csv']}")
    print("=" * 72)

    if os.path.exists(csv_info["master_csv"]):
        df = pd.read_csv(csv_info["master_csv"])
        print("\nPreview of saved dataset (`Dataset/openaq_dhbk_collected.csv`):")
        print(df.tail(8).to_string(index=False))

        print("\nDescriptive Statistics (min / max / mean):")
        stats_cols = ["AQI", "PM25", "PM10", "Temperature", "Relative Humidity", "Pressure"]
        stats = df[stats_cols].agg(["min", "max", "mean"]).T.round(2)
        print(stats.to_string())


if __name__ == "__main__":
    main()

