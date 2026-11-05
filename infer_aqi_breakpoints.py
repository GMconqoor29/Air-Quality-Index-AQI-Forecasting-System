"""Infer which PM2.5 breakpoints the dataset's AQI follows, and what drives the
peaks that PM alone cannot explain.

Idea: AQI = max over pollutants, so AQI >= PM2.5 sub-index everywhere. The LOWER
envelope of AQI versus PM25 (10th percentile per PM25 bin) tracks the PM2.5
sub-index; where it crosses AQI 50/100/150/200/300 reveals the breakpoints used.

Run from the project root, after check_aqi_definition.py (it imports from it):
    python infer_aqi_breakpoints.py
"""
import glob
import os

import numpy as np
import pandas as pd

from check_aqi_definition import (
    DATA_DIR, TIME_COL, AQI_COL, PM25_COL, PM10_COL,
    PM10_TABLE, PM25_2024, PM25_PRE2024, conc_to_aqi,
)

GAS_COLS = ["CO", "NO2", "O3", "SO2"]
TARGETS = {50: (9.0, 12.0), 100: (35.4, 35.4), 150: (55.4, 55.4),
           200: (125.4, 150.4), 300: (225.4, 250.4)}  # AQI: (2024 bp, pre-2024 bp)
MIN_BIN_COUNT = 20
OUT_PATH = os.path.join("results", "aqi_breakpoint_inference.csv")


def load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL).set_index(TIME_COL).asfreq("h")
    cols = [c for c in [AQI_COL, PM25_COL, PM10_COL, *GAS_COLS] if c in df.columns]
    return df[cols]


def crossing(centers, values, target):
    """First PM25 level where the smoothed envelope reaches `target` (linear interpolation)."""
    for i in range(1, len(values)):
        if values[i - 1] < target <= values[i]:
            frac = (target - values[i - 1]) / (values[i] - values[i - 1])
            return centers[i - 1] + frac * (centers[i] - centers[i - 1])
    return np.nan


def main():
    df = load()
    d = df.dropna(subset=[AQI_COL, PM25_COL])
    d = d[d[AQI_COL] <= 500]

    bins = np.concatenate([np.arange(0, 150, 2.5), np.arange(150, 405, 5.0)])
    d = d.assign(bin=pd.cut(d[PM25_COL], bins))
    stats = d.groupby("bin", observed=True).agg(
        n=(AQI_COL, "size"),
        pm25_mean=(PM25_COL, "mean"),
        aqi_p10=(AQI_COL, lambda s: np.percentile(s, 10)),
        aqi_median=(AQI_COL, "median"),
    )
    stats = stats[stats["n"] >= MIN_BIN_COUNT].copy()
    stats["pred_2024"] = conc_to_aqi(stats["pm25_mean"].to_numpy(), PM25_2024, 1)
    stats["pred_pre2024"] = conc_to_aqi(stats["pm25_mean"].to_numpy(), PM25_PRE2024, 1)
    stats["aqi_p10_smooth"] = stats["aqi_p10"].rolling(3, center=True, min_periods=1).median()
    os.makedirs("results", exist_ok=True)
    stats.to_csv(OUT_PATH)

    centers = stats["pm25_mean"].to_numpy()
    envelope = stats["aqi_p10_smooth"].to_numpy()
    print("PM2.5 (ug/m3) at which the lower AQI envelope reaches each AQI level:")
    print("  AQI | observed | 2024 table | pre-2024 table")
    for aqi, (bp_new, bp_old) in TARGETS.items():
        obs = crossing(centers, envelope, aqi)
        print(f"  {aqi:>3} | {obs:8.1f} | {bp_new:10.1f} | {bp_old:14.1f}")
    print("  (the observed column should sit closer to one of the two tables)")

    # Peaks that PM2.5 + PM10 cannot explain
    a25 = conc_to_aqi(df[PM25_COL].to_numpy(), PM25_2024, 1)
    a10 = conc_to_aqi(df[PM10_COL].to_numpy(), PM10_TABLE, 0)
    pm_aqi = np.fmax(a25, a10)
    aqi = df[AQI_COL].to_numpy()
    peaks = aqi >= 200
    unexplained = peaks & (pm_aqi < 150)
    print(f"\nHours with AQI >= 200: {int(peaks.sum())}")
    print(f"  of which PM-based AQI < 150 (unexplained by PM): {int(unexplained.sum())} "
          f"({unexplained.sum() / max(peaks.sum(), 1):.1%})")

    gases = [c for c in GAS_COLS if c in df.columns]
    if gases and unexplained.any():
        ranks = df[gases].rank(pct=True)
        low = np.isfinite(aqi) & (aqi < 100)
        print("\nMean percentile rank of each gas (0 = lowest ever, 1 = highest ever):")
        print("  gas | unexplained peaks | hours with AQI < 100")
        for gas in gases:
            print(f"  {gas:>3} | {ranks[gas][unexplained].mean():17.2f} | {ranks[gas][low].mean():.2f}")
        print("  A gas far above 0.5 in the unexplained hours is a likely driver of those peaks.")
        print("  Check the units of these columns before converting them to sub-indices.")

    print(f"\nSaved per-bin table to {OUT_PATH}")


if __name__ == "__main__":
    main()
