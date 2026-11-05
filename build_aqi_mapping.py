"""Build an empirical PM2.5 -> AQI mapping from the dataset, measure how well it
reproduces the AQI column, and check whether the mapping is stable across years.

Run from the project root, next to check_aqi_definition.py and infer_aqi_breakpoints.py:
    python build_aqi_mapping.py

Outputs: printed report and results/pm25_to_aqi_mapping.json (reusable mapping).
"""
import json
import os

import numpy as np
import pandas as pd

from check_aqi_definition import (
    AQI_COL, PM25_COL, PM10_COL, PM10_TABLE, PM25_2024, PM25_PRE2024,
    category, conc_to_aqi,
)
from infer_aqi_breakpoints import MIN_BIN_COUNT, load

TRAIN_FRACTION = 0.7          # chronological split, same ratio as config.TRAIN_RATIO
CLIFF_DROP = -30.0            # a drop larger than this between adjacent bins = discontinuity
OUT_JSON = os.path.join("results", "pm25_to_aqi_mapping.json")
BINS = np.concatenate([np.arange(0, 150, 2.5), np.arange(150, 405, 5.0)])
YEAR_WINDOWS = [(8, 10), (13, 15), (23, 25), (35, 37), (55, 60), (70, 75),
                (90, 95), (120, 125), (145, 150), (150, 155), (190, 200)]


def fit_mapping(pm25, aqi, min_count=MIN_BIN_COUNT):
    d = pd.DataFrame({"x": np.asarray(pm25, float), "y": np.asarray(aqi, float)}).dropna()
    d = d[d["y"] <= 500]
    d["bin"] = pd.cut(d["x"], BINS)
    table = d.groupby("bin", observed=True).agg(
        n=("y", "size"), x=("x", "mean"), y=("y", "median")
    )
    table = table[table["n"] >= min_count].reset_index(drop=True)
    if len(table) < 5:
        raise ValueError("Not enough populated PM2.5 bins to fit a mapping.")
    step = table["y"].diff().shift(-1)  # step[j] = y[j+1] - y[j]
    j = int(step.idxmin()) if step.min() < CLIFF_DROP else None
    cliff = float((table["x"][j] + table["x"][j + 1]) / 2) if j is not None else None
    tail = table.tail(5)
    slope, intercept = np.polyfit(tail["x"], tail["y"], 1)
    return {"table": table, "j": j, "cliff": cliff, "slope": float(slope),
            "intercept": float(intercept)}


def apply_mapping(mapping, pm25):
    x = np.asarray(pm25, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    xs, ys = mapping["table"]["x"].to_numpy(), mapping["table"]["y"].to_numpy()
    j = mapping["j"]
    if j is None:
        out[ok] = np.interp(x[ok], xs, ys)
    else:  # interpolate on each side of the cliff separately
        lo = ok & (x < mapping["cliff"])
        hi = ok & (x >= mapping["cliff"])
        out[lo] = np.interp(x[lo], xs[: j + 1], ys[: j + 1])
        out[hi] = np.interp(x[hi], xs[j + 1 :], ys[j + 1 :])
    beyond = ok & (x > xs[-1])  # sparse region: linear extrapolation, treat with caution
    out[beyond] = ys[-1] + mapping["slope"] * (x[beyond] - xs[-1])
    return np.floor(out + 0.5)


def metrics(name, ref, pred):
    valid = np.isfinite(ref) & np.isfinite(pred)
    r, p = ref[valid], pred[valid]
    diff = p - r
    peak_ref, peak_pred = r >= 200, p >= 200
    tp = int(np.sum(peak_ref & peak_pred))
    recall = tp / peak_ref.sum() if peak_ref.any() else float("nan")
    precision = tp / peak_pred.sum() if peak_pred.any() else float("nan")
    print(f"  {name:28s} n={valid.sum():6d} MAE={np.mean(np.abs(diff)):6.2f} "
          f"RMSE={np.sqrt(np.mean(diff ** 2)):6.2f} bias={np.mean(diff):6.2f} "
          f"r={np.corrcoef(r, p)[0, 1]:.4f} |d|<=5:{np.mean(np.abs(diff) <= 5):6.1%} "
          f"same-cat:{np.mean(category(r) == category(p)):6.1%} "
          f"peak(>=200) recall={recall:.3f} precision={precision:.3f}")


def evaluate(title, frame, mapping):
    ref = frame[AQI_COL].to_numpy(dtype=np.float64)
    pm25 = frame[PM25_COL].to_numpy(dtype=np.float64)
    pm10_idx = conc_to_aqi(frame[PM10_COL].to_numpy(), PM10_TABLE, 0)
    mapped = apply_mapping(mapping, pm25)
    print(f"\n{title}")
    metrics("empirical mapping (PM25)", ref, mapped)
    metrics("empirical mapping + PM10", ref, np.fmax(mapped, pm10_idx))
    metrics("EPA 2024 table (PM25)", ref, conc_to_aqi(pm25, PM25_2024, 1))
    metrics("EPA pre-2024 table (PM25)", ref, conc_to_aqi(pm25, PM25_PRE2024, 1))


def per_year_check(df):
    print("\nMedian dataset AQI inside PM2.5 windows, per year (n in brackets):")
    header = "  year | " + " | ".join(f"{lo}-{hi}" for lo, hi in YEAR_WINDOWS)
    print(header)
    d = df.dropna(subset=[AQI_COL, PM25_COL])
    for year, g in d.groupby(d.index.year):
        cells = []
        for lo, hi in YEAR_WINDOWS:
            sel = g[(g[PM25_COL] > lo) & (g[PM25_COL] <= hi)][AQI_COL]
            cells.append(f"{sel.median():.0f}({len(sel)})" if len(sel) >= 5 else "n/a")
        print(f"  {year} | " + " | ".join(cells))
    print("\nCliff location detected per year:")
    for year, g in d.groupby(d.index.year):
        try:
            m = fit_mapping(g[PM25_COL], g[AQI_COL], min_count=10)
            where = f"{m['cliff']:.1f} ug/m3" if m["cliff"] is not None else "none detected"
        except ValueError:
            where = "too little data"
        print(f"  {year}: {where}")


def main():
    df = load().dropna(subset=[AQI_COL, PM25_COL, PM10_COL], how="all")
    cut = int(len(df) * TRAIN_FRACTION)
    train, test = df.iloc[:cut], df.iloc[cut:]

    fit = fit_mapping(train[PM25_COL], train[AQI_COL])
    cliff = f"{fit['cliff']:.1f} ug/m3" if fit["cliff"] is not None else "none"
    print(f"Mapping fitted on the first {TRAIN_FRACTION:.0%} of the timeline "
          f"({train.index[0]} to {train.index[-1]}); cliff at {cliff}; "
          f"tail slope beyond last bin = {fit['slope']:.2f} AQI per ug/m3")
    evaluate("In-sample (training period)", train, fit)
    evaluate("Out-of-sample (later 30% of the timeline)", test, fit)
    print("\nThe out-of-sample rows are the honest estimate of the definition-mismatch floor.")

    per_year_check(df)

    final = fit_mapping(df[PM25_COL], df[AQI_COL])
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({
            "cliff": final["cliff"], "j": final["j"],
            "slope": final["slope"], "intercept": final["intercept"],
            "table": final["table"].to_dict(orient="list"),
        }, fh, indent=2)
    print(f"\nSaved mapping fitted on all data to {OUT_JSON}")


if __name__ == "__main__":
    main()
