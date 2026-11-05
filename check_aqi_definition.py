"""Compare the dataset's AQI column with AQI recomputed from PM25 / PM10 (US EPA).

Breakpoints and truncation rules follow EPA's Technical Assistance Document for
the Reporting of Daily AQI (Table 6, Equation 1). Variants tested:
  - hourly concentration (not an official AQI, but what hourly datasets often do)
  - trailing 24 h mean (the official averaging time for PM)
  - NowCast (EPA's real-time method: weighted 12 h mean)

Run from the project root (next to Dataset/):
    python check_aqi_definition.py
Outputs a printed report and results/aqi_definition_check.csv.
"""
import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "Dataset/archive"
TIME_COL, AQI_COL, PM25_COL, PM10_COL = "Local Time", "AQI", "PM25", "PM10"
OUT_PATH = os.path.join("results", "aqi_definition_check.csv")

# (conc_low, conc_high, aqi_low, aqi_high)
PM25_2024 = [
    (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500),
]
PM25_PRE2024 = [  # older table, kept only for comparison
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]
PM10_TABLE = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500),
]
CATEGORY_EDGES = [50, 100, 150, 200, 300]


def truncate(values, decimals):
    factor = 10 ** decimals
    # round() first so 35.4 * 10 = 353.99999... does not floor to 353
    return np.floor(np.round(np.asarray(values, dtype=np.float64) * factor, 6)) / factor


def conc_to_aqi(conc, table, decimals):
    c = truncate(conc, decimals)
    out = np.full(c.shape, np.nan)
    for c_lo, c_hi, i_lo, i_hi in table:
        mask = (c >= c_lo) & (c <= c_hi)
        out[mask] = (i_hi - i_lo) / (c_hi - c_lo) * (c[mask] - c_lo) + i_lo
    above = c > table[-1][1]  # EPA: extend the top segment's slope; here capped at 500
    out[above] = table[-1][3]
    return np.floor(out + 0.5)


def nowcast(series, window=12):
    """EPA PM NowCast: weight factor = 1 - range/max, floored at 0.5; needs 2 of last 3 hours."""
    x = series.to_numpy(dtype=np.float64)
    out = np.full(x.shape, np.nan)
    for i in range(window - 1, len(x)):
        w = x[i - window + 1 : i + 1][::-1]  # w[0] = current hour, w[k] = k hours ago
        valid = np.isfinite(w)
        if np.isfinite(w[:3]).sum() < 2:
            continue
        cmax, cmin = np.nanmax(w), np.nanmin(w)
        if cmax <= 0:
            out[i] = 0.0
            continue
        factor = max(1 - (cmax - cmin) / cmax, 0.5)
        weights = factor ** np.arange(window)
        out[i] = np.sum(w[valid] * weights[valid]) / np.sum(weights[valid])
    return pd.Series(out, index=series.index)


def trailing_24h(series):
    return series.rolling(24, min_periods=18).mean()


def category(aqi):
    return np.digitize(aqi, CATEGORY_EDGES, right=True)


def load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {DATA_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL).set_index(TIME_COL).asfreq("h")
    return df[[AQI_COL, PM25_COL, PM10_COL]]


def report(name, ref, calc):
    valid = np.isfinite(ref) & np.isfinite(calc)
    r, c = ref[valid], calc[valid]
    diff = c - r
    print(f"\n[{name}]  n={valid.sum()}")
    print(f"  MAE={np.mean(np.abs(diff)):.2f}  RMSE={np.sqrt(np.mean(diff**2)):.2f}  "
          f"bias(calc-ref)={np.mean(diff):.2f}  Pearson r={np.corrcoef(r, c)[0, 1]:.4f}")
    print(f"  |diff|<=2: {np.mean(np.abs(diff) <= 2):.1%}   |diff|<=5: {np.mean(np.abs(diff) <= 5):.1%}")
    print(f"  same AQI category: {np.mean(category(r) == category(c)):.1%}")
    print(f"  dataset AQI > computed by >5: {np.mean(r - c > 5):.1%}   "
          f"dataset AQI < computed by >5: {np.mean(c - r > 5):.1%}")
    hi = r >= 200
    if hi.any():
        print(f"  peaks (ref>=200): {hi.sum()}, computed also >=200: {np.mean(c[hi] >= 200):.1%}")


def main():
    df = load()
    pm25, pm10 = df[PM25_COL], df[PM10_COL]
    print(f"Rows: {len(df)}  AQI missing: {df[AQI_COL].isna().mean():.1%}  "
          f"PM25 missing: {pm25.isna().mean():.1%}  PM10 missing: {pm10.isna().mean():.1%}")
    print(f"AQI > 400 rows (the training pipeline masks these): {(df[AQI_COL] > 400).sum()}")

    ref = df[AQI_COL].to_numpy(dtype=np.float64)
    pm25_versions = {"hourly": pm25, "24h_mean": trailing_24h(pm25), "nowcast": nowcast(pm25)}
    pm10_versions = {"hourly": pm10, "24h_mean": trailing_24h(pm10), "nowcast": nowcast(pm10)}

    variants = {}
    for table_name, table in (("PM25_2024", PM25_2024), ("PM25_pre2024", PM25_PRE2024)):
        for ver, conc in pm25_versions.items():
            variants[f"{table_name}|{ver}"] = conc_to_aqi(conc.to_numpy(), table, 1)
    for ver in pm25_versions:  # AQI = max sub-index, as the EPA defines it
        a25 = conc_to_aqi(pm25_versions[ver].to_numpy(), PM25_2024, 1)
        a10 = conc_to_aqi(pm10_versions[ver].to_numpy(), PM10_TABLE, 0)
        variants[f"max(PM25_2024,PM10)|{ver}"] = np.fmax(a25, a10)

    result = df.copy()
    for name, calc in variants.items():
        result[name] = calc
        report(name, ref, calc)

    print("\nLagged correlation (dataset AQI vs computed AQI shifted by k hours):")
    for name in ("PM25_2024|hourly", "PM25_2024|nowcast"):
        calc = pd.Series(variants[name], index=df.index)
        line = []
        for k in range(-6, 7):
            shifted = calc.shift(k)
            valid = df[AQI_COL].notna() & shifted.notna()
            line.append(f"{k:+d}h:{np.corrcoef(df[AQI_COL][valid], shifted[valid])[0, 1]:.3f}")
        print(f"  {name}: " + "  ".join(line))

    os.makedirs("results", exist_ok=True)
    result.to_csv(OUT_PATH)
    print(f"\nSaved per-hour comparison to {OUT_PATH}")


if __name__ == "__main__":
    main()
