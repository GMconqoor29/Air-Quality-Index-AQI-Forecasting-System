"""Empirical PM2.5 / PM10 -> AQI mapping for reference ground-truth calculation.

Implements the exact formula described in `Tìm hiểu thêm.pdf` (Section 7.1.b)
and `build_aqi_mapping.py`:
    AQI = max( f(PM2.5), AQI_PM10 )
where:
- f(PM2.5) is the piecewise-linear mapping with a cliff at ~151 ug/m3 fitted on
  the training timeline and stored in `results/pm25_to_aqi_mapping.json`.
- AQI_PM10 is the standard US EPA PM10 breakpoint table (integer truncation).
"""

import json
import os
import numpy as np

MAPPING_PATH = os.path.join("results", "pm25_to_aqi_mapping.json")

PM10_TABLE = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

_CACHED_MAPPING = None


def load_pm25_mapping(mapping_path: str = MAPPING_PATH) -> dict:
    """Load and cache the empirical PM2.5 -> AQI mapping from JSON."""
    global _CACHED_MAPPING
    if _CACHED_MAPPING is not None and mapping_path == MAPPING_PATH:
        return _CACHED_MAPPING

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Mapping file '{mapping_path}' not found. Run `python build_aqi_mapping.py` first."
        )
    with open(mapping_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    mapping = {
        "cliff": raw.get("cliff"),
        "j": raw.get("j"),
        "slope": float(raw["slope"]),
        "intercept": float(raw["intercept"]),
        "xs": np.asarray(raw["table"]["x"], dtype=np.float64),
        "ys": np.asarray(raw["table"]["y"], dtype=np.float64),
    }
    if mapping_path == MAPPING_PATH:
        _CACHED_MAPPING = mapping
    return mapping


def pm25_to_aqi_array(pm25_values, mapping: dict | None = None) -> np.ndarray:
    """Apply f(PM2.5) empirical mapping to an array of PM2.5 concentrations."""
    if mapping is None:
        mapping = load_pm25_mapping()

    x = np.asarray(pm25_values, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(x) & (x >= 0)
    if not ok.any():
        return out[0] if scalar_input else out

    xs, ys = mapping["xs"], mapping["ys"]
    j = mapping["j"]
    cliff = mapping["cliff"]

    if j is None or cliff is None:
        out[ok] = np.interp(x[ok], xs, ys)
    else:
        lo = ok & (x < cliff)
        hi = ok & (x >= cliff)
        if lo.any():
            out[lo] = np.interp(x[lo], xs[: j + 1], ys[: j + 1])
        if hi.any():
            out[hi] = np.interp(x[hi], xs[j + 1 :], ys[j + 1 :])

    beyond = ok & (x > xs[-1])
    if beyond.any():
        out[beyond] = ys[-1] + mapping["slope"] * (x[beyond] - xs[-1])

    out = np.clip(np.floor(out + 0.5), 0.0, 500.0)
    return out[0] if scalar_input else out


def pm10_to_aqi_array(pm10_values) -> np.ndarray:
    """Compute US EPA PM10 sub-index (integer truncation)."""
    x = np.asarray(pm10_values, dtype=np.float64)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(x) & (x >= 0)
    if not ok.any():
        return out[0] if scalar_input else out

    c = np.floor(x)
    for c_lo, c_hi, i_lo, i_hi in PM10_TABLE:
        mask = ok & (c >= c_lo) & (c <= c_hi)
        if mask.any():
            out[mask] = (i_hi - i_lo) / (c_hi - c_lo) * (c[mask] - c_lo) + i_lo

    above = ok & (c > PM10_TABLE[-1][1])
    if above.any():
        out[above] = 500.0

    out = np.clip(np.floor(out + 0.5), 0.0, 500.0)
    return out[0] if scalar_input else out


def compute_aqi_from_pm(
    pm25: float | None,
    pm10: float | None = None,
    mapping: dict | None = None,
) -> int | None:
    """Compute AQI = max(f(PM2.5), AQI_PM10) for a single observation."""
    aqi_pm25 = np.nan
    aqi_pm10 = np.nan

    if pm25 is not None and np.isfinite(pm25) and pm25 >= 0:
        aqi_pm25 = float(pm25_to_aqi_array(float(pm25), mapping=mapping))
    if pm10 is not None and np.isfinite(pm10) and pm10 >= 0:
        aqi_pm10 = float(pm10_to_aqi_array(float(pm10)))

    if np.isfinite(aqi_pm25) and np.isfinite(aqi_pm10):
        return int(round(max(aqi_pm25, aqi_pm10)))
    if np.isfinite(aqi_pm25):
        return int(round(aqi_pm25))
    if np.isfinite(aqi_pm10):
        return int(round(aqi_pm10))
    return None

