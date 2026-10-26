"""
Plot Untuned XGBoost – Predicted vs Actual AQI at t+1.

Loads best_xgb_model.json from results/no AQI - t+1,
runs inference on the test split and saves a standalone figure.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Match the exact environment used when training the t+1 untuned model
os.environ["HORIZON"] = "1"
os.environ["RESULTS_DIR"] = "results/no AQI - t+1"
os.environ["FEATURE_COLS"] = "PM25,PM10,Temperature,Relative Humidity,Pressure,hour,month"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import src.config as config
from src.data_loader import get_xgboost_data
from src.evaluate import inverse_scale_predictions

RESULTS_DIR = config.RESULTS_DIR
MODEL_PATH  = os.path.join(RESULTS_DIR, "best_xgb_model.json")


def _raw_aqi(scaler, values):
    target_idx = (
        config.FEATURE_COLS.index(config.TARGET_COL)
        if config.TARGET_COL in config.FEATURE_COLS
        else 0
    )
    return inverse_scale_predictions(scaler, np.asarray(values).reshape(-1, 1), target_idx)


def main():
    print(f"HORIZON = {config.HORIZON}  (should be 1)")
    print(f"FEATURE_COLS = {config.FEATURE_COLS}")

    # ── load data ─────────────────────────────────────────────────────────────
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        scaler,
        train_timestamps, val_timestamps, test_timestamps,
    ) = get_xgboost_data(data_dir="Dataset/archive", include_split_timestamps=True)

    # ── load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    print(f"Loading model: {MODEL_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    dtest    = xgb.DMatrix(X_test)
    preds_sc = model.predict(dtest)

    # ── inverse-scale ─────────────────────────────────────────────────────────
    actual    = _raw_aqi(scaler, y_test)
    predicted = _raw_aqi(scaler, preds_sc)

    # ── metrics ───────────────────────────────────────────────────────────────
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae  = float(mean_absolute_error(actual, predicted))
    print(f"Untuned XGBoost t+1  RMSE={rmse:.3f}  MAE={mae:.3f}")

    # ── figure ────────────────────────────────────────────────────────────────
    N = len(actual)
    x = np.arange(N)
    peak_mask = actual >= config.PEAK_AQI_THRESHOLD

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Shade high-AQI regions
    in_peak = False
    start_idx = 0
    for i in range(N):
        if peak_mask[i] and not in_peak:
            start_idx = i
            in_peak = True
        elif not peak_mask[i] and in_peak:
            ax.axvspan(start_idx - 0.5, i - 0.5, color="#ffcccc", alpha=0.35, zorder=0)
            in_peak = False
    if in_peak:
        ax.axvspan(start_idx - 0.5, N - 0.5, color="#ffcccc", alpha=0.35, zorder=0)

    # Peak threshold line
    ax.axhline(
        config.PEAK_AQI_THRESHOLD, color="#cc0000", linewidth=1.0,
        linestyle="--", alpha=0.6, zorder=1,
        label=f"Peak threshold ({config.PEAK_AQI_THRESHOLD:.0f})",
    )

    # Actual & predicted lines
    ax.plot(x, actual,    color="#1a6faf", linewidth=1.6, label="Actual AQI",                zorder=3)
    ax.plot(x, predicted, color="#e07b00", linewidth=1.4, label="Predicted AQI (Untuned XGBoost)",
            alpha=0.88, zorder=4)

    # Metrics box
    textstr = f"RMSE = {rmse:.2f}\nMAE  = {mae:.2f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#aaaaaa", alpha=0.85)
    ax.text(0.01, 0.97, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", bbox=props)

    # X-axis timestamps
    if test_timestamps is not None and len(test_timestamps) == N:
        import pandas as pd
        ts = pd.to_datetime(test_timestamps)
        tick_step = max(1, N // 12)
        tick_idx  = np.arange(0, N, tick_step)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(
            [ts[i].strftime("%Y-%m-%d\n%H:%M") for i in tick_idx],
            fontsize=7, rotation=30, ha="right",
        )
    else:
        ax.set_xlabel("Time Step", fontsize=11)

    ax.set_ylabel("AQI", fontsize=12)
    ax.set_title(
        "Untuned XGBoost — Predicted vs Actual AQI (t+1)\n"
        "Test Set  |  Default Hyperparameters",
        fontsize=13, fontweight="bold", pad=14,
    )

    peak_patch = mpatches.Patch(color="#ffcccc", alpha=0.7, label="High-AQI region (≥200)")
    handles, labels_legend = ax.get_legend_handles_labels()
    ax.legend(
        handles + [peak_patch], labels_legend + ["High-AQI region (≥200)"],
        loc="upper right", fontsize=9, framealpha=0.9,
    )

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-1, N)

    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "untuned_xgb_t1_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()

