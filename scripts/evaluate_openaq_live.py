"""Evaluate trained models directly against live OpenAQ (Trạm 4946813 - ĐHBK) ground truth.

Generates:
  1. Printed leaderboard of empirical MAE & RMSE on the live OpenAQ window
  2. `results/openaq_live_verification.csv` (hourly Actual OpenAQ vs. Model Predictions)
  3. `results/openaq_live_verification.png` (overlay chart: Actual OpenAQ vs. Predicted AQI)

Usage:
    .venv/bin/python scripts/evaluate_openaq_live.py
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.serving.ingestor import RealtimeDataStore
from src.serving.predictor import RealtimePredictor


def main():
    store = RealtimeDataStore()
    store.ensure_warm_start(hours=240)
    predictor = RealtimePredictor()

    history = store.get_historical_readings(hours=240)
    # Count how many trailing hours are real OpenAQ / sensor observations
    openaq_indices = [i for i, r in enumerate(history) if r.get("source") in ("openaq", "esp32_sensor")]
    eval_hours = max(48, len(openaq_indices)) if openaq_indices else 72
    eval_slice_len = min(len(history) - 49, max(48, eval_hours))

    model_names = [
        "XGBoost",
        "Tuned XGBoost",
        "Stacked LSTM",
        "GRU",
        "Base LSTM",
        "Soft-Gated Ensemble",
        "CNN-LSTM Attention",
        "CNN-GRU Attention",
        "Two-Stage Regressor",
    ]

    tail_records = history[-eval_slice_len:]
    timestamps = [pd.to_datetime(r["timestamp"]).strftime("%m-%d %H:%M") for r in tail_records]
    full_timestamps = [r["timestamp"] for r in tail_records]
    y_actual = np.array([float(r["aqi"]) for r in tail_records], dtype=np.float64)
    sources = [r["source"] for r in tail_records]

    # Persistence baseline: y_hat(t) = y(t-1)
    y_persist = np.array(
        [float(history[len(history) - eval_slice_len - 1 + i]["aqi"]) for i in range(eval_slice_len)],
        dtype=np.float64,
    )

    preds_by_model = {"Persistence": y_persist}
    for m_name in model_names:
        enriched = predictor.attach_rolling_predictions(history, horizon=1, model_name=m_name)
        preds_by_model[m_name] = np.array(
            [float(r["predictedAqi"]) for r in enriched[-eval_slice_len:]],
            dtype=np.float64,
        )

    # Filter to purely OpenAQ / ESP32 verified hours if available
    live_mask = np.array([s in ("openaq", "esp32_sensor") for s in sources], dtype=bool)
    if not live_mask.any():
        live_mask = np.ones_like(y_actual, dtype=bool)

    y_true_live = y_actual[live_mask]
    n_live = int(live_mask.sum())

    print("\n" + "=" * 84)
    print(f" LIVE EMPIRICAL VERIFICATION VS. OPENAQ (Trạm 4946813 - Số 1 Giải Phóng, ĐHBK)")
    print(f" Evaluated Window: {n_live} verified hourly observations ({full_timestamps[np.argmax(live_mask)]} -> {full_timestamps[-1]})")
    print("=" * 84)
    print(f"{'Model Architecture':<24} | {'Live MAE (AQI)':<16} | {'Live RMSE (AQI)':<16} | {'Mean Bias':<12}")
    print("-" * 84)

    summary_rows = []
    for m_name, y_pred_all in preds_by_model.items():
        y_p = y_pred_all[live_mask]
        mae = float(mean_absolute_error(y_true_live, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_true_live, y_p)))
        bias = float(np.mean(y_p - y_true_live))
        summary_rows.append({"model": m_name, "live_mae": round(mae, 2), "live_rmse": round(rmse, 2), "bias": round(bias, 2)})
        print(f"{m_name:<24} | {mae:<16.2f} | {rmse:<16.2f} | {bias:+12.2f}")
    print("=" * 84)

    # Save hourly verification CSV
    df_out = pd.DataFrame(
        {
            "timestamp": np.array(full_timestamps)[live_mask],
            "source": np.array(sources)[live_mask],
            "actual_aqi_openaq": y_true_live.astype(int),
        }
    )
    for m_name, y_pred_all in preds_by_model.items():
        col_key = "pred_" + m_name.lower().replace(" ", "_").replace("-", "_")
        df_out[col_key] = y_pred_all[live_mask].astype(int)

    os.makedirs("results", exist_ok=True)
    csv_out = os.path.join("results", "openaq_live_verification.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved hourly verification table to: {csv_out}")

    # Plot Actual OpenAQ vs Predicted AQI
    ts_live = np.array(timestamps)[live_mask]
    x_idx = np.arange(n_live)

    plt.figure(figsize=(14, 6), dpi=220)
    plt.plot(
        x_idx,
        y_true_live,
        label="Actual AQI (OpenAQ - Trạm ĐHBK)",
        color="#0f766e",
        linewidth=2.6,
        marker="o",
        markersize=3.5,
        zorder=5,
    )
    plt.plot(
        x_idx,
        preds_by_model["XGBoost"][live_mask],
        label=f"XGBoost Forecast (t+1) — MAE={mean_absolute_error(y_true_live, preds_by_model['XGBoost'][live_mask]):.2f}",
        color="#d97706",
        linestyle="--",
        linewidth=2.1,
        zorder=4,
    )
    plt.plot(
        x_idx,
        preds_by_model["Tuned XGBoost"][live_mask],
        label=f"Tuned XGBoost (t+1) — MAE={mean_absolute_error(y_true_live, preds_by_model['Tuned XGBoost'][live_mask]):.2f}",
        color="#2563eb",
        linestyle="-.",
        linewidth=1.8,
        alpha=0.85,
    )
    plt.plot(
        x_idx,
        preds_by_model["Soft-Gated Ensemble"][live_mask],
        label=f"Soft-Gated Ensemble (t+1) — MAE={mean_absolute_error(y_true_live, preds_by_model['Soft-Gated Ensemble'][live_mask]):.2f}",
        color="#e11d48",
        linestyle=":",
        linewidth=1.9,
        alpha=0.85,
    )

    step = max(1, n_live // 10)
    plt.xticks(x_idx[::step], ts_live[::step], rotation=30, ha="right", fontsize=9)
    plt.title(
        "Real-Time AQI Forecast (t+1h) vs. Actual OpenAQ Reference (Số 1 Giải Phóng - ĐHBK)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Timestamp (Local Time - Hanoi)", fontsize=11)
    plt.ylabel("AQI (Empirical Weatherbit/EPA Scale)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(loc="upper left", framealpha=0.92, fontsize=9.5)
    plt.tight_layout()

    png_out = os.path.join("results", "openaq_live_verification.png")
    plt.savefig(png_out)
    plt.close()
    print(f"Saved verification chart to: {png_out}")


if __name__ == "__main__":
    main()

