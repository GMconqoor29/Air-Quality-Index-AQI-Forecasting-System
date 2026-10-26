import os
import re
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.data_loader import get_dataloaders, get_xgboost_data
from src.model import LSTMAQIModel, StackedLSTMAQIModel, GRUAQIModel, CNNRecurrentAttentionAQIModel
from src.hybrid_ensemble import SoftGatingEnsemble, make_gating_context, calculate_sample_weights
from src.evaluate import inverse_scale_predictions, calculate_peak_metrics
from src.train import evaluate_model

def load_pt_model(model_cls, pt_path, **kwargs):
    model = model_cls(input_size=len(config.FEATURE_COLS), **kwargs)
    state = torch.load(pt_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()
    return model

def calc_metrics(y_actual, preds_unscaled, peak_mask, scaler, target_idx):
    num_peaks = int(peak_mask.sum())
    rmse = float(np.sqrt(mean_squared_error(y_actual, preds_unscaled)))
    mae = float(mean_absolute_error(y_actual, preds_unscaled))
    p_recall = float((preds_unscaled[peak_mask] >= config.PEAK_AQI_THRESHOLD).sum() / num_peaks) if num_peaks > 0 else 0.0
    p_mae = float(mean_absolute_error(y_actual[peak_mask], preds_unscaled[peak_mask])) if num_peaks > 0 else 0.0

    threshold = config.PEAK_AQI_THRESHOLD
    target_max = float(scaler.target_scaler_.data_max_[0]) if hasattr(scaler, "target_scaler_") else float(scaler.data_max_[target_idx])
    severity_span = max(float(target_max - threshold), 1e-6)
    weights = calculate_sample_weights(y_actual, threshold, severity_span)
    weights *= np.where(preds_unscaled < y_actual, 3.0, 1.0)
    amse = float(np.mean(weights * np.square(preds_unscaled - y_actual)))

    return rmse, mae, p_recall, p_mae, amse


def _two_stage_cutoff(results_dir, classifier_path, fallback=0.5):
    """Recover the OOF-calibrated two-stage cutoff from the walk-forward log."""
    match = re.search(
        r"two_stage_classifier_(\d{8}_\d{6})\.json$",
        os.path.basename(classifier_path),
    )
    if not match:
        return fallback
    log_path = os.path.join(
        results_dir, f"walk_forward_two_stage_{match.group(1)}.txt"
    )
    if not os.path.exists(log_path):
        return fallback
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            if "OOF-calibrated two-stage cutoff" in line:
                try:
                    return float(line.strip().split("=")[-1])
                except ValueError:
                    return fallback
    return fallback


def main():
    config.set_seed(config.RANDOM_SEED)
    results_dir = config.RESULTS_DIR
    print(f"Generating comprehensive comparison in: {results_dir}")
    print(f"Feature columns ({len(config.FEATURE_COLS)}): {config.FEATURE_COLS}")
    print(f"Forecast horizon: t+{config.HORIZON}")

    # 1. Load data
    _, val_ld, te_ld, scaler = get_dataloaders(data_dir='Dataset/archive')
    (
        X_tr_flat,
        y_tr,
        X_va_flat,
        y_va,
        X_te_flat,
        y_te_flat,
        scaler_xgb,
        test_timestamps,
    ) = get_xgboost_data(data_dir='Dataset/archive', include_target_timestamps=True)

    target_idx = config.FEATURE_COLS.index(config.TARGET_COL) if config.TARGET_COL in config.FEATURE_COLS else 0
    criterion = nn.MSELoss()

    # Actual test targets
    all_y = []
    for _, y in te_ld:
        all_y.append(y.numpy())
    y_test_scaled = np.concatenate(all_y)
    y_actual = inverse_scale_predictions(scaler, y_test_scaled, target_idx).ravel()
    num_test = len(y_actual)
    print(f"Total test samples: {num_test}")

    peak_mask = y_actual >= config.PEAK_AQI_THRESHOLD
    num_peaks = int(peak_mask.sum())
    print(f"Total peak samples (AQI >= {config.PEAK_AQI_THRESHOLD}): {num_peaks} ({num_peaks/num_test*100:.2f}%)")

    predictions = {}
    results_list = []

    # 2. Naive Persistence Baseline
    if config.TARGET_COL in config.FEATURE_COLS:
        naive_preds = []
        for X, _ in te_ld:
            naive_preds.append(X[:, -1, target_idx].numpy())
        naive_scaled = np.concatenate(naive_preds)
        y_persist = inverse_scale_predictions(scaler, naive_scaled, target_idx).ravel()
    else:
        df_raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('Dataset/archive/*.csv'))], ignore_index=True)
        df_raw['Local Time'] = pd.to_datetime(df_raw['Local Time'])
        df_raw = df_raw.sort_values('Local Time').drop_duplicates('Local Time').set_index('Local Time').asfreq('h')
        df_raw.loc[df_raw['AQI'] > 400, 'AQI'] = np.nan
        past_times = pd.DatetimeIndex(test_timestamps) - pd.Timedelta(hours=config.HORIZON)
        y_persist = df_raw.reindex(past_times)['AQI'].ffill().bfill().to_numpy(dtype=np.float64)

    valid_persist = np.isfinite(y_actual) & np.isfinite(y_persist)
    persistence_rmse = float(np.sqrt(mean_squared_error(y_actual[valid_persist], y_persist[valid_persist])))
    persistence_mae = float(mean_absolute_error(y_actual[valid_persist], y_persist[valid_persist]))
    persist_peak_mask = peak_mask & valid_persist
    persist_recall = float((y_persist[persist_peak_mask] >= config.PEAK_AQI_THRESHOLD).sum() / peak_mask.sum()) if peak_mask.sum() > 0 else 0.0
    persist_p_mae = float(mean_absolute_error(y_actual[persist_peak_mask], y_persist[persist_peak_mask])) if persist_peak_mask.sum() > 0 else 0.0
    _, _, _, _, persist_amse = calc_metrics(y_actual[valid_persist], y_persist[valid_persist], persist_peak_mask[valid_persist], scaler, target_idx)

    predictions["Persistence"] = y_persist
    results_list.append({
        "model": "Persistence",
        "rmse": persistence_rmse,
        "mae": persistence_mae,
        "peak_recall": persist_recall,
        "peak_mae": persist_p_mae,
        "amse": persist_amse,
    })

    # 3. Base LSTM
    base_pt = os.path.join(results_dir, "base_best_model.pt")
    if os.path.exists(base_pt):
        base_lstm = load_pt_model(LSTMAQIModel, base_pt)
        _, preds_scaled, _ = evaluate_model(base_lstm, te_ld, criterion, config.DEVICE)
        preds_unscaled = inverse_scale_predictions(scaler, preds_scaled, target_idx).ravel()
        predictions["Base LSTM"] = preds_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, preds_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "Base LSTM", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # 4. Stacked LSTM
    stacked_pt = os.path.join(results_dir, "stacked_best_model.pt")
    if os.path.exists(stacked_pt):
        stacked_lstm = load_pt_model(StackedLSTMAQIModel, stacked_pt, num_layers=2)
        _, preds_scaled, _ = evaluate_model(stacked_lstm, te_ld, criterion, config.DEVICE)
        preds_unscaled = inverse_scale_predictions(scaler, preds_scaled, target_idx).ravel()
        predictions["Stacked LSTM"] = preds_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, preds_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "Stacked LSTM", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # 5. GRU
    gru_pt = os.path.join(results_dir, "gru_best_model.pt")
    if os.path.exists(gru_pt):
        gru = load_pt_model(GRUAQIModel, gru_pt)
        _, preds_scaled, _ = evaluate_model(gru, te_ld, criterion, config.DEVICE)
        preds_unscaled = inverse_scale_predictions(scaler, preds_scaled, target_idx).ravel()
        predictions["GRU"] = preds_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, preds_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "GRU", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # 6. XGBoost (Untuned)
    xgb_json = os.path.join(results_dir, "best_xgb_model.json")
    if os.path.exists(xgb_json):
        bst = xgb.Booster()
        bst.load_model(xgb_json)
        dtest = xgb.DMatrix(X_te_flat)
        preds_scaled = bst.predict(dtest)
        preds_unscaled = inverse_scale_predictions(scaler_xgb, preds_scaled.reshape(-1, 1), target_idx).ravel()
        predictions["XGBoost"] = preds_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, preds_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "XGBoost", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # 7. Tuned XGBoost
    tuned_xgb_files = sorted(glob.glob(os.path.join(results_dir, "tuned_best_xgb_*.json")))
    if tuned_xgb_files:
        tuned_bst = xgb.Booster()
        tuned_bst.load_model(tuned_xgb_files[-1])
        dtest = xgb.DMatrix(X_te_flat)
        preds_scaled = tuned_bst.predict(dtest)
        preds_unscaled = inverse_scale_predictions(scaler_xgb, preds_scaled.reshape(-1, 1), target_idx).ravel()
        predictions["Tuned XGBoost"] = preds_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, preds_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "Tuned XGBoost", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # 8. CNN-GRU + Attention
    cnn_gru_pt = os.path.join(results_dir, "cnn_gru_attention_best.pt")
    if os.path.exists(cnn_gru_pt):
        cnn_gru = load_pt_model(CNNRecurrentAttentionAQIModel, cnn_gru_pt, recurrent_type="GRU")
        _, preds_scaled, _ = evaluate_model(cnn_gru, te_ld, criterion, config.DEVICE)
        preds_unscaled = inverse_scale_predictions(scaler, preds_scaled, target_idx).ravel()
        predictions["CNN-GRU+Attention"] = preds_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, preds_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "CNN-GRU+Attention", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # 9. CNN-LSTM + Attention
    cnn_lstm_pt = os.path.join(results_dir, "cnn_lstm_attention_best.pt")
    if os.path.exists(cnn_lstm_pt):
        cnn_lstm = load_pt_model(CNNRecurrentAttentionAQIModel, cnn_lstm_pt, recurrent_type="LSTM")
        _, preds_scaled, _ = evaluate_model(cnn_lstm, te_ld, criterion, config.DEVICE)
        preds_unscaled = inverse_scale_predictions(scaler, preds_scaled, target_idx).ravel()
        predictions["CNN-LSTM+Attention"] = preds_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, preds_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "CNN-LSTM+Attention", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # 10. Soft-Gated Hybrid Ensemble
    gate_pt = os.path.join(results_dir, "hybrid_soft_gate.pt")
    hybrid_xgb_path = os.path.join(results_dir, "hybrid_xgb.json")
    gru_amse_path = os.path.join(results_dir, "gru_amse_best.pt")
    stacked_amse_path = os.path.join(results_dir, "stacked_lstm_amse_best.pt")
    if os.path.exists(gate_pt) and os.path.exists(hybrid_xgb_path) and os.path.exists(gru_amse_path) and os.path.exists(stacked_amse_path):
        h_xgb = xgb.Booster()
        h_xgb.load_model(hybrid_xgb_path)
        dtest = xgb.DMatrix(X_te_flat)
        h_xgb_preds = h_xgb.predict(dtest)

        h_gru = load_pt_model(GRUAQIModel, gru_amse_path)
        _, h_gru_preds, _ = evaluate_model(h_gru, te_ld, criterion, config.DEVICE)

        h_stacked = load_pt_model(StackedLSTMAQIModel, stacked_amse_path, num_layers=2)
        _, h_stacked_preds, _ = evaluate_model(h_stacked, te_ld, criterion, config.DEVICE)

        preds_matrix = np.column_stack((h_xgb_preds.ravel(), h_gru_preds.ravel(), h_stacked_preds.ravel()))
        context_idx = (
            config.FEATURE_COLS.index(config.TARGET_COL)
            if config.TARGET_COL in config.FEATURE_COLS
            else (config.FEATURE_COLS.index("PM25") if "PM25" in config.FEATURE_COLS else 0)
        )
        test_context = make_gating_context(
            X_te_flat.reshape(-1, config.WINDOW_SIZE, len(config.FEATURE_COLS)),
            context_idx
        )

        gate_data = torch.load(gate_pt, map_location=config.DEVICE, weights_only=True)
        state_dict = gate_data["state_dict"]
        feat_mean = state_dict["feature_mean"].cpu()
        feat_std = state_dict["feature_std"].cpu()
        model_count = len(gate_data["model_names"])
        gate = SoftGatingEnsemble(feat_mean, feat_std, model_count)
        gate.load_state_dict(state_dict)
        gate.to(config.DEVICE)
        gate.eval()

        with torch.no_grad():
            gated_test, _ = gate(
                torch.as_tensor(preds_matrix, dtype=torch.float32, device=config.DEVICE),
                torch.as_tensor(test_context, dtype=torch.float32, device=config.DEVICE)
            )
        gated_unscaled = inverse_scale_predictions(scaler, gated_test.cpu().numpy().reshape(-1, 1), target_idx).ravel()
        predictions["Soft-Gated Ensemble"] = gated_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, gated_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "Soft-Gated Ensemble", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # 11. Two-Stage Walk-Forward Architecture
    clf_files = sorted(glob.glob(os.path.join(results_dir, "two_stage_classifier_*.json")))
    gen_files = sorted(glob.glob(os.path.join(results_dir, "two_stage_general_regressor_*.json")))
    exp_files = sorted(glob.glob(os.path.join(results_dir, "two_stage_peak_expert_*.json")))
    if clf_files and gen_files and exp_files:
        c_model = xgb.Booster()
        c_model.load_model(clf_files[-1])
        g_model = xgb.Booster()
        g_model.load_model(gen_files[-1])
        e_model = xgb.Booster()
        e_model.load_model(exp_files[-1])

        dtest = xgb.DMatrix(X_te_flat)
        c_prob = c_model.predict(dtest)
        g_pred = g_model.predict(dtest)
        e_pred = e_model.predict(dtest)

        two_stage_cutoff = _two_stage_cutoff(results_dir, clf_files[-1])
        print(f"Two-stage cutoff (from walk-forward log): {two_stage_cutoff:.6f}")
        two_stage_scaled = np.where(c_prob >= two_stage_cutoff, e_pred, g_pred)
        two_stage_unscaled = inverse_scale_predictions(scaler_xgb, two_stage_scaled.reshape(-1, 1), target_idx).ravel()
        predictions["Two-Stage Regressor"] = two_stage_unscaled
        rmse, mae, p_rec, p_mae, amse = calc_metrics(y_actual, two_stage_unscaled, peak_mask, scaler, target_idx)
        results_list.append({
            "model": "Two-Stage Regressor", "rmse": rmse, "mae": mae, "peak_recall": p_rec, "peak_mae": p_mae, "amse": amse
        })

    # Find best models
    learned_models = [r for r in results_list if r["model"] != "Persistence"]
    best_by_rmse = min(learned_models, key=lambda x: x["rmse"])["model"] if learned_models else "None"
    best_by_mae = min(learned_models, key=lambda x: x["mae"])["model"] if learned_models else "None"
    best_by_peak = max(learned_models, key=lambda x: x["peak_recall"])["model"] if learned_models else "None"
    best_by_amse = min(learned_models, key=lambda x: x["amse"])["model"] if learned_models else "None"

    # Save comparison.json
    comp_data = {
        "features": config.FEATURE_COLS,
        "window_size": config.WINDOW_SIZE,
        "horizon": config.HORIZON,
        "test_samples": num_test,
        "peak_samples": num_peaks,
        "device": str(config.DEVICE),
        "results": results_list,
        "best_by_rmse": best_by_rmse,
        "best_by_mae": best_by_mae,
        "best_by_peak_recall": best_by_peak,
        "best_by_amse": best_by_amse
    }

    comp_path = os.path.join(results_dir, "comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)
    print(f"\nSaved comparison to: {comp_path}")

    # Print summary table
    print("\n" + "="*105)
    print(f"{'Model':<22} | {'Test RMSE':<10} | {'Test MAE':<10} | {'AMSE (Risk)':<12} | {'Peak Recall':<12} | {'Peak MAE':<10}")
    print("-" * 105)
    for r in results_list:
        p_rec_str = f"{r.get('peak_recall', 0)*100:.2f}%"
        p_mae_str = f"{r.get('peak_mae', 0):.2f}" if "peak_mae" in r else "N/A"
        amse_str = f"{r.get('amse', 0):.2f}" if "amse" in r else "N/A"
        print(f"{r['model']:<22} | {r['rmse']:<10.2f} | {r['mae']:<10.2f} | {amse_str:<12} | {p_rec_str:<12} | {p_mae_str:<10}")
    print("="*105)

    # 7. Plot overlay comparison
    if predictions:
        plt.figure(figsize=(18, 8), dpi=300)
        num_samples = min(250, num_test)
        x_axis = range(num_samples)

        plt.plot(x_axis, y_actual[:num_samples], label="Actual AQI", color="black", linewidth=2.2, zorder=10)
        
        palette = {
            "Persistence": {"color": "#9E9E9E", "ls": ":", "lw": 1.2},
            "Base LSTM": {"color": "#64B5F6", "ls": "--", "lw": 1.2},
            "Stacked LSTM": {"color": "#BA68C8", "ls": "--", "lw": 1.2},
            "GRU": {"color": "#81C784", "ls": "--", "lw": 1.2},
            "XGBoost": {"color": "#FF8A65", "ls": "-", "lw": 1.3},
            "Tuned XGBoost": {"color": "#D84315", "ls": "-", "lw": 1.5},
            "CNN-GRU+Attention": {"color": "#00897B", "ls": "-.", "lw": 1.6},
            "CNN-LSTM+Attention": {"color": "#00ACC1", "ls": "-.", "lw": 1.6},
            "Soft-Gated Ensemble": {"color": "#E91E63", "ls": "-", "lw": 2.0},
            "Two-Stage Regressor": {"color": "#7B1FA2", "ls": "-", "lw": 1.8},
        }

        for model_name, preds_arr in predictions.items():
            if model_name in palette:
                plt.plot(
                    x_axis,
                    preds_arr[:num_samples],
                    label=model_name,
                    color=palette[model_name]["color"],
                    linestyle=palette[model_name]["ls"],
                    linewidth=palette[model_name]["lw"],
                    alpha=0.85
                )

        plt.axhline(config.PEAK_AQI_THRESHOLD, color="red", linestyle=":", alpha=0.7, label=f"Peak AQI Threshold ({int(config.PEAK_AQI_THRESHOLD)})")
        feature_desc = (
            "No AQI in Features"
            if config.TARGET_COL not in config.FEATURE_COLS
            else "AQI in Features"
        )
        plt.title(
            f"Comprehensive Model Comparison ({feature_desc}, t+{config.HORIZON} Horizon)",
            fontsize=14, fontweight="bold",
        )
        plt.xlabel("Test Step (Hours)", fontsize=12)
        plt.ylabel("AQI", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right", framealpha=0.92, fontsize=10, ncol=2)
        plt.tight_layout()

        overlay_path = os.path.join(results_dir, "models_overlay.png")
        plt.savefig(overlay_path)
        plt.close()
        print(f"Saved comprehensive overlay plot to: {overlay_path}")

if __name__ == "__main__":
    main()
