"""
compare_tuned_models.py
-----------------------
Loads all four tuned model checkpoints and XGBoost model, collects their test-set
predictions, and plots them together on a single overlay chart for easy comparison.

Usage:
    python compare_tuned_models.py
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Ensure the src package is importable from project root ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

import src.config as config
from src.config import set_seed
from src.data_loader import get_dataloaders, get_xgboost_data
from src.model import LSTMAQIModel, StackedLSTMAQIModel, GRUAQIModel
from src.train import evaluate_model
from src.evaluate import inverse_scale_predictions

import glob

def find_checkpoint(pattern):
    """Resolve a checkpoint path under RESULTS_DIR.

    Wildcard patterns glob for the newest matching file; literal filenames
    return a path that may not exist yet (so callers can train it on demand).
    """
    if any(ch in pattern for ch in "*?["):
        matches = glob.glob(os.path.join(config.RESULTS_DIR, pattern))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint matching {pattern!r} found in {config.RESULTS_DIR}."
            )
        return max(matches, key=os.path.getmtime)
    return os.path.join(config.RESULTS_DIR, pattern)

# ── Checkpoint paths ─────────────────────────────────────────────────────────
TUNED_BASE_LSTM_PT    = find_checkpoint("tuned_best_model_*.pt")
TUNED_STACKED_LSTM_PT = find_checkpoint("tuned_stacked_best_model_*.pt")
TUNED_GRU_PT          = find_checkpoint("tuned_best_gru_*.pt")
TUNED_XGB_JSON        = find_checkpoint("tuned_best_xgb_*.json")

# ── Plot settings ─────────────────────────────────────────────────────────────
NUM_SAMPLES   = 200          # number of test-set time steps to display
SAVE_PATH     = os.path.join(config.RESULTS_DIR, "tuned_models_overlay.png")
FIGSIZE       = (18, 7)
DPI           = 300

# ── Colour palette (accessible & distinct) ───────────────────────────────────
PALETTE = {
    "Actual AQI":         {"color": "#1f1f1f", "lw": 2.0,  "ls": "-",  "alpha": 1.0,  "zorder": 10},
    "Tuned Base LSTM":    {"color": "#2196F3", "lw": 1.4,  "ls": "--", "alpha": 0.85, "zorder": 4},
    "Tuned Stacked LSTM": {"color": "#9C27B0", "lw": 1.4,  "ls": "-.", "alpha": 0.85, "zorder": 5},
    "Tuned GRU":          {"color": "#4CAF50", "lw": 1.4,  "ls": ":",  "alpha": 0.85, "zorder": 6},
    "Tuned XGBoost":      {"color": "#FF5722", "lw": 1.4,  "ls": "-",  "alpha": 0.80, "zorder": 7},
}


def get_nn_predictions(model, loader, scaler):
    """Run inference for a PyTorch model and return unscaled predictions."""
    criterion = nn.MSELoss()
    _, preds, y_true = evaluate_model(model, loader, criterion, config.DEVICE)
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL) if config.TARGET_COL in config.FEATURE_COLS else 0
    preds_unscaled  = inverse_scale_predictions(scaler, preds,  target_idx)
    y_true_unscaled = inverse_scale_predictions(scaler, y_true, target_idx)
    return preds_unscaled, y_true_unscaled


def load_lstm(checkpoint_path, hidden_size=None):
    state = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    if hidden_size is None:
        hidden_size = state['fc.weight'].shape[1] if 'fc.weight' in state else 128
    model = LSTMAQIModel(hidden_size=hidden_size, dropout=0.0)
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()
    return model


def load_stacked_lstm(checkpoint_path, hidden_size=None, num_layers=None):
    state = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    if hidden_size is None:
        hidden_size = state['fc.weight'].shape[1] if 'fc.weight' in state else 256
    if num_layers is None:
        layer_indices = [int(k.split('_l')[1]) for k in state.keys() if '_l' in k and 'weight_ih_l' in k]
        num_layers = max(layer_indices) + 1 if layer_indices else 2
    model = StackedLSTMAQIModel(hidden_size=hidden_size, num_layers=num_layers, dropout=0.0)
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()
    return model


def load_gru(checkpoint_path, hidden_size=None):
    state = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    if hidden_size is None:
        hidden_size = state['fc.weight'].shape[1] if 'fc.weight' in state else 256
    model = GRUAQIModel(hidden_size=hidden_size, dropout=0.0)
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()
    return model


def main():
    set_seed(config.RANDOM_SEED)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data …")
    _, _, te_ld, scaler = get_dataloaders()
    _, _, _, _, X_te_flat, y_te_flat, scaler_xgb = get_xgboost_data()

    target_idx = config.FEATURE_COLS.index(config.TARGET_COL) if config.TARGET_COL in config.FEATURE_COLS else 0

    # ── 2. Tuned Base LSTM ────────────────────────────────────────────────────
    print("Loading Tuned Base LSTM …")
    base_lstm = load_lstm(TUNED_BASE_LSTM_PT)
    pred_base, y_true = get_nn_predictions(base_lstm, te_ld, scaler)

    # ── 3. Tuned Stacked LSTM ────────────────────────────────────────────────
    print("Loading Tuned Stacked LSTM …")
    stacked_lstm = load_stacked_lstm(TUNED_STACKED_LSTM_PT)
    pred_stacked, _ = get_nn_predictions(stacked_lstm, te_ld, scaler)

    # ── 4. Tuned GRU ─────────────────────────────────────────────────────────
    print("Loading Tuned GRU …")
    gru = load_gru(TUNED_GRU_PT)
    pred_gru, _ = get_nn_predictions(gru, te_ld, scaler)

    # ── 5. Tuned XGBoost ─────────────────────────────────────────────────────
    print("Loading Tuned XGBoost …")
    bst = xgb.Booster()
    bst.load_model(TUNED_XGB_JSON)
    d_test = xgb.DMatrix(X_te_flat)
    pred_xgb_scaled = bst.predict(d_test)
    pred_xgb = inverse_scale_predictions(scaler_xgb, pred_xgb_scaled.reshape(-1, 1), target_idx)

    # ── 6. Slice to NUM_SAMPLES ───────────────────────────────────────────────
    sl = slice(0, NUM_SAMPLES)
    xs = np.arange(NUM_SAMPLES)

    actual       = y_true[sl]
    pred_base_s  = pred_base[sl]
    pred_stack_s = pred_stacked[sl]
    pred_gru_s   = pred_gru[sl]
    pred_xgb_s   = pred_xgb[sl]

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    def _plot(label, y):
        s = PALETTE[label]
        ax.plot(xs, y, label=label,
                color=s["color"], linewidth=s["lw"],
                linestyle=s["ls"], alpha=s["alpha"], zorder=s["zorder"])

    _plot("Actual AQI",         actual)
    _plot("Tuned Base LSTM",    pred_base_s)
    _plot("Tuned Stacked LSTM", pred_stack_s)
    _plot("Tuned GRU",          pred_gru_s)
    _plot("Tuned XGBoost",      pred_xgb_s)

    ax.set_title(
        f"Tuned Models — Overlay Comparison on Test Set (first {NUM_SAMPLES} samples, t+{config.HORIZON}h)",
        fontsize=14, fontweight="bold", pad=14
    )
    ax.set_xlabel("Time Step (test set index)", fontsize=11)
    ax.set_ylabel("AQI", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=DPI)
    plt.close()
    print(f"\nDone! Saved overlay chart -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
