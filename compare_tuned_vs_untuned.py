"""
compare_tuned_vs_untuned.py
---------------------------
For each model family (Base LSTM, Stacked LSTM, GRU, XGBoost) this script
produces one subplot showing the untuned vs tuned version side-by-side on
the same test-set window, together with the Actual AQI ground truth.

Usage:
    python compare_tuned_vs_untuned.py
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

import src.config as config
from src.config import set_seed
from src.data_loader import get_dataloaders, get_xgboost_data
from src.model import LSTMAQIModel, StackedLSTMAQIModel, GRUAQIModel
from src.train import train_model, evaluate_model
from src.evaluate import inverse_scale_predictions

# ── Checkpoint paths ──────────────────────────────────────────────────────────
# Untuned (re-trained in-memory; saved here for reproducibility)
BASE_LSTM_UNTUNED_PT    = "results/base_best_model.pt"
STACKED_LSTM_UNTUNED_PT = "results/stacked_best_model.pt"
GRU_UNTUNED_PT          = "results/gru_best_model.pt"
XGB_UNTUNED_JSON        = "results/best_xgb_model_untuned.json"

# Tuned
BASE_LSTM_TUNED_PT      = "results/tuned_best_model_20260716_170213.pt"
STACKED_LSTM_TUNED_PT   = "results/tuned_stacked_best_model_20260716_210810.pt"
GRU_TUNED_PT            = "results/tuned_best_gru_20260716_231314.pt"
XGB_TUNED_JSON          = "results/tuned_best_xgb_20260717_075406.json"

# ── Untuned XGBoost default hyperparams (from run_xgb.py) ────────────────────
XGB_DEFAULT_PARAMS = {
    "verbosity": 0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "booster": "gbtree",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "seed": config.RANDOM_SEED,
}

# ── Plot settings ─────────────────────────────────────────────────────────────
NUM_SAMPLES = 200
SAVE_PATH   = "results/tuned_vs_untuned_overlay.png"
DPI         = 300

# ── Colour scheme ─────────────────────────────────────────────────────────────
C_ACTUAL   = "#111111"
C_UNTUNED  = "#90CAF9"   # light blue
C_TUNED    = "#1565C0"   # dark blue

PANEL_COLORS = {
    "Base LSTM":    ("#BBDEFB", "#1565C0"),   # light / dark blue
    "Stacked LSTM": ("#E1BEE7", "#6A1B9A"),   # light / dark purple
    "GRU":          ("#C8E6C9", "#2E7D32"),   # light / dark green
    "XGBoost":      ("#FFCCBC", "#BF360C"),   # light / dark orange
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_nn_predictions(model, loader, scaler):
    criterion = nn.MSELoss()
    _, preds, y_true = evaluate_model(model, loader, criterion, config.DEVICE)
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    return (
        inverse_scale_predictions(scaler, preds,  target_idx),
        inverse_scale_predictions(scaler, y_true, target_idx),
    )


def load_nn(model_obj, checkpoint_path):
    model_obj.load_state_dict(
        torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    )
    model_obj.to(config.DEVICE)
    model_obj.eval()
    return model_obj


def train_and_save(model_obj, tr_ld, va_ld, checkpoint_path, patience=5):
    """Train an untuned model with default config params and save its checkpoint."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=config.LEARNING_RATE)
    model_obj, _ = train_model(
        model=model_obj,
        train_loader=tr_ld,
        val_loader=va_ld,
        criterion=criterion,
        optimizer=optimizer,
        device=config.DEVICE,
        epochs=config.EPOCHS,
        patience=patience,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    return model_obj


def train_xgb_untuned(X_tr, y_tr, X_va, y_va):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    bst = xgb.train(
        XGB_DEFAULT_PARAMS,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dvalid, "val")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    bst.save_model(XGB_UNTUNED_JSON)
    return bst


def xgb_predict(bst, X_te, scaler):
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    d = xgb.DMatrix(X_te)
    preds_scaled = bst.predict(d)
    return inverse_scale_predictions(scaler, preds_scaled.reshape(-1, 1), target_idx)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(config.RANDOM_SEED)

    # ── 1. Load shared data ───────────────────────────────────────────────────
    print("Loading data ...")
    tr_ld, va_ld, te_ld, scaler = get_dataloaders(data_dir="Dataset/archive")
    X_tr, y_tr, X_va, y_va, X_te, y_te, scaler_xgb = get_xgboost_data(data_dir="Dataset/archive")

    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)

    # ── 2a. Untuned Base LSTM ─────────────────────────────────────────────────
    if os.path.exists(BASE_LSTM_UNTUNED_PT):
        print("Loading cached untuned Base LSTM ...")
        base_untuned = load_nn(LSTMAQIModel(), BASE_LSTM_UNTUNED_PT)
    else:
        print("Training untuned Base LSTM ...")
        set_seed(config.RANDOM_SEED)
        base_untuned = train_and_save(
            LSTMAQIModel().to(config.DEVICE), tr_ld, va_ld, BASE_LSTM_UNTUNED_PT
        )

    pred_base_untuned, y_true = get_nn_predictions(base_untuned, te_ld, scaler)

    # ── 2b. Tuned Base LSTM ───────────────────────────────────────────────────
    print("Loading tuned Base LSTM ...")
    base_tuned = load_nn(LSTMAQIModel(hidden_size=128, dropout=0.0), BASE_LSTM_TUNED_PT)
    pred_base_tuned, _ = get_nn_predictions(base_tuned, te_ld, scaler)

    # ── 3a. Untuned Stacked LSTM ──────────────────────────────────────────────
    print("Loading untuned Stacked LSTM ...")
    stacked_untuned = load_nn(StackedLSTMAQIModel(hidden_size=128, dropout=0.0), STACKED_LSTM_UNTUNED_PT)
    pred_stacked_untuned, _ = get_nn_predictions(stacked_untuned, te_ld, scaler)

    # ── 3b. Tuned Stacked LSTM ────────────────────────────────────────────────
    print("Loading tuned Stacked LSTM ...")
    stacked_tuned = load_nn(StackedLSTMAQIModel(hidden_size=256, num_layers=2, dropout=0.0), STACKED_LSTM_TUNED_PT)
    pred_stacked_tuned, _ = get_nn_predictions(stacked_tuned, te_ld, scaler)

    # ── 4a. Untuned GRU ───────────────────────────────────────────────────────
    if os.path.exists(GRU_UNTUNED_PT):
        print("Loading cached untuned GRU ...")
        gru_untuned = load_nn(GRUAQIModel(hidden_size=128, dropout=0.0), GRU_UNTUNED_PT)
    else:
        print("Training untuned GRU ...")
        set_seed(config.RANDOM_SEED)
        gru_untuned = train_and_save(
            GRUAQIModel().to(config.DEVICE), tr_ld, va_ld, GRU_UNTUNED_PT
        )

    pred_gru_untuned, _ = get_nn_predictions(gru_untuned, te_ld, scaler)

    # ── 4b. Tuned GRU ─────────────────────────────────────────────────────────
    print("Loading tuned GRU ...")
    gru_tuned = load_nn(GRUAQIModel(hidden_size=256, dropout=0.0), GRU_TUNED_PT)
    pred_gru_tuned, _ = get_nn_predictions(gru_tuned, te_ld, scaler)

    # ── 5a. Untuned XGBoost ───────────────────────────────────────────────────
    if os.path.exists(XGB_UNTUNED_JSON):
        print("Loading cached untuned XGBoost ...")
        bst_untuned = xgb.Booster()
        bst_untuned.load_model(XGB_UNTUNED_JSON)
    else:
        print("Training untuned XGBoost ...")
        set_seed(config.RANDOM_SEED)
        bst_untuned = train_xgb_untuned(X_tr, y_tr, X_va, y_va)

    pred_xgb_untuned = xgb_predict(bst_untuned, X_te, scaler_xgb)

    # ── 5b. Tuned XGBoost ─────────────────────────────────────────────────────
    print("Loading tuned XGBoost ...")
    bst_tuned = xgb.Booster()
    bst_tuned.load_model(XGB_TUNED_JSON)
    pred_xgb_tuned = xgb_predict(bst_tuned, X_te, scaler_xgb)

    # ── 6. Slice to NUM_SAMPLES ───────────────────────────────────────────────
    sl  = slice(0, NUM_SAMPLES)
    xs  = np.arange(NUM_SAMPLES)
    act = y_true[sl]

    panels = [
        ("Base LSTM",    pred_base_untuned[sl],    pred_base_tuned[sl]),
        ("Stacked LSTM", pred_stacked_untuned[sl], pred_stacked_tuned[sl]),
        ("GRU",          pred_gru_untuned[sl],     pred_gru_tuned[sl]),
        ("XGBoost",      pred_xgb_untuned[sl],     pred_xgb_tuned[sl]),
    ]

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor("#F9F9F9")
    fig.suptitle(
        f"Untuned vs Tuned — Per-Model Overlay on Test Set\n"
        f"(first {NUM_SAMPLES} samples, t+{config.HORIZON}h forecast)",
        fontsize=15, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

    for i, (name, pred_un, pred_tu) in enumerate(panels):
        c_light, c_dark = PANEL_COLORS[name]
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#F5F5F5")

        # Actual
        ax.plot(xs, act, color=C_ACTUAL, linewidth=1.8, label="Actual AQI",
                zorder=10, alpha=0.95)
        # Untuned
        ax.plot(xs, pred_un, color=c_light, linewidth=1.5,
                linestyle="--", label=f"{name} (Untuned)",
                alpha=0.85, zorder=5)
        # Tuned
        ax.plot(xs, pred_tu, color=c_dark, linewidth=1.5,
                linestyle="-", label=f"{name} (Tuned)",
                alpha=0.90, zorder=6)

        ax.set_title(name, fontsize=12, fontweight="bold",
                     color=c_dark, pad=6)
        ax.set_ylabel("AQI", fontsize=10)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.88)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i == len(panels) - 1:
            ax.set_xlabel("Time Step (test set index)", fontsize=10)

    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"\nDone! Saved overlay chart -> {SAVE_PATH}")


if __name__ == "__main__":
    main()
