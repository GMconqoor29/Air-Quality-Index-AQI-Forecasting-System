"""Export the training MinMaxScaler (including target_scaler_) for real-time inference.

Usage:
    .venv/bin/python scripts/export_scaler.py
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.data_loader import _get_scaled_sequence_splits

SCALER_PATH = os.path.join("results", "scaler.joblib")


def export_scaler(data_dir: str = "Dataset/archive", out_path: str = SCALER_PATH):
    print(f"Fitting MinMaxScaler on training split from '{data_dir}'...")
    print(f"Feature columns ({len(config.FEATURE_COLS)}): {config.FEATURE_COLS}")
    _, scaler = _get_scaled_sequence_splits(data_dir, verbose=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bundle = {
        "scaler": scaler,
        "feature_cols": list(config.FEATURE_COLS),
        "window_size": int(config.WINDOW_SIZE),
        "target_col": config.TARGET_COL,
    }
    joblib.dump(bundle, out_path)
    print(f"Saved scaler bundle to '{out_path}'.")
    return bundle


if __name__ == "__main__":
    export_scaler()

