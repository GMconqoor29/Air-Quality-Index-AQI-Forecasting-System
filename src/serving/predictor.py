"""Real-time multi-model AQI predictor for t+1 and t+6 horizons.

Loads the persisted MinMaxScaler (`results/scaler.joblib`) and all trained models
from `results/t1/` and `results/t6/`, supporting:
  - Best available (Tuned XGBoost / XGBoost)
  - XGBoost
  - Tuned XGBoost
  - Soft-Gated Ensemble
  - Two-Stage Regressor
  - CNN-LSTM Attention
  - CNN-GRU Attention
  - Stacked LSTM
  - GRU
  - Base LSTM
"""

import glob
import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

import src.config as config
from src.evaluate import inverse_scale_predictions
from src.hybrid_ensemble import SoftGatingEnsemble, make_gating_context
from src.model import (
    CNNRecurrentAttentionAQIModel,
    GRUAQIModel,
    LSTMAQIModel,
    StackedLSTMAQIModel,
)
from src.serving.aqi_calculator import compute_aqi_from_pm

SCALER_PATH = os.path.join("results", "scaler.joblib")

# Holdout benchmark table (t+1) from `Tìm hiểu thêm.pdf` (6,020 test samples, 1,027 peaks)
MODEL_BENCHMARKS_T1 = [
    {"model": "Persistence",         "rmse": 31.24, "mae": 18.91, "peakRecall": 0.692, "peakMAE": 26.96, "amse": 2533.0},
    {"model": "Base LSTM",           "rmse": 30.47, "mae": 22.04, "peakRecall": 0.393, "peakMAE": 39.45, "amse": 2721.7},
    {"model": "GRU",                 "rmse": 30.24, "mae": 21.95, "peakRecall": 0.375, "peakMAE": 39.75, "amse": 2672.6},
    {"model": "Stacked LSTM",        "rmse": 29.44, "mae": 20.69, "peakRecall": 0.722, "peakMAE": 27.72, "amse": 2003.8},
    {"model": "Tuned XGBoost",       "rmse": 28.66, "mae": 19.82, "peakRecall": 0.621, "peakMAE": 31.33, "amse": 2281.9},
    {"model": "XGBoost",             "rmse": 28.42, "mae": 19.66, "peakRecall": 0.639, "peakMAE": 31.14, "amse": 2257.4},
    {"model": "Soft-Gated Ensemble", "rmse": 31.67, "mae": 22.67, "peakRecall": 0.834, "peakMAE": 20.57, "amse": 1664.3},
    {"model": "CNN-LSTM Attention",  "rmse": 32.53, "mae": 23.25, "peakRecall": 0.850, "peakMAE": 20.41, "amse": 1714.8},
    {"model": "CNN-GRU Attention",   "rmse": 33.43, "mae": 23.46, "peakRecall": 0.857, "peakMAE": 21.01, "amse": 1776.7},
    {"model": "Two-Stage Regressor", "rmse": 33.90, "mae": 23.10, "peakRecall": 0.904, "peakMAE": 21.68, "amse": 1979.2},
]

RMSE_BY_MODEL = {row["model"]: row["rmse"] for row in MODEL_BENCHMARKS_T1}


def _find_artifact(horizon_dir: str, subdir: str, pattern: str) -> str | None:
    """Locate a model checkpoint in either `horizon_dir/subdir/pattern` or `horizon_dir/pattern`."""
    candidates = []
    for base in (os.path.join(horizon_dir, subdir), horizon_dir):
        full_pattern = os.path.join(base, pattern)
        if any(ch in pattern for ch in "*?["):
            candidates.extend(glob.glob(full_pattern))
        elif os.path.exists(full_pattern):
            candidates.append(full_pattern)
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _extract_two_stage_cutoff(horizon_dir: str, fallback: float = 0.50375) -> float:
    log_files = sorted(
        glob.glob(os.path.join(horizon_dir, "two_stage", "walk_forward_two_stage_*.txt"))
        + glob.glob(os.path.join(horizon_dir, "walk_forward_two_stage_*.txt"))
    )
    for path in reversed(log_files):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if "OOF-calibrated two-stage cutoff=" in line:
                        return float(line.strip().split("=")[-1])
        except OSError:
            continue
    return fallback


class RealtimePredictor:
    """Loads scaler and trained models for t+1 and t+6 real-time inference."""

    def __init__(self, scaler_path: str = SCALER_PATH, device: str = "cpu"):
        self.device = device
        self.scaler_path = scaler_path
        self.scaler = self._load_or_create_scaler(scaler_path)
        self.feature_cols = list(config.FEATURE_COLS)
        self.window_size = int(config.WINDOW_SIZE)
        self.target_idx = (
            self.feature_cols.index(config.TARGET_COL)
            if config.TARGET_COL in self.feature_cols
            else 0
        )
        self.context_idx = (
            self.feature_cols.index("PM25")
            if "PM25" in self.feature_cols
            else 0
        )
        self._models_by_horizon: dict[int, dict] = {}
        for h in (1, 6):
            h_dir = os.path.join("results", f"t{h}")
            if os.path.isdir(h_dir):
                self._models_by_horizon[h] = self._load_horizon_models(h_dir)

    def _load_or_create_scaler(self, scaler_path: str):
        if os.path.exists(scaler_path):
            bundle = joblib.load(scaler_path)
            return bundle["scaler"] if isinstance(bundle, dict) else bundle
        from scripts.export_scaler import export_scaler
        bundle = export_scaler(out_path=scaler_path)
        return bundle["scaler"]

    def _load_pt(self, model_cls, path: str, **kwargs):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if "fc.weight" in state and "hidden_size" not in kwargs:
            kwargs["hidden_size"] = int(state["fc.weight"].shape[1])
        model = model_cls(input_size=len(self.feature_cols), **kwargs)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _load_xgb(self, path: str) -> xgb.Booster:
        booster = xgb.Booster()
        booster.load_model(path)
        return booster

    def _load_horizon_models(self, h_dir: str) -> dict:
        loaded = {}

        # 1. Base LSTM
        p = _find_artifact(h_dir, "lstm", "base_best_model.pt")
        if p:
            loaded["Base LSTM"] = ("nn", self._load_pt(LSTMAQIModel, p))

        # 2. Stacked LSTM
        p = _find_artifact(h_dir, "stacked_lstm", "stacked_best_model.pt")
        if p:
            loaded["Stacked LSTM"] = ("nn", self._load_pt(StackedLSTMAQIModel, p, num_layers=2))

        # 3. GRU
        p = _find_artifact(h_dir, "gru", "gru_best_model.pt")
        if p:
            loaded["GRU"] = ("nn", self._load_pt(GRUAQIModel, p))

        # 4. XGBoost
        p = _find_artifact(h_dir, "xgboost", "best_xgb_model.json")
        if p:
            loaded["XGBoost"] = ("xgb", self._load_xgb(p))

        # 5. Tuned XGBoost
        p = _find_artifact(h_dir, "tuned_xgboost", "tuned_best_xgb_*.json")
        if p:
            loaded["Tuned XGBoost"] = ("xgb", self._load_xgb(p))

        # 6. CNN-GRU Attention
        p = _find_artifact(h_dir, "cnn_recurrent_attention", "cnn_gru_attention_best.pt")
        if p:
            loaded["CNN-GRU Attention"] = (
                "nn",
                self._load_pt(CNNRecurrentAttentionAQIModel, p, recurrent_type="GRU"),
            )

        # 7. CNN-LSTM Attention
        p = _find_artifact(h_dir, "cnn_recurrent_attention", "cnn_lstm_attention_best.pt")
        if p:
            loaded["CNN-LSTM Attention"] = (
                "nn",
                self._load_pt(CNNRecurrentAttentionAQIModel, p, recurrent_type="LSTM"),
            )

        # 8. Soft-Gated Ensemble
        gate_pt = _find_artifact(h_dir, "hybrid_ensemble", "hybrid_soft_gate.pt")
        h_xgb_p = _find_artifact(h_dir, "hybrid_ensemble", "hybrid_xgb.json")
        h_gru_p = _find_artifact(h_dir, "hybrid_ensemble", "gru_amse_best.pt")
        h_stk_p = _find_artifact(h_dir, "hybrid_ensemble", "stacked_lstm_amse_best.pt")
        if gate_pt and h_xgb_p and h_gru_p and h_stk_p:
            gate_data = torch.load(gate_pt, map_location=self.device, weights_only=True)
            sd = gate_data["state_dict"]
            gate = SoftGatingEnsemble(
                sd["feature_mean"].cpu(),
                sd["feature_std"].cpu(),
                len(gate_data["model_names"]),
            )
            gate.load_state_dict(sd)
            gate.to(self.device)
            gate.eval()
            loaded["Soft-Gated Ensemble"] = (
                "soft_gate",
                {
                    "xgb": self._load_xgb(h_xgb_p),
                    "gru": self._load_pt(GRUAQIModel, h_gru_p),
                    "stacked": self._load_pt(StackedLSTMAQIModel, h_stk_p, num_layers=2),
                    "gate": gate,
                },
            )

        # 9. Two-Stage Regressor
        clf_p = _find_artifact(h_dir, "two_stage", "two_stage_classifier_*.json")
        gen_p = _find_artifact(h_dir, "two_stage", "two_stage_general_regressor_*.json")
        exp_p = _find_artifact(h_dir, "two_stage", "two_stage_peak_expert_*.json")
        if clf_p and gen_p and exp_p:
            loaded["Two-Stage Regressor"] = (
                "two_stage",
                {
                    "clf": self._load_xgb(clf_p),
                    "gen": self._load_xgb(gen_p),
                    "exp": self._load_xgb(exp_p),
                    "cutoff": _extract_two_stage_cutoff(h_dir),
                },
            )

        # 10. Dedicated Peak Classifier
        pk_p = _find_artifact(h_dir, "peak_classifier", "xgb_peak_classifier.json")
        if pk_p:
            loaded["_peak_classifier"] = self._load_xgb(pk_p)

        return loaded

    def prepare_window_tensor(self, records: list[dict]) -> np.ndarray:
        """Convert 48 hourly observation dicts into a (1, 48, 7) scaled numpy array."""
        if len(records) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} hourly observations, got {len(records)}."
            )
        window = records[-self.window_size :]
        rows = []
        for item in window:
            ts = pd.to_datetime(item["timestamp"])
            row_map = {
                "PM25": float(item.get("pm25", np.nan)),
                "PM10": float(item.get("pm10", np.nan)),
                "Temperature": float(item.get("temperature", np.nan)),
                "Relative Humidity": float(item.get("relativeHumidity", item.get("relative_humidity", np.nan))),
                "Pressure": float(item.get("pressure", np.nan)),
                "hour": float(ts.hour),
                "month": float(ts.month),
                "AQI": float(
                    item.get("aqi")
                    if item.get("aqi") is not None
                    else (compute_aqi_from_pm(item.get("pm25"), item.get("pm10")) or np.nan)
                ),
            }
            rows.append([row_map[col] for col in self.feature_cols])

        df_feat = pd.DataFrame(rows, columns=self.feature_cols)
        # Forward-fill up to MAX_FORWARD_FILL_HOURS=1, then fill remaining edge NaNs
        df_feat = df_feat.ffill(limit=config.MAX_FORWARD_FILL_HOURS).bfill(limit=1)
        if df_feat.isna().any().any():
            df_feat = df_feat.interpolate(limit_direction="both").ffill().bfill()

        scaled = self.scaler.transform(df_feat.to_numpy(dtype=np.float32))
        return scaled.reshape(1, self.window_size, len(self.feature_cols)).astype(np.float32)

    def _inverse_aqi(self, scaled_val: float) -> float:
        arr = np.asarray([[scaled_val]], dtype=np.float32)
        unscaled = inverse_scale_predictions(self.scaler, arr, self.target_idx)
        return float(np.clip(unscaled.ravel()[0], 0.0, 500.0))

    def _predict_single_loaded(self, entry, x_seq: np.ndarray, x_flat: np.ndarray) -> tuple[float, float | None]:
        kind, obj = entry
        peak_prob = None
        if kind == "xgb":
            dmat = xgb.DMatrix(x_flat)
            pred_scaled = float(obj.predict(dmat)[0])
        elif kind == "nn":
            with torch.no_grad():
                tensor_x = torch.as_tensor(x_seq, dtype=torch.float32, device=self.device)
                pred_scaled = float(obj(tensor_x).cpu().numpy().ravel()[0])
        elif kind == "soft_gate":
            dmat = xgb.DMatrix(x_flat)
            p_xgb = float(obj["xgb"].predict(dmat)[0])
            with torch.no_grad():
                tensor_x = torch.as_tensor(x_seq, dtype=torch.float32, device=self.device)
                p_gru = float(obj["gru"](tensor_x).cpu().numpy().ravel()[0])
                p_stk = float(obj["stacked"](tensor_x).cpu().numpy().ravel()[0])
                preds_mat = np.array([[p_xgb, p_gru, p_stk]], dtype=np.float32)
                ctx = make_gating_context(x_seq, self.context_idx)
                gated, _ = obj["gate"](
                    torch.as_tensor(preds_mat, dtype=torch.float32, device=self.device),
                    torch.as_tensor(ctx, dtype=torch.float32, device=self.device),
                )
                pred_scaled = float(gated.cpu().numpy().ravel()[0])
        elif kind == "two_stage":
            dmat = xgb.DMatrix(x_flat)
            c_prob = float(obj["clf"].predict(dmat)[0])
            peak_prob = c_prob
            g_pred = float(obj["gen"].predict(dmat)[0])
            e_pred = float(obj["exp"].predict(dmat)[0])
            pred_scaled = e_pred if c_prob >= obj["cutoff"] else g_pred
        else:
            raise ValueError(f"Unknown model type '{kind}'.")

        return self._inverse_aqi(pred_scaled), peak_prob

    def predict(
        self,
        records: list[dict],
        horizon: int = 1,
        model_name: str = "Best available",
    ) -> dict:
        """Run real-time inference on the latest 48h window."""
        h = 6 if int(horizon) == 6 else 1
        horizon_models = self._models_by_horizon.get(h, {})
        if not horizon_models:
            raise RuntimeError(f"No trained models found for horizon t+{h}.")

        resolved_name = model_name
        if model_name in ("Best available", "", None):
            resolved_name = (
                "XGBoost"
                if "XGBoost" in horizon_models
                else ("Tuned XGBoost" if "Tuned XGBoost" in horizon_models else next(iter(horizon_models)))
            )
        # Normalize slight naming differences
        alias_map = {
            "CNN-LSTM+Attention": "CNN-LSTM Attention",
            "CNN-GRU+Attention": "CNN-GRU Attention",
        }
        resolved_name = alias_map.get(resolved_name, resolved_name)
        if resolved_name not in horizon_models:
            resolved_name = "XGBoost" if "XGBoost" in horizon_models else next(
                k for k in horizon_models if not k.startswith("_")
            )

        x_seq = self.prepare_window_tensor(records)
        x_flat = x_seq.reshape(1, -1)

        aqi_val, peak_prob = self._predict_single_loaded(
            horizon_models[resolved_name], x_seq, x_flat
        )

        # If peak_prob wasn't produced by Two-Stage, query the peak classifier if available
        if peak_prob is None and "_peak_classifier" in horizon_models:
            dmat = xgb.DMatrix(x_flat)
            peak_prob = float(horizon_models["_peak_classifier"].predict(dmat)[0])

        predicted_aqi = int(round(aqi_val))
        base_rmse = RMSE_BY_MODEL.get(resolved_name, 29.0)
        if h == 6:
            base_rmse *= 1.35
        # 90% confidence interval half-width (~0.55 * RMSE for tighter operational band)
        margin = int(round(max(8.0, base_rmse * 0.52)))

        return {
            "predicted_aqi": predicted_aqi,
            "predicted_aqi_raw": round(aqi_val, 2),
            "model_used": resolved_name,
            "confidence_interval": [
                max(0, predicted_aqi - margin),
                min(500, predicted_aqi + margin),
            ],
            "horizon": h,
            "peak_probability": round(float(peak_prob), 4) if peak_prob is not None else None,
            "is_peak_warning": bool(predicted_aqi >= config.PEAK_AQI_THRESHOLD or (peak_prob is not None and peak_prob >= 0.5)),
        }

    def _inverse_aqi_array(self, scaled_vals: np.ndarray) -> np.ndarray:
        arr = np.asarray(scaled_vals, dtype=np.float32).reshape(-1, 1)
        unscaled = inverse_scale_predictions(self.scaler, arr, self.target_idx).ravel()
        return np.clip(np.round(unscaled), 0.0, 500.0).astype(int)

    def _predict_batch_loaded(self, entry, x_seq_batch: np.ndarray) -> np.ndarray:
        """Vectorized batch inference for N sliding windows of shape (N, 48, 7)."""
        kind, obj = entry
        x_flat_batch = x_seq_batch.reshape(x_seq_batch.shape[0], -1)
        if kind == "xgb":
            dmat = xgb.DMatrix(x_flat_batch)
            pred_scaled = obj.predict(dmat)
        elif kind == "nn":
            with torch.no_grad():
                tensor_x = torch.as_tensor(x_seq_batch, dtype=torch.float32, device=self.device)
                pred_scaled = obj(tensor_x).cpu().numpy().ravel()
        elif kind == "soft_gate":
            dmat = xgb.DMatrix(x_flat_batch)
            p_xgb = obj["xgb"].predict(dmat).ravel()
            with torch.no_grad():
                tensor_x = torch.as_tensor(x_seq_batch, dtype=torch.float32, device=self.device)
                p_gru = obj["gru"](tensor_x).cpu().numpy().ravel()
                p_stk = obj["stacked"](tensor_x).cpu().numpy().ravel()
                preds_mat = np.column_stack((p_xgb, p_gru, p_stk)).astype(np.float32)
                ctx = make_gating_context(x_seq_batch, self.context_idx)
                gated, _ = obj["gate"](
                    torch.as_tensor(preds_mat, dtype=torch.float32, device=self.device),
                    torch.as_tensor(ctx, dtype=torch.float32, device=self.device),
                )
                pred_scaled = gated.cpu().numpy().ravel()
        elif kind == "two_stage":
            dmat = xgb.DMatrix(x_flat_batch)
            c_prob = obj["clf"].predict(dmat).ravel()
            g_pred = obj["gen"].predict(dmat).ravel()
            e_pred = obj["exp"].predict(dmat).ravel()
            pred_scaled = np.where(c_prob >= obj["cutoff"], e_pred, g_pred)
        else:
            raise ValueError(f"Unknown model type '{kind}'.")

        return self._inverse_aqi_array(pred_scaled)

    def attach_rolling_predictions(
        self,
        records: list[dict],
        horizon: int = 1,
        model_name: str = "Best available",
    ) -> list[dict]:
        """Compute rolling 1-step-ahead (or h-step-ahead) predictions aligned with actual readings.

        For each index `i >= window_size - 1 + horizon`, the prediction `predictedAqi`
        at `records[i]` is generated causally from the 48-hour window
        `records[i - horizon - window_size + 1 : i - horizon + 1]`.
        """
        if not records:
            return []
        h = 6 if int(horizon) == 6 else 1
        horizon_models = self._models_by_horizon.get(h, {})
        if not horizon_models:
            return [dict(r, predictedAqi=None) for r in records]

        resolved_name = model_name
        if model_name in ("Best available", "", None):
            resolved_name = "XGBoost" if "XGBoost" in horizon_models else next(iter(horizon_models))
        alias_map = {
            "CNN-LSTM+Attention": "CNN-LSTM Attention",
            "CNN-GRU+Attention": "CNN-GRU Attention",
        }
        resolved_name = alias_map.get(resolved_name, resolved_name)
        if resolved_name not in horizon_models:
            resolved_name = "XGBoost"

        rows = []
        for item in records:
            ts = pd.to_datetime(item["timestamp"])
            row_map = {
                "PM25": float(item.get("pm25", np.nan)),
                "PM10": float(item.get("pm10", np.nan)),
                "Temperature": float(item.get("temperature", np.nan)),
                "Relative Humidity": float(item.get("relativeHumidity", item.get("relative_humidity", np.nan))),
                "Pressure": float(item.get("pressure", np.nan)),
                "hour": float(ts.hour),
                "month": float(ts.month),
                "AQI": float(item.get("aqi", np.nan)),
            }
            rows.append([row_map[col] for col in self.feature_cols])

        df_feat = pd.DataFrame(rows, columns=self.feature_cols)
        df_feat = df_feat.ffill(limit=config.MAX_FORWARD_FILL_HOURS).bfill(limit=1)
        if df_feat.isna().any().any():
            df_feat = df_feat.interpolate(limit_direction="both").ffill().bfill()

        scaled_all = self.scaler.transform(df_feat.to_numpy(dtype=np.float32)).astype(np.float32)

        n_total = len(records)
        w = self.window_size
        start_target_idx = w - 1 + h
        out_records = [dict(r, predictedAqi=None, modelUsed=resolved_name) for r in records]
        if n_total <= start_target_idx:
            return out_records

        seq_list = []
        target_positions = []
        for target_i in range(start_target_idx, n_total):
            end_i = target_i - h
            seq_list.append(scaled_all[end_i - w + 1 : end_i + 1])
            target_positions.append(target_i)

        x_seq_batch = np.asarray(seq_list, dtype=np.float32)
        preds = self._predict_batch_loaded(horizon_models[resolved_name], x_seq_batch)
        for pos, pred_val in zip(target_positions, preds):
            out_records[pos]["predictedAqi"] = int(pred_val)

        return out_records

    def predict_all_models(self, records: list[dict], horizon: int = 1) -> dict[str, int]:
        """Return predictions across all loaded models for comparison."""
        h = 6 if int(horizon) == 6 else 1
        horizon_models = self._models_by_horizon.get(h, {})
        x_seq = self.prepare_window_tensor(records)
        x_flat = x_seq.reshape(1, -1)
        out = {}
        for name, entry in horizon_models.items():
            if name.startswith("_"):
                continue
            val, _ = self._predict_single_loaded(entry, x_seq, x_flat)
            out[name] = int(round(val))
        return out

