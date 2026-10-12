import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import src.config as config
from src.data_loader import get_dataloaders, get_xgboost_data
from src.evaluate import calculate_peak_metrics, inverse_scale_predictions
from src.hybrid_ensemble import (
    AsymmetricWeightedMSE,
    calculate_sample_weights,
    fit_soft_gate,
    make_gating_context,
    xgboost_asymmetric_metric,
    xgboost_asymmetric_objective,
)
from src.model import GRUAQIModel, StackedLSTMAQIModel
from src.train import train_model
from src.utils import DualLogger


def _collect_model_predictions(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for features, labels in loader:
            predictions.append(model(features.to(device)).cpu().numpy().reshape(-1))
            targets.append(labels.numpy().reshape(-1))
    return np.concatenate(predictions), np.concatenate(targets)


def _evaluate_forecast(name, scaled_predictions, scaled_targets, scaler, timestamps):
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    predictions = inverse_scale_predictions(
        scaler, np.asarray(scaled_predictions).reshape(-1, 1), target_idx
    )
    targets = inverse_scale_predictions(
        scaler, np.asarray(scaled_targets).reshape(-1, 1), target_idx
    )
    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    mae = float(mean_absolute_error(targets, predictions))
    threshold = config.PEAK_AQI_THRESHOLD
    severity_span = max(float(scaler.data_max_[target_idx] - threshold), 1e-6)
    weights = calculate_sample_weights(targets, threshold, severity_span)
    weights *= np.where(predictions < targets, 3.0, 1.0)
    amse = float(np.mean(weights * np.square(predictions - targets)))
    metrics = calculate_peak_metrics(
        targets, predictions, threshold, timestamps=timestamps
    )
    print(
        f"{name:22s} RMSE={rmse:8.3f} MAE={mae:8.3f} "
        f"AMSE={amse:9.3f} peak_recall={metrics['recall']:.4f} "
        f"peak_F1={metrics['f1']:.4f} underforecast_rate="
        f"{np.mean(predictions < targets):.4f}"
    )


def main():
    os.makedirs("results", exist_ok=True)
    with DualLogger("results/hybrid_ensemble_results.txt"):
        config.set_seed(config.RANDOM_SEED)
        device = config.DEVICE
        target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
        target_scale = target_idx

        train_loader, val_loader, test_loader, scaler = get_dataloaders(
            data_dir="Dataset/archive"
        )
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            xgb_scaler,
            _train_timestamps,
            _val_timestamps,
            test_timestamps,
        ) = get_xgboost_data(
            data_dir="Dataset/archive", include_split_timestamps=True
        )
        if not np.array_equal(scaler.data_min_, xgb_scaler.data_min_) or not np.array_equal(
            scaler.data_max_, xgb_scaler.data_max_
        ):
            raise RuntimeError("Neural and XGBoost splits produced different scalers.")

        threshold = config.PEAK_AQI_THRESHOLD * scaler.scale_[target_scale] + scaler.min_[target_scale]
        severity_span = max(
            float((scaler.data_max_[target_scale] - config.PEAK_AQI_THRESHOLD) * scaler.scale_[target_scale]),
            1e-6,
        )
        criterion = AsymmetricWeightedMSE(
            threshold=threshold,
            severity_span=severity_span,
            high_aqi_weight=3.0,
            underprediction_penalty=2.0,
        )

        print("\nTraining sample-weighted XGBoost with asymmetric objective...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_params = {
            "verbosity": 0,
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": config.RANDOM_SEED,
        }
        booster = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "validation")],
            obj=xgboost_asymmetric_objective(threshold, severity_span),
            custom_metric=xgboost_asymmetric_metric(threshold, severity_span),
            early_stopping_rounds=25,
            verbose_eval=25,
        )
        xgb_predictions = {
            split: booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
            for split, matrix in (("val", dval), ("test", dtest))
        }

        model_specs = (
            ("GRU", GRUAQIModel()),
            ("Stacked_LSTM", StackedLSTMAQIModel(num_layers=2)),
        )
        models = {}
        rnn_predictions = {}
        for name, model in model_specs:
            print(f"\nTraining {name} with AQI-weighted asymmetric MSE...")
            model = model.to(device)
            model, _ = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE),
                device=device,
                epochs=config.EPOCHS,
                patience=10,
                checkpoint_path=f"results/{name.lower()}_amse_best.pt",
            )
            models[name] = model
            rnn_predictions[name] = {
                "val": _collect_model_predictions(model, val_loader, device)[0],
                "test": _collect_model_predictions(model, test_loader, device)[0],
            }

        predictions_by_split = {
            split: np.column_stack(
                (
                    xgb_predictions[split],
                    rnn_predictions["GRU"][split],
                    rnn_predictions["Stacked_LSTM"][split],
                )
            )
            for split in ("val", "test")
        }
        _, y_val_neural = _collect_model_predictions(models["GRU"], val_loader, device)
        _, y_test_neural = _collect_model_predictions(models["GRU"], test_loader, device)
        if not np.allclose(y_val_neural, y_val) or not np.allclose(y_test_neural, y_test):
            raise RuntimeError("Neural and XGBoost labels are not aligned.")

        val_context = make_gating_context(
            X_val.reshape(-1, config.WINDOW_SIZE, len(config.FEATURE_COLS)),
            target_idx,
        )
        test_context = make_gating_context(
            X_test.reshape(-1, config.WINDOW_SIZE, len(config.FEATURE_COLS)),
            target_idx,
        )
        gate = fit_soft_gate(
            predictions_by_split["val"],
            val_context,
            y_val,
            criterion,
            device,
        )
        with torch.no_grad():
            gated_test, gate_weights = gate(
                torch.as_tensor(predictions_by_split["test"], dtype=torch.float32, device=device),
                torch.as_tensor(test_context, dtype=torch.float32, device=device),
            )
        torch.save(
            {
                "state_dict": gate.state_dict(),
                "model_names": ["XGBoost", "GRU", "Stacked_LSTM"],
                "target_index": target_idx,
            },
            "results/hybrid_soft_gate.pt",
        )
        booster.save_model("results/hybrid_xgb.json")

        print("\nTest comparison (AMSE: higher penalty for under-forecasting and high AQI):")
        test_predictions = predictions_by_split["test"]
        persistence = X_test.reshape(
            -1, config.WINDOW_SIZE, len(config.FEATURE_COLS)
        )[:, -1, target_idx]
        _evaluate_forecast("Persistence", persistence, y_test, scaler, test_timestamps)
        _evaluate_forecast(
            "XGBoost (weighted)", test_predictions[:, 0], y_test, scaler, test_timestamps
        )
        _evaluate_forecast(
            "GRU (AMSE)", test_predictions[:, 1], y_test, scaler, test_timestamps
        )
        _evaluate_forecast(
            "Stacked LSTM (AMSE)",
            test_predictions[:, 2],
            y_test,
            scaler,
            test_timestamps,
        )
        _evaluate_forecast(
            "Equal-weight ensemble",
            test_predictions.mean(axis=1),
            y_test,
            scaler,
            test_timestamps,
        )
        _evaluate_forecast(
            "Soft-gated ensemble",
            gated_test.cpu().numpy(),
            y_test,
            scaler,
            test_timestamps,
        )
        print("\nMean soft-gate weights on test [XGBoost, GRU, Stacked LSTM]:")
        print(gate_weights.mean(dim=0).cpu().numpy())
        print("\nGate weights are fitted on the chronological validation split only.")


if __name__ == "__main__":
    main()
