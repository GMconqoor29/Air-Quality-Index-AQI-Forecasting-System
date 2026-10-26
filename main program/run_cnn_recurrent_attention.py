import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

import src.config as config
from src.data_loader import get_dataloaders, get_xgboost_data
from src.evaluate import calculate_peak_metrics, inverse_scale_predictions
from src.hybrid_ensemble import AsymmetricWeightedMSE, calculate_sample_weights
from src.model import CNNRecurrentAttentionAQIModel
from src.train import train_model
from src.utils import DualLogger


def collect_predictions(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for features, labels in loader:
            predictions.append(model(features.to(device)).cpu().numpy().reshape(-1))
            targets.append(labels.numpy().reshape(-1))
    return np.concatenate(predictions), np.concatenate(targets)


def report_metrics(name, predictions, targets, scaler, timestamps):
    target_idx = (
        config.FEATURE_COLS.index(config.TARGET_COL)
        if config.TARGET_COL in config.FEATURE_COLS
        else 0
    )
    predictions = inverse_scale_predictions(
        scaler, np.asarray(predictions).reshape(-1, 1), target_idx
    )
    targets = inverse_scale_predictions(
        scaler, np.asarray(targets).reshape(-1, 1), target_idx
    )
    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    mae = float(mean_absolute_error(targets, predictions))
    threshold = config.PEAK_AQI_THRESHOLD
    target_max = (
        float(scaler.target_scaler_.data_max_[0])
        if hasattr(scaler, "target_scaler_")
        else float(scaler.data_max_[target_idx])
    )
    severity_span = max(float(target_max - threshold), 1e-6)
    weights = calculate_sample_weights(targets, threshold, severity_span)
    weights *= np.where(predictions < targets, 3.0, 1.0)
    amse = float(np.mean(weights * np.square(predictions - targets)))
    peaks = calculate_peak_metrics(targets, predictions, threshold, timestamps)
    print(
        f"{name:30s} RMSE={rmse:8.3f} MAE={mae:8.3f} "
        f"AMSE={amse:9.3f} peak_recall={peaks['recall']:.4f} "
        f"peak_F1={peaks['f1']:.4f} "
        f"underforecast_rate={np.mean(predictions < targets):.4f}"
    )


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with DualLogger(os.path.join(config.RESULTS_DIR, "cnn_recurrent_attention_results.txt")):
        config.set_seed(config.RANDOM_SEED)
        device = config.DEVICE
        train_loader, val_loader, test_loader, scaler = get_dataloaders(
            data_dir="Dataset/archive"
        )
        (
            _X_train,
            _y_train,
            _X_val,
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

        if hasattr(scaler, "target_scaler_"):
            target_scale = float(scaler.target_scaler_.scale_[0])
            target_min = float(scaler.target_scaler_.min_[0])
            target_max = float(scaler.target_scaler_.data_max_[0])
        else:
            target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
            target_scale = float(scaler.scale_[target_idx])
            target_min = float(scaler.min_[target_idx])
            target_max = float(scaler.data_max_[target_idx])

        threshold = (
            config.PEAK_AQI_THRESHOLD * target_scale
            + target_min
        )
        severity_span = max(
            float(
                (target_max - config.PEAK_AQI_THRESHOLD)
                * target_scale
            ),
            1e-6,
        )
        criterion = AsymmetricWeightedMSE(
            threshold=threshold,
            severity_span=severity_span,
            high_aqi_weight=3.0,
            underprediction_penalty=2.0,
        )

        trained_models = {}
        for recurrent_type in ("GRU", "LSTM"):
            config.set_seed(config.RANDOM_SEED)
            model_name = f"CNN-{recurrent_type}+Attention"
            print(f"\nTraining {model_name} with AMSE...")
            model = CNNRecurrentAttentionAQIModel(
                recurrent_type=recurrent_type
            ).to(device)
            model, _ = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=torch.optim.Adam(
                    model.parameters(), lr=config.LEARNING_RATE
                ),
                device=device,
                epochs=config.EPOCHS,
                patience=10,
                checkpoint_path=os.path.join(
                    config.RESULTS_DIR,
                    f"cnn_{recurrent_type.lower()}_attention_best.pt"
                ),
            )
            trained_models[model_name] = model

        _, y_test_neural = collect_predictions(
            trained_models["CNN-GRU+Attention"], test_loader, device
        )
        if not np.allclose(y_test_neural, y_test):
            raise RuntimeError("Neural and XGBoost test labels are not aligned.")
        _, y_val_neural = collect_predictions(
            trained_models["CNN-GRU+Attention"], val_loader, device
        )
        if not np.allclose(y_val_neural, y_val):
            raise RuntimeError("Neural and XGBoost validation labels are not aligned.")

        if config.TARGET_COL in config.FEATURE_COLS:
            target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
            persistence = X_test.reshape(
                -1, config.WINDOW_SIZE, len(config.FEATURE_COLS)
            )[:, -1, target_idx]
        else:
            import glob
            import pandas as pd
            df_raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('Dataset/archive/*.csv'))], ignore_index=True)
            df_raw['Local Time'] = pd.to_datetime(df_raw['Local Time'])
            df_raw = df_raw.sort_values('Local Time').drop_duplicates('Local Time').set_index('Local Time').asfreq('h')
            df_raw.loc[df_raw['AQI'] > 400, 'AQI'] = np.nan
            past_times = pd.DatetimeIndex(test_timestamps) - pd.Timedelta(hours=config.HORIZON)
            persist_raw = df_raw.reindex(past_times)['AQI'].ffill().bfill().to_numpy(dtype=np.float64)
            persistence = scaler.target_scaler_.transform(persist_raw.reshape(-1, 1)).reshape(-1)

        print("\nTest comparison (same chronological holdout and AMSE definition):")
        report_metrics("Persistence", persistence, y_test, scaler, test_timestamps)
        for name, model in trained_models.items():
            predictions, targets = collect_predictions(model, test_loader, device)
            report_metrics(name, predictions, targets, scaler, test_timestamps)


if __name__ == "__main__":
    main()
