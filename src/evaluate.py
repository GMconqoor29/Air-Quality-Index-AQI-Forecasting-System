import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import src.config as config
from src.train import evaluate_model


def calculate_peak_metrics(actual, predicted, threshold, timestamps=None):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("actual and predicted must have the same non-zero length.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("actual and predicted must contain only finite values.")

    actual_peak = actual >= threshold
    predicted_peak = predicted >= threshold
    true_positive = int(np.count_nonzero(actual_peak & predicted_peak))
    false_positive = int(np.count_nonzero(~actual_peak & predicted_peak))
    false_negative = int(np.count_nonzero(actual_peak & ~predicted_peak))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if actual_peak.any():
        peak_rmse = float(
            np.sqrt(mean_squared_error(actual[actual_peak], predicted[actual_peak]))
        )
        peak_mae = float(mean_absolute_error(actual[actual_peak], predicted[actual_peak]))
    else:
        peak_rmse = float("nan")
        peak_mae = float("nan")

    false_alarms_per_week = None
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype="datetime64[ns]").reshape(-1)
        if timestamps.size != actual.size:
            raise ValueError("timestamps must have the same length as actual.")
        if np.isnat(timestamps).any():
            raise ValueError("timestamps must not contain NaT values.")
        elapsed_hours = (
            (timestamps.max() - timestamps.min()).astype("timedelta64[ns]").astype(np.int64)
            / 3_600_000_000_000
            + 1
        )
        false_alarms_per_week = false_positive / (elapsed_hours / 168)

    return {
        "threshold": float(threshold),
        "actual_peak_count": int(np.count_nonzero(actual_peak)),
        "predicted_peak_count": int(np.count_nonzero(predicted_peak)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "peak_rmse": peak_rmse,
        "peak_mae": peak_mae,
        "false_alarms_per_week": false_alarms_per_week,
    }


def print_peak_metrics(actual, predicted, threshold, timestamps=None, prefix="Test"):
    metrics = calculate_peak_metrics(actual, predicted, threshold, timestamps)
    print(f"{prefix} AQI >= {threshold:g} Peak Metrics:")
    print(
        f"  Actual peaks: {metrics['actual_peak_count']} | "
        f"Forecast peaks: {metrics['predicted_peak_count']}"
    )
    print(
        f"  Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
    )
    print(
        f"  Peak RMSE: {metrics['peak_rmse']:.4f} | "
        f"Peak MAE: {metrics['peak_mae']:.4f}"
    )
    print(
        f"  False positives: {metrics['false_positive']} | "
        f"Missed peaks: {metrics['false_negative']}"
    )
    false_alarms_per_week = metrics["false_alarms_per_week"]
    if false_alarms_per_week is not None:
        print(f"  False forecast alerts per week: {false_alarms_per_week:.2f}")
    return metrics


def inverse_scale_predictions(scaler, preds, target_idx=0):
    """
    Because the scaler was fit on all 13 features, we must reconstruct 
    a dummy array of shape (N, 13) to use inverse_transform.
    """
    dummy = np.zeros((len(preds), len(config.FEATURE_COLS)))
    dummy[:, target_idx] = preds.flatten()
    inv_dummy = scaler.inverse_transform(dummy)
    return inv_dummy[:, target_idx]

def print_evaluation_metrics(model, loader, criterion, device, scaler, prefix="Test", plot=False, plot_save_path=None):
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    
    # Get scaled predictions
    loss, preds, y_true = evaluate_model(model, loader, criterion, device)
    
    # Unscale predictions and true values
    preds_unscaled = inverse_scale_predictions(scaler, preds, target_idx)
    y_true_unscaled = inverse_scale_predictions(scaler, y_true, target_idx)
    
    # Calculate metrics
    mse = mean_squared_error(y_true_unscaled, preds_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_unscaled, preds_unscaled)
    
    print(f"{prefix} Scaled MSE: {loss:.4f}")
    print(f"{prefix} Unscaled RMSE: {rmse:.4f}")
    print(f"{prefix} Unscaled MAE:  {mae:.4f}")
    
    if plot:
        from src.visualize import plot_predictions
        if plot_save_path is None:
            plot_save_path = f"results/{prefix.lower()}_predictions.png"
        plot_predictions(
            y_true_unscaled, 
            preds_unscaled, 
            title=f"{prefix} Set: Actual vs Predicted AQI (t+{config.HORIZON})", 
            save_path=plot_save_path
        )
    
    return loss, rmse, mae

def print_baseline_metrics(loader, scaler, prefix="Test"):
    """
    Calculates the Persistence Baseline: "The AQI at t+6 will be exactly the AQI at time t."
    """
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    all_naive, all_y = [], []
    
    for X, y in loader:
        # X shape is (batch, window, features). We want the last timestep's AQI.
        naive_pred = X[:, -1, target_idx].numpy()
        all_naive.append(naive_pred)
        all_y.append(y.numpy())
        
    preds = np.concatenate(all_naive).reshape(-1, 1)
    y_true = np.concatenate(all_y).reshape(-1, 1)
    
    preds_unscaled = inverse_scale_predictions(scaler, preds, target_idx)
    y_true_unscaled = inverse_scale_predictions(scaler, y_true, target_idx)
    
    mse = mean_squared_error(y_true_unscaled, preds_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_unscaled, preds_unscaled)
    
    print(f"{prefix} BASELINE (Persistence) Unscaled RMSE: {rmse:.4f}")
    print(f"{prefix} BASELINE (Persistence) Unscaled MAE:  {mae:.4f}")
    
    return rmse, mae

def print_xgb_evaluation_metrics(
    bst,
    dmatrix,
    y_true,
    scaler,
    prefix="Test",
    plot=False,
    plot_save_path=None,
    timestamps=None,
    peak_threshold=config.PEAK_AQI_THRESHOLD,
):
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    
    # Get scaled predictions
    preds = bst.predict(dmatrix)
    loss = mean_squared_error(y_true, preds)
    
    # Unscale predictions and true values
    preds_unscaled = inverse_scale_predictions(scaler, preds.reshape(-1, 1), target_idx)
    y_true_unscaled = inverse_scale_predictions(scaler, y_true.reshape(-1, 1), target_idx)
    
    # Calculate metrics
    mse = mean_squared_error(y_true_unscaled, preds_unscaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_unscaled, preds_unscaled)
    
    print(f"{prefix} Scaled MSE: {loss:.4f}")
    print(f"{prefix} Unscaled RMSE: {rmse:.4f}")
    print(f"{prefix} Unscaled MAE:  {mae:.4f}")
    print_peak_metrics(
        y_true_unscaled,
        preds_unscaled,
        peak_threshold,
        timestamps=timestamps,
        prefix=prefix,
    )
    
    if plot:
        from src.visualize import plot_predictions
        if plot_save_path is None:
            plot_save_path = f"results/{prefix.lower()}_predictions.png"
        plot_predictions(
            y_true_unscaled, 
            preds_unscaled, 
            title=f"{prefix} Set: Actual vs Predicted AQI (t+{config.HORIZON})", 
            save_path=plot_save_path
        )
    
    return loss, rmse, mae
