import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import src.config as config
from src.train import evaluate_model

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
