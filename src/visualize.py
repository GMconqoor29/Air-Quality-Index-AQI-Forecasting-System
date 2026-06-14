import matplotlib.pyplot as plt

def plot_loss_curve(history, save_path="loss_curve.png"):
    """
    Plots the training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss (MSE)', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss (MSE)', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to {save_path}")

def plot_predictions(y_true, y_pred, title="AQI Predictions (t+6)", save_path="predictions.png", num_samples=200):
    """
    Plots actual vs predicted AQI values.
    num_samples limits the plot to the first N time steps for readability.
    """
    plt.figure(figsize=(15, 6))
    
    # We only plot a slice to ensure the graph isn't too cluttered
    actual_slice = y_true[:num_samples]
    pred_slice = y_pred[:num_samples]
    
    plt.plot(actual_slice, label='Actual AQI', color='blue', marker='.', linewidth=1.5, zorder=1)
    plt.plot(pred_slice, label='Predicted AQI', color='orange', marker='.', linewidth=1.5, alpha=0.8, zorder=2)
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('AQI')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved prediction plot to {save_path}")
