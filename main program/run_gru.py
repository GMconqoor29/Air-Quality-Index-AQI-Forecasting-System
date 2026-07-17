import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import src.config as config
from src.data_loader import get_dataloaders
from src.model import GRUAQIModel
from src.train import train_model
from src.evaluate import print_evaluation_metrics, print_baseline_metrics
from src.utils import DualLogger

def main():
    with DualLogger("results/gru_results.txt") as logger:
        config.set_seed(config.RANDOM_SEED)
        print(f"Using device: {config.DEVICE}")

        print("\nLoading and preprocessing data...")
        tr_ld, va_ld, te_ld, scaler = get_dataloaders(data_dir='Dataset/archive')

        print("\nInitializing GRU model...")
        model = GRUAQIModel().to(config.DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        print("\nStarting training...")
        model, history = train_model(
            model=model, 
            train_loader=tr_ld, 
            val_loader=va_ld, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=config.DEVICE, 
            epochs=config.EPOCHS, 
            patience=5,
            checkpoint_path="best_gru_model.pt"
        )

        from src.visualize import plot_loss_curve
        plot_loss_curve(history, save_path="results/gru_loss_curve.png")

        print("\nEvaluating best model...")
        print_evaluation_metrics(model, tr_ld, criterion, config.DEVICE, scaler, prefix="Train", plot=False)
        print_evaluation_metrics(model, va_ld, criterion, config.DEVICE, scaler, prefix="Val", plot=False)
        print_evaluation_metrics(model, te_ld, criterion, config.DEVICE, scaler, prefix="Test", plot=True, plot_save_path="results/gru_test_predictions.png")

if __name__ == "__main__":
    main()
