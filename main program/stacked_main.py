import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import src.config as config
from src.data_loader import get_dataloaders
from src.model import StackedLSTMAQIModel
from src.train import train_model
from src.evaluate import print_evaluation_metrics, print_baseline_metrics
from src.utils import DualLogger

def main():
    # Setup logger to save all prints to a text file
    with DualLogger("results/stacked_results.txt") as logger:
        # 1. Setup Configuration & Device
        config.set_seed(config.RANDOM_SEED)
        print(f"Using device: {config.DEVICE}")

        # 2. Load Data
        print("\nLoading and preprocessing data...")
        tr_ld, va_ld, te_ld, scaler = get_dataloaders(data_dir='Dataset/archive')

        # 3. Initialize Model, Loss, Optimizer
        print("\nInitializing Stacked LSTM model (num_layers=2)...")
        model = StackedLSTMAQIModel(num_layers=2).to(config.DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # 4. Train Model
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
            checkpoint_path="results/stacked_best_model.pt"
        )

        from src.visualize import plot_loss_curve
        plot_loss_curve(history, save_path="results/stacked_loss_curve.png")

        # 5. Evaluate Best Model
        print("\nEvaluating best model...")
        print_evaluation_metrics(model, tr_ld, criterion, config.DEVICE, scaler, prefix="Train")
        print_evaluation_metrics(model, va_ld, criterion, config.DEVICE, scaler, prefix="Val")
        print_evaluation_metrics(model, te_ld, criterion, config.DEVICE, scaler, prefix="Stacked_Test", plot=True, plot_save_path="results/stacked_test_predictions.png")
        
        print("\n--- Naive Baseline Comparison ---")
        print_baseline_metrics(te_ld, scaler, prefix="Test")

if __name__ == "__main__":
    main()
