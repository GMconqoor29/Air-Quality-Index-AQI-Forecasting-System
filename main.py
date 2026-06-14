import torch
import torch.nn as nn
import src.config as config
from src.data_loader import get_dataloaders
from src.model import LSTMAQIModel
from src.train import train_model
from src.evaluate import print_evaluation_metrics, print_baseline_metrics
from src.utils import DualLogger

def main():
    # Setup logger to save all prints to a text file
    with DualLogger("results/base_results.txt") as logger:
        # 1. Setup Configuration & Device
        config.set_seed(config.RANDOM_SEED)
        print(f"Using device: {config.DEVICE}")

        # 2. Load Data
        print("\nLoading and preprocessing data...")
        tr_ld, va_ld, te_ld, scaler = get_dataloaders(data_dir='Dataset/archive')

        # 3. Initialize Model, Loss, Optimizer
        print("\nInitializing model...")
        model = LSTMAQIModel().to(config.DEVICE)
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
            patience=5
        )

        from src.visualize import plot_loss_curve
        plot_loss_curve(history, save_path="results/base_loss_curve.png")

        # 5. Evaluate Best Model
        print("\nEvaluating best model...")
        print_evaluation_metrics(model, tr_ld, criterion, config.DEVICE, scaler, prefix="Train")
        print_evaluation_metrics(model, va_ld, criterion, config.DEVICE, scaler, prefix="Val")
        print_evaluation_metrics(model, te_ld, criterion, config.DEVICE, scaler, prefix="Base_Test", plot=True)
        
        print("\n--- Naive Baseline Comparison ---")
        print_baseline_metrics(te_ld, scaler, prefix="Test")

if __name__ == "__main__":
    main()

