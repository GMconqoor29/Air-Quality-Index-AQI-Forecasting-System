import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import torch
import torch.nn as nn
from datetime import datetime
import src.config as config
from src.data_loader import get_dataloaders
from src.tune import gru_objective
from src.model import GRUAQIModel
from src.train import train_model
from src.evaluate import print_evaluation_metrics, print_baseline_metrics
from src.utils import DualLogger

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    with DualLogger(f"results/tuned_gru_results_{run_id}.txt") as logger:
        config.set_seed(config.RANDOM_SEED)
        print("Loading data for GRU tuning...")
        tr_ld, va_ld, te_ld, scaler = get_dataloaders(data_dir='Dataset/archive')
        
        print("\nStarting Optuna Hyperparameter Optimization for GRU...")
        study = optuna.create_study(
            direction="minimize", 
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(
            lambda trial: gru_objective(trial, tr_ld, va_ld, config.DEVICE), 
            n_trials=20,
            n_jobs=1
        )
        
        print("\nOptimization Finished!")
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Val MSE: {study.best_trial.value:.4f}")
        print("Best Params:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
            
        print("\nRetraining on Best Parameters...")
        best_params = study.best_trial.params
        
        config.set_seed(config.RANDOM_SEED)
        
        model = GRUAQIModel(
            input_size=len(config.FEATURE_COLS), 
            hidden_size=best_params["hidden_size"], 
            dropout=best_params["dropout"]
        ).to(config.DEVICE)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=best_params["lr"], 
            weight_decay=best_params["weight_decay"]
        )
        
        model, history = train_model(
            model=model, 
            train_loader=tr_ld, 
            val_loader=va_ld, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=config.DEVICE, 
            epochs=config.EPOCHS, 
            patience=10,
            checkpoint_path=f'results/tuned_best_gru_{run_id}.pt'
        )
        
        from src.visualize import plot_loss_curve
        plot_loss_curve(history, save_path=f"results/tuned_gru_loss_{run_id}.png")
        
        print("\nEvaluating Best GRU Model...")
        print_evaluation_metrics(model, tr_ld, criterion, config.DEVICE, scaler, prefix="Train")
        print_evaluation_metrics(model, va_ld, criterion, config.DEVICE, scaler, prefix="Val")
        print_evaluation_metrics(model, te_ld, criterion, config.DEVICE, scaler, prefix="Test", plot=True, plot_save_path=f"results/tuned_gru_test_predictions_{run_id}.png")
        
        print("\n--- Naive Baseline Comparison ---")
        print_baseline_metrics(te_ld, scaler, prefix="Test")
        
        print(f"\nSaved results/tuned_best_gru_{run_id}.pt!")

if __name__ == "__main__":
    main()
