import optuna
import os
import torch
import torch.nn as nn
from src.model import LSTMAQIModel
from src.train import train_model
import src.config as config

def objective(trial, tr_ld, va_ld, device):
    # Reset seed at the beginning of each trial to ensure consistent initialization and shuffling
    config.set_seed(config.RANDOM_SEED)

    # 1. Sample Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # 2. Initialize Model
    model = LSTMAQIModel(input_size=len(config.FEATURE_COLS), hidden_size=hidden_size, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    
    # We use weight_decay in Adam to help regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 3. Train the model (silent during tuning, using temp checkpoint)
    checkpoint_path = f'_optuna_trial_{trial.number}.pt'
    try:
        trained_model, history = train_model(
            model=model, 
            train_loader=tr_ld, 
            val_loader=va_ld, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            epochs=config.EPOCHS, 
            patience=10,
            checkpoint_path=checkpoint_path,
            verbose=False,
            trial=trial
        )
    finally:
        # Clean up temporary checkpoint
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass
    
    # 4. Return the best validation loss for Optuna to minimize
    best_val_loss = min(history["val_loss"])
    return best_val_loss
