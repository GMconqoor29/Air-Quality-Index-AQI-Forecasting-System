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

from src.model import StackedLSTMAQIModel

def stacked_objective(trial, tr_ld, va_ld, device):
    config.set_seed(config.RANDOM_SEED)

    # 1. Sample Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    # Also tune the number of layers
    num_layers = trial.suggest_int("num_layers", 2, 4)
    
    # 2. Initialize Model
    model = StackedLSTMAQIModel(input_size=len(config.FEATURE_COLS), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    checkpoint_path = f'_optuna_trial_stacked_{trial.number}.pt'
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
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass
    
    return min(history["val_loss"])

from src.model import GRUAQIModel

def gru_objective(trial, tr_ld, va_ld, device):
    config.set_seed(config.RANDOM_SEED)

    # 1. Sample Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # 2. Initialize Model
    model = GRUAQIModel(input_size=len(config.FEATURE_COLS), hidden_size=hidden_size, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    checkpoint_path = f'_optuna_trial_gru_{trial.number}.pt'
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
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass
    
    return min(history["val_loss"])


import xgboost as xgb
from sklearn.metrics import mean_squared_error

def xgb_objective(trial, X_tr, y_tr, X_va, y_va):
    # Setup hyperparameters for XGBoost
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9)
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)

    try:
        from optuna.integration import XGBoostPruningCallback
        pruning_callback = XGBoostPruningCallback(trial, "validation-rmse")
        callbacks = [pruning_callback]
    except ImportError:
        callbacks = []
    
    bst = xgb.train(
        param, 
        dtrain, 
        num_boost_round=100,
        evals=[(dvalid, "validation")], 
        callbacks=callbacks,
        early_stopping_rounds=10,
        verbose_eval=False
    )

    preds = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
    mse = mean_squared_error(y_va, preds)
    return mse
