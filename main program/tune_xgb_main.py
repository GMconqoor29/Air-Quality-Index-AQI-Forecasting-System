import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import xgboost as xgb
import numpy as np
from datetime import datetime
import src.config as config
from src.data_loader import get_xgboost_data, get_dataloaders
from src.tune import xgb_objective
from src.evaluate import inverse_scale_predictions, print_baseline_metrics, print_xgb_evaluation_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils import DualLogger

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    with DualLogger(f"results/tuned_xgb_results_{run_id}.txt") as logger:
        config.set_seed(config.RANDOM_SEED)
        print("Loading data for XGBoost tuning...")
        X_tr_flat, y_tr, X_va_flat, y_va, X_te_flat, y_te, scaler = get_xgboost_data(data_dir='Dataset/archive')
        
        print("\nStarting Optuna Hyperparameter Optimization for XGBoost...")
        study = optuna.create_study(
            direction="minimize", 
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(
            lambda trial: xgb_objective(trial, X_tr_flat, y_tr, X_va_flat, y_va), 
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
        best_params["objective"] = "reg:squarederror"
        best_params["eval_metric"] = "rmse"
        best_params["verbosity"] = 0
        
        dtrain = xgb.DMatrix(X_tr_flat, label=y_tr)
        dvalid = xgb.DMatrix(X_va_flat, label=y_va)
        dtest = xgb.DMatrix(X_te_flat, label=y_te)
        
        bst = xgb.train(
            best_params, 
            dtrain, 
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dvalid, "validation")], 
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        print("\nEvaluating Best XGBoost Model...")
        print_xgb_evaluation_metrics(bst, dtrain, y_tr, scaler, prefix="Train")
        print_xgb_evaluation_metrics(bst, dvalid, y_va, scaler, prefix="Val")
        print_xgb_evaluation_metrics(bst, dtest, y_te, scaler, prefix="Test", plot=True, plot_save_path=f"results/tuned_xgb_test_predictions_{run_id}.png")
        
        print("\n--- Naive Baseline Comparison ---")
        _, _, te_ld, _ = get_dataloaders(data_dir='Dataset/archive')
        print_baseline_metrics(te_ld, scaler, prefix="Test")
        
        model_path = f'results/tuned_best_xgb_{run_id}.json'
        bst.save_model(model_path)
        print(f"\nSaved {model_path}!")

if __name__ == "__main__":
    main()
