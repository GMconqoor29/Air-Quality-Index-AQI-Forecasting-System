import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
import src.config as config
from src.data_loader import get_xgboost_data, get_dataloaders
from src.evaluate import print_baseline_metrics, print_xgb_evaluation_metrics
from src.utils import DualLogger

def main():
    with DualLogger("results/xgb_results.txt") as logger:
        config.set_seed(config.RANDOM_SEED)

        print("Loading and flattening data for XGBoost...")
        X_tr, y_tr, X_va, y_va, X_te, y_te, scaler = get_xgboost_data(data_dir='Dataset/archive')

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_te, label=y_te)

        print("\nInitializing XGBoost model...")
        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "seed": config.RANDOM_SEED
        }

        print("\nStarting training...")
        bst = xgb.train(
            param, 
            dtrain, 
            num_boost_round=200,
            evals=[(dtrain, "train"), (dvalid, "validation")], 
            early_stopping_rounds=10,
            verbose_eval=10
        )

        print("\nEvaluating best model...")
        print_xgb_evaluation_metrics(bst, dtrain, y_tr, scaler, prefix="Train")
        print_xgb_evaluation_metrics(bst, dvalid, y_va, scaler, prefix="Val")
        print_xgb_evaluation_metrics(bst, dtest, y_te, scaler, prefix="Test", plot=True, plot_save_path="results/xgb_test_predictions.png")

        print("\n--- Naive Baseline Comparison ---")
        _, _, te_ld, _ = get_dataloaders(data_dir='Dataset/archive')
        print_baseline_metrics(te_ld, scaler, prefix="Test")
        
        model_path = 'results/best_xgb_model.json'
        bst.save_model(model_path)
        print(f"\nSaved {model_path}!")

if __name__ == "__main__":
    main()
