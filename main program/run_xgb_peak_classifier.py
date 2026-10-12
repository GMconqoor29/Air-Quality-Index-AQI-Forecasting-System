"""Train and evaluate a six-hour AQI peak classifier."""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb

import src.config as config
from src.data_loader import get_xgboost_data
from src.evaluate import inverse_scale_predictions
from src.peak_classifier import (
    calculate_classifier_metrics,
    select_recall_thresholds,
)
from src.utils import DualLogger


RECALL_TARGETS = (0.50, 0.75, 0.90, 0.95, 0.99, 1.00)


def _raw_aqi(scaler, values):
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    return inverse_scale_predictions(scaler, values.reshape(-1, 1), target_idx)


def _labels(scaler, values):
    return (_raw_aqi(scaler, values) >= config.PEAK_AQI_THRESHOLD).astype(np.int32)


def _print_split_summary(name, labels, probabilities, timestamps):
    average_precision = calculate_classifier_metrics(
        labels, probabilities, threshold=0.5, timestamps=timestamps
    )["average_precision"]
    print(
        f"{name}: {int(labels.sum())}/{len(labels)} peaks "
        f"({labels.mean():.2%}); PR average precision: {average_precision:.4f}"
    )


def _print_threshold_metrics(name, labels, probabilities, timestamps, thresholds):
    print(f"\n{name} metrics at validation-selected thresholds:")
    print(
        "  Val target | Cutoff  | Recall | Precision | F1     | "
        "TP | FP | Missed | Forecasts | FP/week"
    )
    for recall_target, threshold in thresholds.items():
        metrics = calculate_classifier_metrics(
            labels, probabilities, threshold, timestamps
        )
        fp_per_week = metrics["false_alerts_per_week"]
        fp_text = f"{fp_per_week:.2f}" if fp_per_week is not None else "n/a"
        print(
            f"  {recall_target:>9.0%} | {threshold:>7.4f} | "
            f"{metrics['recall']:>6.1%} | {metrics['precision']:>9.1%} | "
            f"{metrics['f1']:>6.3f} | {metrics['true_positive']:>2} | "
            f"{metrics['false_positive']:>2} | {metrics['false_negative']:>6} | "
            f"{metrics['forecast_peaks']:>9} | {fp_text:>8}"
        )


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    with DualLogger(f"results/xgb_peak_classifier_{run_id}.txt"):
        config.set_seed(config.RANDOM_SEED)
        (
            X_train,
            y_train_scaled,
            X_val,
            y_val_scaled,
            X_test,
            y_test_scaled,
            scaler,
            train_timestamps,
            val_timestamps,
            test_timestamps,
        ) = get_xgboost_data(
            data_dir="Dataset/archive", include_split_timestamps=True
        )

        y_train = _labels(scaler, y_train_scaled)
        y_val = _labels(scaler, y_val_scaled)
        y_test = _labels(scaler, y_test_scaled)
        positive_count = int(y_train.sum())
        negative_count = len(y_train) - positive_count
        if positive_count == 0 or negative_count == 0:
            raise ValueError("Training data must contain both peak and non-peak labels.")

        print(
            f"\nTraining AQI >= {config.PEAK_AQI_THRESHOLD:g} classifier "
            f"(6-hour forecast horizon)..."
        )
        print(
            f"Training labels: {positive_count} peaks, {negative_count} non-peaks; "
            f"scale_pos_weight={negative_count / positive_count:.3f}"
        )
        for name, labels in (
            ("Train", y_train),
            ("Validation", y_val),
            ("Test", y_test),
        ):
            if labels.sum() == 0 or labels.sum() == len(labels):
                raise ValueError(
                    f"{name} split must contain both peak and non-peak labels."
                )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            {
                "verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "booster": "gbtree",
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 2,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": negative_count / positive_count,
                "tree_method": "hist",
                "seed": config.RANDOM_SEED,
            },
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dvalid, "validation")],
            early_stopping_rounds=30,
            verbose_eval=50,
        )

        best_iteration = model.best_iteration
        predict_options = {"iteration_range": (0, best_iteration + 1)}
        train_probabilities = model.predict(dtrain, **predict_options)
        val_probabilities = model.predict(dvalid, **predict_options)
        dtest = xgb.DMatrix(X_test)
        test_probabilities = model.predict(dtest, **predict_options)

        print("\nPeak prevalence and ranking quality:")
        _print_split_summary(
            "Train", y_train, train_probabilities, train_timestamps
        )
        _print_split_summary("Validation", y_val, val_probabilities, val_timestamps)
        _print_split_summary("Test", y_test, test_probabilities, test_timestamps)

        thresholds = select_recall_thresholds(
            y_val, val_probabilities, RECALL_TARGETS
        )
        print(
            "\nThresholds below are selected using validation recall only; "
            "test metrics are held out."
        )
        _print_threshold_metrics(
            "Validation", y_val, val_probabilities, val_timestamps, thresholds
        )
        _print_threshold_metrics(
            "Test", y_test, test_probabilities, test_timestamps, thresholds
        )

        model_path = f"results/xgb_peak_classifier_{run_id}.json"
        model.save_model(model_path)
        print(f"\nSaved classifier to {model_path}")


if __name__ == "__main__":
    main()
