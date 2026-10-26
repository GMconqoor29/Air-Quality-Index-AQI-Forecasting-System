"""Walk-forward validation and two-stage AQI peak regression experiment."""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import src.config as config
from src.data_loader import get_xgboost_data
from src.evaluate import calculate_peak_metrics, inverse_scale_predictions
from src.peak_classifier import (
    calculate_classifier_metrics,
    select_recall_thresholds,
)
from src.utils import DualLogger
from src.walk_forward import rolling_origin_splits

FOLD_COUNT = 4
INITIAL_TRAIN_FRACTION = 0.5
RECALL_TARGETS = (0.50, 0.75, 0.90)


def _raw_aqi(scaler, values):
    target_idx = (
        config.FEATURE_COLS.index(config.TARGET_COL)
        if config.TARGET_COL in config.FEATURE_COLS
        else 0
    )
    return inverse_scale_predictions(scaler, np.asarray(values).reshape(-1, 1), target_idx)


def _peak_labels(scaler, values):
    return (_raw_aqi(scaler, values) >= config.PEAK_AQI_THRESHOLD).astype(np.int32)


def _train_fold_models(X_fit, y_fit, X_valid, y_valid, threshold_scaled, seed):
    labels = (y_fit >= threshold_scaled).astype(np.int32)
    positive_count = int(labels.sum())
    negative_count = labels.size - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Each rolling training fold must contain peak and non-peak targets.")
    valid_labels = (y_valid >= threshold_scaled).astype(np.int32)
    if valid_labels.sum() == 0:
        raise ValueError(
            "A rolling validation fold has no high-AQI peaks; cannot assess peak recall."
        )

    dfit = xgb.DMatrix(X_fit, label=labels)
    dvalid_classifier = xgb.DMatrix(X_valid, label=valid_labels)
    classifier = xgb.train(
        {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "booster": "gbtree",
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": negative_count / positive_count,
            "tree_method": "hist",
            "seed": seed,
            "nthread": 4,
        },
        dfit,
        num_boost_round=500,
        evals=[(dvalid_classifier, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    dfit_regression = xgb.DMatrix(X_fit, label=y_fit)
    dvalid_regression = xgb.DMatrix(X_valid, label=y_valid)
    general = xgb.train(
        {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "seed": seed,
            "nthread": 4,
        },
        dfit_regression,
        num_boost_round=500,
        evals=[(dvalid_regression, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    peak_fit = labels == 1
    peak_valid = valid_labels == 1
    dfit_peak = xgb.DMatrix(X_fit[peak_fit], label=y_fit[peak_fit])
    dvalid_peak = xgb.DMatrix(X_valid[peak_valid], label=y_valid[peak_valid])
    peak_expert = xgb.train(
        {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 2,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "tree_method": "hist",
            "seed": seed,
            "nthread": 4,
        },
        dfit_peak,
        num_boost_round=500,
        evals=[(dvalid_peak, "validation_peak")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return classifier, general, peak_expert


def _best_iteration(model):
    return int(model.best_iteration) + 1


def _report_numeric_metrics(name, predictions, targets, scaler, timestamps):
    actual = _raw_aqi(scaler, targets)
    predicted = _raw_aqi(scaler, predictions)
    actual_peaks = actual >= config.PEAK_AQI_THRESHOLD
    if not actual_peaks.any():
        raise ValueError("Cannot report peak regression metrics without actual peaks.")
    errors = predicted[actual_peaks] - actual[actual_peaks]
    peak_classification = calculate_peak_metrics(
        actual, predicted, config.PEAK_AQI_THRESHOLD, timestamps=timestamps
    )
    print(
        f"{name:23s} RMSE={np.sqrt(mean_squared_error(actual, predicted)):8.3f} "
        f"MAE={mean_absolute_error(actual, predicted):8.3f} "
        f"peak_RMSE={np.sqrt(np.mean(errors ** 2)):8.3f} "
        f"peak_MAE={np.mean(np.abs(errors)):8.3f} "
        f"peak_bias={np.mean(errors):8.3f} "
        f"peak_recall={peak_classification['recall']:.4f} "
        f"peak_F1={peak_classification['f1']:.4f}"
    )


def _report_classifier_metrics(name, labels, probabilities, timestamps, threshold):
    metrics = calculate_classifier_metrics(
        labels, probabilities, threshold, timestamps=timestamps
    )
    fp_week = metrics["false_alerts_per_week"]
    print(
        f"{name:23s} cutoff={threshold:.4f} recall={metrics['recall']:.4f} "
        f"precision={metrics['precision']:.4f} F1={metrics['f1']:.4f} "
        f"FP/week={fp_week:.2f}"
    )


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    with DualLogger(os.path.join(config.RESULTS_DIR, f"walk_forward_two_stage_{run_id}.txt")):
        config.set_seed(config.RANDOM_SEED)
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            scaler,
            train_timestamps,
            val_timestamps,
            test_timestamps,
        ) = get_xgboost_data(
            data_dir="Dataset/archive", include_split_timestamps=True
        )
        X_pretest = np.concatenate((X_train, X_val))
        y_pretest = np.concatenate((y_train, y_val))
        pretest_timestamps = np.concatenate((train_timestamps, val_timestamps))
        if not np.all(pretest_timestamps[:-1] <= pretest_timestamps[1:]):
            raise RuntimeError("Pre-test examples must be in chronological order.")

        if hasattr(scaler, "target_scaler_"):
            threshold_scaled = (
                config.PEAK_AQI_THRESHOLD * float(scaler.target_scaler_.scale_[0])
                + float(scaler.target_scaler_.min_[0])
            )
        else:
            target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
            threshold_scaled = (
                config.PEAK_AQI_THRESHOLD * float(scaler.scale_[target_idx])
                + float(scaler.min_[target_idx])
            )
        labels_pretest = _peak_labels(scaler, y_pretest)
        print(
            f"\nWalk-forward region: {pretest_timestamps[0]} through "
            f"{pretest_timestamps[-1]}; peaks={labels_pretest.sum()}/{len(labels_pretest)}"
        )
        print(
            f"Test remains untouched: {test_timestamps[0]} through "
            f"{test_timestamps[-1]}; peaks={_peak_labels(scaler, y_test).sum()}/{len(y_test)}"
        )

        folds = rolling_origin_splits(
            len(X_pretest),
            fold_count=FOLD_COUNT,
            initial_fraction=INITIAL_TRAIN_FRACTION,
        )
        out_of_fold = []
        best_iterations = []
        for fold_number, (fit_start, valid_start, valid_stop) in enumerate(folds, 1):
            X_fit = X_pretest[fit_start:valid_start]
            y_fit = y_pretest[fit_start:valid_start]
            X_valid = X_pretest[valid_start:valid_stop]
            y_valid = y_pretest[valid_start:valid_stop]
            fold_times = pretest_timestamps[valid_start:valid_stop]
            fold_labels = labels_pretest[valid_start:valid_stop]
            print(
                f"\nFold {fold_number}: fit {pretest_timestamps[fit_start]} to "
                f"{pretest_timestamps[valid_start - 1]} ({len(X_fit)} samples); "
                f"validate {fold_times[0]} to {fold_times[-1]} "
                f"({len(X_valid)} samples, peaks={fold_labels.sum()})"
            )
            classifier, general, expert = _train_fold_models(
                X_fit,
                y_fit,
                X_valid,
                y_valid,
                threshold_scaled,
                config.RANDOM_SEED + fold_number,
            )
            dvalid = xgb.DMatrix(X_valid)
            probabilities = classifier.predict(
                dvalid, iteration_range=(0, _best_iteration(classifier))
            )
            general_predictions = general.predict(
                dvalid, iteration_range=(0, _best_iteration(general))
            )
            expert_predictions = expert.predict(
                dvalid, iteration_range=(0, _best_iteration(expert))
            )
            out_of_fold.append(
                (
                    probabilities,
                    general_predictions,
                    expert_predictions,
                    y_valid,
                    fold_labels,
                    fold_times,
                )
            )
            best_iterations.append(
                (
                    _best_iteration(classifier),
                    _best_iteration(general),
                    _best_iteration(expert),
                )
            )
            _report_numeric_metrics(
                f"Fold {fold_number} general",
                general_predictions,
                y_valid,
                scaler,
                fold_times,
            )

        oof_probabilities = np.concatenate([fold[0] for fold in out_of_fold])
        oof_general = np.concatenate([fold[1] for fold in out_of_fold])
        oof_expert = np.concatenate([fold[2] for fold in out_of_fold])
        oof_targets = np.concatenate([fold[3] for fold in out_of_fold])
        oof_labels = np.concatenate([fold[4] for fold in out_of_fold])
        oof_timestamps = np.concatenate([fold[5] for fold in out_of_fold])
        threshold_map = select_recall_thresholds(
            oof_labels, oof_probabilities, RECALL_TARGETS
        )
        selected_threshold = threshold_map[0.75]
        oof_two_stage = np.where(
            oof_probabilities >= selected_threshold, oof_expert, oof_general
        )

        print("\nPooled rolling-origin validation (thresholds calibrated on OOF predictions):")
        _report_numeric_metrics(
            "General regressor",
            oof_general,
            oof_targets,
            scaler,
            oof_timestamps,
        )
        _report_numeric_metrics(
            "Two-stage, recall 75%",
            oof_two_stage,
            oof_targets,
            scaler,
            oof_timestamps,
        )
        for recall_target, threshold in threshold_map.items():
            _report_classifier_metrics(
                f"Classifier recall {recall_target:.0%}",
                oof_labels,
                oof_probabilities,
                oof_timestamps,
                threshold,
            )
        print(f"\nUsing OOF-calibrated two-stage cutoff={selected_threshold:.6f}")

        final_rounds = np.median(np.asarray(best_iterations), axis=0).astype(int)
        print(f"Final fixed boosting rounds [classifier, general, peak expert]: {final_rounds.tolist()}")
        final_labels = labels_pretest
        positive_count = int(final_labels.sum())
        negative_count = len(final_labels) - positive_count
        if positive_count == 0 or negative_count == 0:
            raise ValueError("Final pre-test data must contain peak and non-peak targets.")
        dpretest_classifier = xgb.DMatrix(X_pretest, label=final_labels)
        final_classifier = xgb.train(
            {
                "verbosity": 0,
                "objective": "binary:logistic",
                "booster": "gbtree",
                "learning_rate": 0.05,
                "max_depth": 5,
                "min_child_weight": 2,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": negative_count / positive_count,
                "tree_method": "hist",
                "seed": config.RANDOM_SEED,
                "nthread": 4,
            },
            dpretest_classifier,
            num_boost_round=int(final_rounds[0]),
            verbose_eval=False,
        )
        dpretest_general = xgb.DMatrix(X_pretest, label=y_pretest)
        final_general = xgb.train(
            {
                "verbosity": 0,
                "objective": "reg:squarederror",
                "booster": "gbtree",
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "seed": config.RANDOM_SEED,
                "nthread": 4,
            },
            dpretest_general,
            num_boost_round=int(final_rounds[1]),
            verbose_eval=False,
        )
        peak_fit = final_labels == 1
        dpretest_peak = xgb.DMatrix(X_pretest[peak_fit], label=y_pretest[peak_fit])
        final_expert = xgb.train(
            {
                "verbosity": 0,
                "objective": "reg:squarederror",
                "booster": "gbtree",
                "learning_rate": 0.03,
                "max_depth": 3,
                "min_child_weight": 2,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "tree_method": "hist",
                "seed": config.RANDOM_SEED,
                "nthread": 4,
            },
            dpretest_peak,
            num_boost_round=int(final_rounds[2]),
            verbose_eval=False,
        )

        dtest = xgb.DMatrix(X_test)
        test_probabilities = final_classifier.predict(dtest)
        test_general = final_general.predict(dtest)
        test_expert = final_expert.predict(dtest)
        test_labels = _peak_labels(scaler, y_test)
        test_two_stage = np.where(
            test_probabilities >= selected_threshold, test_expert, test_general
        )
        print("\nFinal untouched test results:")
        _report_numeric_metrics(
            "General regressor",
            test_general,
            y_test,
            scaler,
            test_timestamps,
        )
        _report_numeric_metrics(
            "Two-stage regressor",
            test_two_stage,
            y_test,
            scaler,
            test_timestamps,
        )
        _report_classifier_metrics(
            "Peak classifier",
            test_labels,
            test_probabilities,
            test_timestamps,
            selected_threshold,
        )
        for model_name, model in (
            ("classifier", final_classifier),
            ("general_regressor", final_general),
            ("peak_expert", final_expert),
        ):
            model.save_model(os.path.join(config.RESULTS_DIR, f"two_stage_{model_name}_{run_id}.json"))
        print(f"Saved final two-stage models under {config.RESULTS_DIR}.")


if __name__ == "__main__":
    main()
