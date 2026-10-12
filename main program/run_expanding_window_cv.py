"""Compare AQI forecasters with expanding-window time-series cross-validation."""

import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

import src.config as config
from src.data_loader import AQIDataset
from src.evaluate import calculate_peak_metrics, inverse_scale_predictions
from src.hybrid_ensemble import (
    AsymmetricWeightedMSE,
    fit_soft_gate,
    make_gating_context,
    xgboost_asymmetric_metric,
    xgboost_asymmetric_objective,
)
from src.model import GRUAQIModel, StackedLSTMAQIModel
from src.peak_classifier import (
    calculate_classifier_metrics,
    select_recall_thresholds,
)
from src.time_series_cv import build_expanding_window_folds
from src.train import train_model
from src.utils import DualLogger

FOLD_COUNT = 4
INITIAL_TRAIN_FRACTION = 0.5
CALIBRATION_FRACTION = 0.5
RECALL_TARGET = 0.75
PEAK_THRESHOLD = config.PEAK_AQI_THRESHOLD


def _inverse_targets(values, scaler):
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
    return inverse_scale_predictions(scaler, np.asarray(values).reshape(-1, 1), target_idx).ravel()


def _make_loader(features, targets, shuffle):
    return DataLoader(
        AQIDataset(features, targets),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
    )


def _collect_predictions(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for features, labels in loader:
            predictions.append(model(features.to(device)).cpu().numpy().reshape(-1))
            targets.append(labels.numpy().reshape(-1))
    return np.concatenate(predictions), np.concatenate(targets)


def _train_regressor(train_matrix, train_targets, calibration_matrix, calibration_targets, seed):
    dtrain = xgb.DMatrix(train_matrix, label=train_targets)
    dcal = xgb.DMatrix(calibration_matrix, label=calibration_targets)
    return xgb.train(
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
        dtrain,
        num_boost_round=500,
        evals=[(dcal, "calibration")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )


def _best_iteration(model):
    return int(model.best_iteration) + 1


def _boost_predictions(model, features):
    return model.predict(
        xgb.DMatrix(features),
        iteration_range=(0, _best_iteration(model)),
    )


def _train_peak_models(X_train, y_train, X_cal, y_cal, scaler, seed):
    threshold_scaled = (
        PEAK_THRESHOLD
        * scaler.scale_[config.FEATURE_COLS.index(config.TARGET_COL)]
        + scaler.min_[config.FEATURE_COLS.index(config.TARGET_COL)]
    )
    train_labels = (y_train >= threshold_scaled).astype(np.int32)
    cal_labels = (y_cal >= threshold_scaled).astype(np.int32)
    positive_count = int(train_labels.sum())
    negative_count = len(train_labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("A fold's training data must contain both peak classes.")
    if not cal_labels.any() or cal_labels.all():
        raise ValueError("A calibration block must contain both peak classes.")

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
        xgb.DMatrix(X_train, label=train_labels),
        num_boost_round=500,
        evals=[(xgb.DMatrix(X_cal, label=cal_labels), "calibration")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    peak_train = train_labels == 1
    peak_calibration = cal_labels == 1
    expert = xgb.train(
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
        xgb.DMatrix(X_train[peak_train], label=y_train[peak_train]),
        num_boost_round=500,
        evals=[
            (
                xgb.DMatrix(
                    X_cal[peak_calibration], label=y_cal[peak_calibration]
                ),
                "calibration_peaks",
            )
        ],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return classifier, expert, cal_labels


def _fold_duration_weeks(timestamps):
    elapsed_hours = (
        (timestamps.max() - timestamps.min())
        .astype("timedelta64[ns]")
        .astype(np.int64)
        / 3_600_000_000_000
        + 1
    )
    return elapsed_hours / 168


def _report_regression(name, actual, predicted, scored_weeks):
    actual_peak = actual >= PEAK_THRESHOLD
    if not actual_peak.any():
        raise ValueError(f"{name} has no actual peaks in its scored folds.")
    peak_metrics = calculate_peak_metrics(actual, predicted, PEAK_THRESHOLD)
    false_alerts_per_week = peak_metrics["false_positive"] / scored_weeks
    print(
        f"{name:28s} RMSE={np.sqrt(mean_squared_error(actual, predicted)):8.3f} "
        f"MAE={mean_absolute_error(actual, predicted):8.3f} "
        f"peak_MAE={peak_metrics['peak_mae']:8.3f} "
        f"peak_recall={peak_metrics['recall']:.4f} "
        f"peak_F1={peak_metrics['f1']:.4f} "
        f"FP/week={false_alerts_per_week:.2f}"
    )


def main():
    os.makedirs("results", exist_ok=True)
    with DualLogger("results/expanding_window_cv_results.txt"):
        folds = build_expanding_window_folds(
            data_dir="Dataset/archive",
            fold_count=FOLD_COUNT,
            initial_fraction=INITIAL_TRAIN_FRACTION,
            calibration_fraction=CALIBRATION_FRACTION,
        )
        print(
            f"Expanding-window CV: {len(folds)} folds; "
            f"{CALIBRATION_FRACTION:.0%} of each validation block for calibration, "
            "the later portion for scoring."
        )
        print("The final 20% test period is excluded from fold training and scoring.")

        results = {
            name: {
                "actual": [],
                "predictions": [],
                "scored_weeks": 0.0,
            }
            for name in (
                "XGBoost (standard)",
                "XGBoost (AMSE)",
                "GRU (AMSE)",
                "Stacked LSTM (AMSE)",
                "Soft-gated ensemble",
                "Two-stage regressor",
            )
        }
        classifier_results = {
            "labels": [],
            "probabilities": [],
            "calibration_labels": [],
            "calibration_probabilities": [],
            "predicted_labels": [],
            "scored_weeks": 0.0,
        }
        device = config.DEVICE
        target_idx = config.FEATURE_COLS.index(config.TARGET_COL)

        for fold_number, fold in enumerate(folds, 1):
            config.set_seed(config.RANDOM_SEED + fold_number)
            fold_scaler = fold.scaler
            X_train = fold.X_train
            X_cal = fold.X_calibration
            X_score = fold.X_score
            y_train = fold.y_train
            y_cal = fold.y_calibration
            y_score = fold.y_score
            X_train_flat = X_train.reshape(len(X_train), -1)
            X_cal_flat = X_cal.reshape(len(X_cal), -1)
            X_score_flat = X_score.reshape(len(X_score), -1)
            print(
                f"\nFold {fold_number}: train={len(X_train)}, "
                f"calibration={len(X_cal)}, score={len(X_score)}; "
                f"score time={fold.score_timestamps[0]} to "
                f"{fold.score_timestamps[-1]}"
            )

            standard = _train_regressor(
                X_train_flat, y_train, X_cal_flat, y_cal, config.RANDOM_SEED + fold_number
            )
            standard_cal = _boost_predictions(standard, X_cal_flat)
            standard_score = _boost_predictions(standard, X_score_flat)

            threshold_scaled = (
                PEAK_THRESHOLD * fold_scaler.scale_[target_idx]
                + fold_scaler.min_[target_idx]
            )
            severity_span = max(
                float(
                    (fold_scaler.data_max_[target_idx] - PEAK_THRESHOLD)
                    * fold_scaler.scale_[target_idx]
                ),
                1e-6,
            )
            dtrain = xgb.DMatrix(X_train_flat, label=y_train)
            dcal = xgb.DMatrix(X_cal_flat, label=y_cal)
            asymmetric = xgb.train(
                {
                    "verbosity": 0,
                    "objective": "reg:squarederror",
                    "tree_method": "hist",
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "seed": config.RANDOM_SEED + fold_number,
                    "nthread": 4,
                },
                dtrain,
                num_boost_round=500,
                evals=[(dcal, "calibration")],
                obj=xgboost_asymmetric_objective(threshold_scaled, severity_span),
                custom_metric=xgboost_asymmetric_metric(
                    threshold_scaled, severity_span
                ),
                early_stopping_rounds=25,
                verbose_eval=False,
            )
            asymmetric_cal = _boost_predictions(asymmetric, X_cal_flat)
            asymmetric_score = _boost_predictions(asymmetric, X_score_flat)

            criterion = AsymmetricWeightedMSE(
                threshold=threshold_scaled,
                severity_span=severity_span,
                high_aqi_weight=3.0,
                underprediction_penalty=2.0,
            )
            train_loader = _make_loader(X_train, y_train, shuffle=True)
            calibration_loader = _make_loader(X_cal, y_cal, shuffle=False)
            score_loader = _make_loader(X_score, y_score, shuffle=False)
            neural_predictions = {}
            with tempfile.TemporaryDirectory(prefix=f"aqi_cv_fold_{fold_number}_") as temp_dir:
                for name, model in (
                    ("GRU (AMSE)", GRUAQIModel()),
                    ("Stacked LSTM (AMSE)", StackedLSTMAQIModel(num_layers=2)),
                ):
                    model = model.to(device)
                    model, _ = train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=calibration_loader,
                        criterion=criterion,
                        optimizer=torch.optim.Adam(
                            model.parameters(), lr=config.LEARNING_RATE
                        ),
                        device=device,
                        epochs=config.EPOCHS,
                        patience=10,
                        checkpoint_path=os.path.join(
                            temp_dir, name.lower().replace(" ", "_") + ".pt"
                        ),
                        verbose=False,
                    )
                    calibration_prediction, calibration_target = _collect_predictions(
                        model, calibration_loader, device
                    )
                    score_prediction, score_target = _collect_predictions(
                        model, score_loader, device
                    )
                    if not np.allclose(calibration_target, y_cal) or not np.allclose(
                        score_target, y_score
                    ):
                        raise RuntimeError("Neural and XGBoost fold labels are misaligned.")
                    neural_predictions[name] = (
                        calibration_prediction,
                        score_prediction,
                    )

            calibration_context = make_gating_context(X_cal, target_idx)
            score_context = make_gating_context(X_score, target_idx)
            gate_cal_predictions = np.column_stack(
                (
                    asymmetric_cal,
                    neural_predictions["GRU (AMSE)"][0],
                    neural_predictions["Stacked LSTM (AMSE)"][0],
                )
            )
            gate_score_predictions = np.column_stack(
                (
                    asymmetric_score,
                    neural_predictions["GRU (AMSE)"][1],
                    neural_predictions["Stacked LSTM (AMSE)"][1],
                )
            )
            gate = fit_soft_gate(
                gate_cal_predictions,
                calibration_context,
                y_cal,
                criterion,
                device,
            )
            with torch.no_grad():
                gated_score, _ = gate(
                    torch.as_tensor(gate_score_predictions, dtype=torch.float32, device=device),
                    torch.as_tensor(score_context, dtype=torch.float32, device=device),
                )

            classifier, expert, cal_labels = _train_peak_models(
                X_train_flat,
                y_train,
                X_cal_flat,
                y_cal,
                fold_scaler,
                config.RANDOM_SEED + fold_number,
            )
            classifier_cal = classifier.predict(
                xgb.DMatrix(X_cal_flat),
                iteration_range=(0, _best_iteration(classifier)),
            )
            classifier_score = classifier.predict(
                xgb.DMatrix(X_score_flat),
                iteration_range=(0, _best_iteration(classifier)),
            )
            cutoff = select_recall_thresholds(
                cal_labels, classifier_cal, (RECALL_TARGET,)
            )[RECALL_TARGET]
            expert_score = _boost_predictions(expert, X_score_flat)
            two_stage_score = np.where(
                classifier_score >= cutoff, expert_score, standard_score
            )

            actual_raw = _inverse_targets(y_score, fold_scaler)
            predictions_scaled = {
                "XGBoost (standard)": standard_score,
                "XGBoost (AMSE)": asymmetric_score,
                "GRU (AMSE)": neural_predictions["GRU (AMSE)"][1],
                "Stacked LSTM (AMSE)": neural_predictions["Stacked LSTM (AMSE)"][1],
                "Soft-gated ensemble": gated_score.cpu().numpy().reshape(-1),
                "Two-stage regressor": two_stage_score,
            }
            for name, prediction in predictions_scaled.items():
                results[name]["actual"].append(actual_raw)
                results[name]["predictions"].append(
                    _inverse_targets(prediction, fold_scaler)
                )
                results[name]["scored_weeks"] += _fold_duration_weeks(
                    fold.score_timestamps
                )

            classifier_results["labels"].append(
                (actual_raw >= PEAK_THRESHOLD).astype(np.int32)
            )
            classifier_results["probabilities"].append(classifier_score)
            classifier_results["calibration_labels"].append(cal_labels)
            classifier_results["calibration_probabilities"].append(classifier_cal)
            classifier_results["predicted_labels"].append(
                (classifier_score >= cutoff).astype(np.int32)
            )
            classifier_results["scored_weeks"] += _fold_duration_weeks(
                fold.score_timestamps
            )

            classifier_metrics = calculate_classifier_metrics(
                (actual_raw >= PEAK_THRESHOLD).astype(np.int32),
                classifier_score,
                cutoff,
                timestamps=fold.score_timestamps,
            )
            print(
                f"  Fold cutoff={cutoff:.4f}; peak classifier recall="
                f"{classifier_metrics['recall']:.3f}, precision="
                f"{classifier_metrics['precision']:.3f}, "
                f"FP/week={classifier_metrics['false_alerts_per_week']:.2f}"
            )

        print("\nPooled out-of-fold score results (calibration portions excluded):")
        for name, values in results.items():
            actual = np.concatenate(values["actual"])
            predictions = np.concatenate(values["predictions"])
            _report_regression(name, actual, predictions, values["scored_weeks"])

        labels = np.concatenate(classifier_results["labels"])
        probabilities = np.concatenate(classifier_results["probabilities"])
        calibration_labels = np.concatenate(classifier_results["calibration_labels"])
        calibration_probabilities = np.concatenate(
            classifier_results["calibration_probabilities"]
        )
        predicted_labels = np.concatenate(classifier_results["predicted_labels"])
        true_positive = int(np.count_nonzero((labels == 1) & (predicted_labels == 1)))
        false_positive = int(np.count_nonzero((labels == 0) & (predicted_labels == 1)))
        false_negative = int(np.count_nonzero((labels == 1) & (predicted_labels == 0)))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        print(
            "Peak classifier (fold-calibrated cutoffs): "
            f"recall={recall:.3f}, precision={precision:.3f}, F1={f1:.3f}, "
            f"FP/week={false_positive / classifier_results['scored_weeks']:.2f}"
        )
        # The cutoff is selected from calibration predictions, then measured
        # on score predictions whose labels did not choose the cutoff.
        for target in (0.50, 0.75, 0.90):
            cutoff = select_recall_thresholds(
                calibration_labels, calibration_probabilities, (target,)
            )[target]
            metrics = calculate_classifier_metrics(
                labels, probabilities, cutoff
            )
            print(
                f"Peak classifier OOF recall-target={target:.0%}, "
                f"cutoff={cutoff:.4f}, recall={metrics['recall']:.3f}, "
                f"precision={metrics['precision']:.3f}, F1={metrics['f1']:.3f}, "
                f"FP/week={metrics['false_positive'] / classifier_results['scored_weeks']:.2f}"
            )
        print(
            "\nThese fold results are for model comparison only. "
            "They do not evaluate or tune on the final test period."
        )


if __name__ == "__main__":
    main()
