import numpy as np
from sklearn.metrics import average_precision_score


def _validate_binary_inputs(labels, probabilities):
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.size != probabilities.size:
        raise ValueError("labels and probabilities must have the same non-zero length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("probabilities must be finite values between 0 and 1.")
    return labels.astype(np.int8), probabilities


def select_recall_thresholds(labels, probabilities, recall_targets):
    """Return the highest probability cutoff reaching each requested recall."""
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError("labels must contain at least one positive example.")

    thresholds = {}
    for recall_target in recall_targets:
        if not 0 < recall_target <= 1:
            raise ValueError("recall targets must be greater than 0 and at most 1.")
        required_positives = int(np.ceil(recall_target * positive_count))
        positive_probabilities = np.sort(probabilities[labels == 1])[::-1]
        thresholds[recall_target] = float(
            positive_probabilities[required_positives - 1]
        )
    return thresholds


def calculate_classifier_metrics(labels, probabilities, threshold, timestamps=None):
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value between 0 and 1.")

    predicted = probabilities >= threshold
    actual = labels == 1
    true_positive = int(np.count_nonzero(actual & predicted))
    false_positive = int(np.count_nonzero(~actual & predicted))
    false_negative = int(np.count_nonzero(actual & ~predicted))
    predicted_positive = true_positive + false_positive
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / int(actual.sum()) if actual.any() else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    false_alerts_per_week = None
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype="datetime64[ns]").reshape(-1)
        if timestamps.size != labels.size:
            raise ValueError("timestamps must have the same length as labels.")
        if np.isnat(timestamps).any():
            raise ValueError("timestamps must not contain NaT values.")
        elapsed_hours = (
            (timestamps.max() - timestamps.min())
            .astype("timedelta64[ns]")
            .astype(np.int64)
            / 3_600_000_000_000
            + 1
        )
        false_alerts_per_week = false_positive / (elapsed_hours / 168)

    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "actual_peaks": int(actual.sum()),
        "forecast_peaks": int(predicted.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_week": false_alerts_per_week,
    }
