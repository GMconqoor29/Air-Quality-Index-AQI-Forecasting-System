import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import src.config as config
from src.data_loader import (
    _split_and_forward_fill_features,
    create_sequences,
    load_and_preprocess_data,
)
from src.evaluate import calculate_peak_metrics
from src.peak_classifier import (
    calculate_classifier_metrics,
    select_recall_thresholds,
)


class CausalPreprocessingTests(unittest.TestCase):
    def test_feature_config_uses_selected_measurements_and_preserves_baseline(self):
        self.assertEqual(
            config.FEATURE_COLS,
            [
                "PM25", "PM10", "Temperature", "Relative Humidity", "Pressure",
                "hour", "month", "AQI",
            ],
        )
        self.assertEqual(
            config.BASELINE_FEATURE_COLS,
            [
                "AQI", "PM25", "PM10", "CO", "NO2", "O3", "SO2",
                "Temperature", "Relative Humidity", "Wind Speed", "Pressure",
                "hour", "month",
            ],
        )

    def test_loader_inserts_missing_hours_without_interpolating_values(self):
        rows = []
        for timestamp, aqi, pm25 in (
            ("2025-01-01 00:00:00", 401.0, 10.0),
            ("2025-01-01 02:00:00", 70.0, 20.0),
        ):
            row: dict[str, float | str] = {
                name: 1.0 for name in config.FEATURE_COLS
            }
            row["AQI"] = aqi
            row["PM25"] = pm25
            row[config.TIME_COL] = timestamp
            rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame(rows).to_csv(Path(directory) / "readings.csv", index=False)
            result = load_and_preprocess_data(directory)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[config.TIME_COL].iloc[1], pd.Timestamp("2025-01-01 01:00:00"))
        self.assertTrue(pd.isna(result["AQI"].iloc[0]))
        self.assertTrue(pd.isna(result["AQI"].iloc[1]))
        self.assertTrue(pd.isna(result["PM25"].iloc[1]))

    def test_forward_fill_uses_past_only_and_preserves_missing_labels(self):
        timestamps = pd.date_range("2025-01-01", periods=30, freq="h")
        features = pd.DataFrame(
            {
                name: np.arange(len(timestamps), dtype=float)
                for name in config.FEATURE_COLS
            }
        )
        features[config.TIME_COL] = timestamps
        features.loc[0, "PM25"] = np.nan
        features.loc[2, "PM25"] = np.nan
        features.loc[10:11, "PM10"] = np.nan
        features.loc[21, "Pressure"] = np.nan
        features.loc[22, config.TARGET_COL] = np.nan

        train, validation, _ = _split_and_forward_fill_features(features)
        pm25_idx = config.FEATURE_COLS.index("PM25")
        pm10_idx = config.FEATURE_COLS.index("PM10")
        pressure_idx = config.FEATURE_COLS.index("Pressure")

        # A one-hour gap uses the previous value, not a later observation.
        self.assertEqual(train[0][2, pm25_idx], 1.0)
        # Longer gaps stay missing so windows containing stale inputs are skipped.
        self.assertEqual(train[0][10, pm10_idx], 9.0)
        self.assertTrue(np.isnan(train[0][11, pm10_idx]))
        # At the split boundary, only the final training observation is carried forward.
        self.assertEqual(validation[0][0, pressure_idx], train[0][-1, pressure_idx])
        # AQI is filled as an input feature, but a missing AQI target stays missing.
        self.assertTrue(np.isnan(validation[1][1]))

    def test_sequence_builder_skips_missing_targets(self):
        data = np.arange(20, dtype=float).reshape(10, 2)
        labels = np.arange(10, dtype=float)
        labels[4] = np.nan
        data[3, 0] = np.nan

        X, y, target_indices = create_sequences(
            data,
            window_size=2,
            horizon=1,
            target_values=labels,
            return_target_indices=True,
        )

        np.testing.assert_array_equal(target_indices, [2, 3, 6, 7, 8, 9])
        np.testing.assert_array_equal(y, labels[target_indices])
        self.assertEqual(X.shape, (6, 2, 2))


class PeakMetricTests(unittest.TestCase):
    def test_peak_metrics_include_confusion_counts_and_weekly_false_alert_rate(self):
        actual = np.array([210.0, 190.0, 240.0, 100.0])
        predicted = np.array([180.0, 210.0, 230.0, 150.0])
        timestamps = (
            np.datetime64("2025-01-01T00", "h") + np.arange(len(actual))
        )

        metrics = calculate_peak_metrics(
            actual, predicted, threshold=200, timestamps=timestamps
        )

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["peak_mae"], 20.0)
        self.assertAlmostEqual(metrics["false_alarms_per_week"], 42.0)

    def test_peak_metrics_reject_misaligned_timestamps(self):
        with self.assertRaisesRegex(ValueError, "timestamps must have the same length"):
            calculate_peak_metrics(
                [210, 190], [205, 180], threshold=200, timestamps=[np.datetime64("2025-01-01")]
            )


class PeakClassifierMetricTests(unittest.TestCase):
    def test_recall_thresholds_choose_highest_cutoff_meeting_each_target(self):
        labels = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.6, 0.8, 0.2])

        thresholds = select_recall_thresholds(
            labels, probabilities, recall_targets=(0.5, 1.0)
        )

        self.assertEqual(thresholds, {0.5: 0.9, 1.0: 0.6})

    def test_classifier_metrics_report_recall_false_alarms_and_weekly_rate(self):
        labels = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.4, 0.8, 0.2])
        timestamps = np.datetime64("2025-01-01T00", "h") + np.arange(4)

        metrics = calculate_classifier_metrics(
            labels, probabilities, threshold=0.5, timestamps=timestamps
        )

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["false_alerts_per_week"], 42.0)


if __name__ == "__main__":
    unittest.main()
