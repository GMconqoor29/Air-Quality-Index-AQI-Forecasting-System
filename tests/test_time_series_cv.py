import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import src.config as config
from src.time_series_cv import build_expanding_window_folds


class ExpandingWindowFoldTests(unittest.TestCase):
    def test_folds_are_chronological_and_scalers_are_fold_local(self):
        row_count = 500
        timestamps = pd.date_range("2024-01-01", periods=row_count, freq="h")
        values = np.arange(row_count, dtype=np.float64)
        frame = pd.DataFrame(
            {column: 10.0 + values * 0.2 for column in config.FEATURE_COLS}
        )
        frame[config.TIME_COL] = timestamps
        frame[config.TARGET_COL] = 10.0 + values * 0.2

        with tempfile.TemporaryDirectory() as directory:
            frame.to_csv(Path(directory) / "readings.csv", index=False)
            folds = build_expanding_window_folds(
                directory,
                fold_count=4,
                initial_fraction=0.5,
                calibration_fraction=0.5,
            )

        self.assertEqual(len(folds), 4)
        target_idx = config.FEATURE_COLS.index(config.TARGET_COL)
        self.assertLess(
            folds[0].scaler.data_max_[target_idx],
            folds[-1].scaler.data_max_[target_idx],
        )
        for fold in folds:
            self.assertGreater(len(fold.X_train), 0)
            self.assertGreater(len(fold.X_calibration), 0)
            self.assertGreater(len(fold.X_score), 0)
            self.assertEqual(fold.X_train.shape[1:], (config.WINDOW_SIZE, len(config.FEATURE_COLS)))
            self.assertLess(
                fold.calibration_timestamps[-1],
                fold.score_timestamps[0],
            )
        for previous, current in zip(folds, folds[1:]):
            self.assertLess(previous.score_timestamps[-1], current.score_timestamps[0])


if __name__ == "__main__":
    unittest.main()
