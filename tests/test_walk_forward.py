import unittest

from src.walk_forward import rolling_origin_splits


class RollingOriginSplitTests(unittest.TestCase):
    def test_folds_expand_training_prefix_and_keep_validation_ordered(self):
        folds = rolling_origin_splits(100, fold_count=4, initial_fraction=0.5)

        self.assertEqual(folds, [(0, 50, 62), (0, 62, 75), (0, 75, 87), (0, 87, 100)])
        for index, (train_start, valid_start, valid_stop) in enumerate(folds):
            self.assertEqual(train_start, 0)
            self.assertLess(train_start, valid_start)
            self.assertLess(valid_start, valid_stop)
            if index:
                self.assertEqual(folds[index - 1][2], valid_start)

    def test_rejects_invalid_fold_layouts(self):
        with self.assertRaises(ValueError):
            rolling_origin_splits(10, fold_count=10)
        with self.assertRaises(ValueError):
            rolling_origin_splits(10, initial_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
