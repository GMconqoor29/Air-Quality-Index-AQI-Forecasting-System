import unittest

import numpy as np
import torch

from src.hybrid_ensemble import (
    AsymmetricWeightedMSE,
    calculate_sample_weights,
    fit_soft_gate,
)


class AsymmetricWeightedMSETests(unittest.TestCase):
    def test_high_aqi_targets_get_larger_sample_weights(self):
        weights = calculate_sample_weights(
            np.array([100.0, 200.0, 300.0]), threshold=200.0, severity_span=100.0
        )

        np.testing.assert_allclose(weights, [1.0, 1.0, 4.0])

    def test_underprediction_is_penalized_more_than_equal_overprediction(self):
        criterion = AsymmetricWeightedMSE(
            threshold=0.5,
            severity_span=0.5,
            high_aqi_weight=0.0,
            underprediction_penalty=2.0,
        )
        target = torch.tensor([[0.6]])

        underforecast_loss = criterion(torch.tensor([[0.5]]), target)
        overforecast_loss = criterion(torch.tensor([[0.7]]), target)

        self.assertAlmostEqual(underforecast_loss.item(), 0.03, places=6)
        self.assertAlmostEqual(overforecast_loss.item(), 0.01, places=6)

    def test_loss_weights_high_aqi_samples_more(self):
        criterion = AsymmetricWeightedMSE(
            threshold=0.5,
            severity_span=0.5,
            high_aqi_weight=3.0,
            underprediction_penalty=0.0,
        )

        weights = criterion.sample_weights(torch.tensor([[0.4], [1.0]]))

        torch.testing.assert_close(weights, torch.tensor([[1.0], [4.0]]))


class SoftGatingTests(unittest.TestCase):
    def test_fitted_gate_returns_convex_weights_and_forecasts(self):
        predictions = np.array(
            [[0.1, 0.2], [0.2, 0.3], [0.7, 0.6], [0.8, 0.7]], dtype=np.float32
        )
        context = np.array([[0.1, 0.0], [0.2, 0.1], [0.7, 0.5], [0.8, 0.1]], dtype=np.float32)
        targets = np.array([0.15, 0.25, 0.65, 0.75], dtype=np.float32)
        criterion = AsymmetricWeightedMSE(0.5, 0.5)
        gate = fit_soft_gate(
            predictions, context, targets, criterion, device="cpu", epochs=10
        )

        with torch.no_grad():
            forecasts, weights = gate(
                torch.tensor(predictions), torch.tensor(context)
            )

        self.assertEqual(tuple(forecasts.shape), (4, 1))
        self.assertEqual(tuple(weights.shape), (4, 2))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))
        self.assertTrue(torch.all(weights >= 0))


if __name__ == "__main__":
    unittest.main()
