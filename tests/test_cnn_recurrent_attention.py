import unittest

import torch

from src.model import CNNRecurrentAttentionAQIModel


class CNNRecurrentAttentionTests(unittest.TestCase):
    def test_cnn_gru_and_cnn_lstm_return_forecast_and_normalized_attention(self):
        features = torch.randn(4, 48, 13)
        for recurrent_type in ("GRU", "LSTM"):
            with self.subTest(recurrent_type=recurrent_type):
                model = CNNRecurrentAttentionAQIModel(
                    recurrent_type=recurrent_type,
                    conv_channels=16,
                    hidden_size=32,
                )
                prediction, attention = model(features, return_attention=True)

                self.assertEqual(tuple(prediction.shape), (4, 1))
                self.assertEqual(tuple(attention.shape), (4, 48))
                torch.testing.assert_close(
                    attention.sum(dim=1), torch.ones(4), atol=1e-6, rtol=1e-6
                )
                self.assertTrue(torch.all(attention >= 0))

    def test_rejects_unknown_recurrent_type(self):
        with self.assertRaisesRegex(ValueError, "recurrent_type"):
            CNNRecurrentAttentionAQIModel(recurrent_type="RNN")


if __name__ == "__main__":
    unittest.main()
