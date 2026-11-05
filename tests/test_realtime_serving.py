import os
import tempfile
import unittest

from src.serving.aqi_calculator import compute_aqi_from_pm
from src.serving.ingestor import RealtimeDataStore
from src.serving.predictor import RealtimePredictor


class TestRealtimeServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = RealtimePredictor()

    def test_empirical_aqi_mapping(self):
        # Low PM2.5 (~12 ug/m3) -> Good/Moderate border (~50)
        aqi_low = compute_aqi_from_pm(12.0, 20.0)
        self.assertIsNotNone(aqi_low)
        self.assertTrue(35 <= aqi_low <= 65)

        # High PM2.5 (~95 ug/m3) -> reaches peak regime (>= 200) per Weatherbit empirical curve
        aqi_peak = compute_aqi_from_pm(95.0, 120.0)
        self.assertIsNotNone(aqi_peak)
        self.assertGreaterEqual(aqi_peak, 200)

    def test_all_models_predict_t1_and_t6(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_aqi.db")
            store = RealtimeDataStore(db_path=db_path)
            store.ensure_warm_start(hours=72)
            history = store.get_historical_readings(hours=48)
            self.assertEqual(len(history), 48)

            preds_t1 = self.predictor.predict_all_models(history, horizon=1)
            preds_t6 = self.predictor.predict_all_models(history, horizon=6)

            expected_models = {
                "Base LSTM",
                "Stacked LSTM",
                "GRU",
                "XGBoost",
                "Tuned XGBoost",
                "CNN-GRU Attention",
                "CNN-LSTM Attention",
                "Soft-Gated Ensemble",
                "Two-Stage Regressor",
            }
            self.assertEqual(set(preds_t1.keys()), expected_models)
            self.assertEqual(set(preds_t6.keys()), expected_models)
            for name, val in preds_t1.items():
                self.assertTrue(0 <= val <= 500, f"{name} t+1 out of bounds: {val}")

            # Verify ESP32 telemetry ingestion updates current reading
            obs = store.ingest_esp32_telemetry(
                {
                    "pm25": 98.4,
                    "pm10": 132.0,
                    "temperature": 24.5,
                    "relativeHumidity": 82.0,
                    "pressure": 1011.2,
                }
            )
            self.assertEqual(obs["source"], "esp32_sensor")
            self.assertGreaterEqual(obs["aqi"], 200)


if __name__ == "__main__":
    unittest.main()

