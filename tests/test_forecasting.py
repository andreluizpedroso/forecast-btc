import unittest

import numpy as np
import pandas as pd

from forecast_btc.forecasting import (
    PriceForecaster,
    build_forecast_frame,
    moving_average_recommendation,
)


class LastValueModel:
    """Modelo mínimo usado para testar o fluxo sem depender do TensorFlow."""

    def predict(self, values, verbose=0):
        return np.array([[values[0, -1, 0]]])


class PriceForecasterTests(unittest.TestCase):
    def setUp(self):
        self.forecaster = PriceForecaster(LastValueModel(), sequence_length=3)

    def test_predicts_requested_number_of_days(self):
        predictions = self.forecaster.predict([10, 20, 30, 40], days=3)

        np.testing.assert_allclose(predictions, [40, 40, 40])

    def test_rejects_history_shorter_than_model_window(self):
        with self.assertRaisesRegex(ValueError, "ao menos 3"):
            self.forecaster.predict([10, 20])

    def test_rejects_non_future_target_date(self):
        with self.assertRaisesRegex(ValueError, "deve ser futura"):
            self.forecaster.predict_to_date(
                [10, 20, 30], "2026-01-02", "2026-01-02"
            )


class ForecastFrameTests(unittest.TestCase):
    def test_builds_daily_forecast_and_moving_averages(self):
        history = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=80),
                "Close": np.arange(1, 81, dtype=float),
            }
        )

        result = build_forecast_frame(history, [81.0, 82.0])

        self.assertEqual(pd.Timestamp("2025-03-23"), result["Date"].iloc[-1])
        self.assertEqual(82.0, result["Close"].iloc[-1])
        self.assertFalse(np.isnan(result["MM80"].iloc[-1]))

    def test_recommends_buy_when_short_average_is_higher(self):
        history = pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=80),
                "Close": np.arange(1, 81, dtype=float),
            }
        )
        result = build_forecast_frame(history, [81.0])

        recommendation = moving_average_recommendation(result, "2025-03-22")

        self.assertIn("Comprar", recommendation)


if __name__ == "__main__":
    unittest.main()
