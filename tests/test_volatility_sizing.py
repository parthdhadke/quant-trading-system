import unittest

import numpy as np
import pandas as pd

from config import MAX_POSITION_SIZE, MIN_POSITION_SIZE, TARGET_VOLATILITY
from src.backtest import run_backtest
from src.signals import apply_volatility_sizing, volatility_position_size


class SizingHelperTests(unittest.TestCase):
    def test_higher_volatility_reduces_position_size(self):
        sizes = [volatility_position_size(value, 0.15) for value in (0.15, 0.30, 0.60)]
        np.testing.assert_allclose(sizes, [1.0, 0.5, 0.25])

    def test_sizes_clipped_to_configured_bounds(self):
        for volatility in (1e-300, 0.01, 0.15, 0.30, 0.60, 10):
            size = volatility_position_size(volatility, 0.15, max_position_size=0.8, min_position_size=0.1)
            self.assertGreaterEqual(size, 0.1)
            self.assertLessEqual(size, 0.8)

    def test_leverage_above_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "leverage"):
            volatility_position_size(0.01, max_position_size=1.01)

    def test_positive_negative_neutral_signals(self):
        data = pd.DataFrame({"signal": [1, -1, 0], "volatility": [0.3, 0.3, 0.3]})
        sized = apply_volatility_sizing(data, "signal", "volatility", 0.15)
        np.testing.assert_allclose(sized["position"], [0.5, -0.5, 0])
        np.testing.assert_allclose(sized["position_size"], [0.5, 0.5, 0.5])

    def test_nan_and_none_volatility_use_minimum_size(self):
        for value in (np.nan, None):
            self.assertEqual(volatility_position_size(value, 0.15), 0)
            self.assertEqual(volatility_position_size(value, 0.15, min_position_size=0.1), 0.1)

    def test_infinite_volatility_uses_minimum_size(self):
        for value in (np.inf, -np.inf):
            self.assertEqual(volatility_position_size(value, 0.15), 0)

    def test_zero_and_negative_volatility_preserve_capped_fallback(self):
        for value in (0, -0.1):
            size = volatility_position_size(value, 0.15)
            self.assertTrue(np.isfinite(size))
            self.assertEqual(size, MAX_POSITION_SIZE)

    def test_nonfinite_sizing_settings_are_rejected(self):
        for name in ("target_volatility", "max_position_size", "min_position_size"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    volatility_position_size(0.15, **{name: value})

    def test_invalid_bounds_and_target_are_rejected(self):
        for settings in ({"target_volatility": 0}, {"min_position_size": -0.1},
                         {"max_position_size": 0}, {"min_position_size": 0.6, "max_position_size": 0.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                volatility_position_size(0.15, **settings)

    def test_configuration_defaults_are_used(self):
        self.assertEqual(volatility_position_size(0.3), TARGET_VOLATILITY / 0.3)
        self.assertEqual(volatility_position_size(np.nan), MIN_POSITION_SIZE)

    def test_fallbacks_never_produce_invalid_positions(self):
        data = pd.DataFrame({"signal": [1, -1, 0, 1, -1, 0],
                             "volatility": [np.nan, np.inf, -np.inf, 0, -0.1, 0.3]})
        sized = apply_volatility_sizing(data, "signal", "volatility")
        self.assertTrue(np.isfinite(sized[["position", "position_size"]].to_numpy()).all())
        self.assertTrue(sized["position"].abs().le(1).all())

    def test_input_and_directional_signals_are_unchanged(self):
        data = pd.DataFrame({"signal": [1, 0, -1], "volatility": [0.3, 0.6, 0.15]})
        original = data.copy(deep=True)
        sized = apply_volatility_sizing(data, "signal", "volatility")
        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_series_equal(sized["signal"], data["signal"])

    def test_cannot_overwrite_directional_signal(self):
        data = pd.DataFrame({"signal": [1], "volatility": [0.3]})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            apply_volatility_sizing(data, "signal", "volatility", output_column="signal")

    def test_invalid_directional_signals_are_rejected(self):
        for value in (2, 0.5, np.nan, np.inf):
            data = pd.DataFrame({"signal": [value], "volatility": [0.3]})
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Directional"):
                apply_volatility_sizing(data, "signal", "volatility")

    def test_future_volatility_does_not_affect_earlier_sizes(self):
        data = pd.DataFrame({"signal": [1, 1, -1], "volatility": [0.3, 0.6, 0.15]})
        first = apply_volatility_sizing(data, "signal", "volatility")
        data.loc[2, "volatility"] = 10
        changed = apply_volatility_sizing(data, "signal", "volatility")
        pd.testing.assert_frame_equal(first.iloc[:2], changed.iloc[:2])

    def test_fractional_positions_have_one_execution_delay(self):
        data = pd.DataFrame({"signal": [1, -1, 0, 1], "volatility": [0.3, 0.6, 0.15, 0.15],
                             "return": [0.4, 0.02, -0.04, 0.1]})
        sized = apply_volatility_sizing(data, "signal", "volatility")
        backtest = run_backtest(sized, "position", commission_bps=0, slippage_bps=0)
        np.testing.assert_allclose(backtest["executed_position"], [0, 0.5, -0.25, 0])
        np.testing.assert_allclose(backtest["gross_return"], [0, 0.01, 0.01, 0])

    def test_transaction_costs_follow_fractional_turnover(self):
        data = pd.DataFrame({"signal": [1, -1, 0, 1], "volatility": [0.3, 0.6, 0.15, 0.15],
                             "return": [0, 0.02, -0.04, 0]})
        sized = apply_volatility_sizing(data, "signal", "volatility")
        backtest = run_backtest(sized, "position", commission_bps=5, slippage_bps=5)
        np.testing.assert_allclose(backtest["turnover"], [0, 0.5, 0.75, 0.25])
        np.testing.assert_allclose(backtest["transaction_cost"], [0, 0.0005, 0.00075, 0.00025])
        np.testing.assert_allclose(backtest["net_return"], [0, 0.0095, 0.00925, -0.00025])


if __name__ == "__main__":
    unittest.main()
