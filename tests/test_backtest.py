import unittest

import numpy as np
import pandas as pd

from config import COMMISSION_BPS, SLIPPAGE_BPS
from src.backtest import run_backtest
from src.evaluation import compare_strategies, performance_metrics


def synthetic_backtest(positions, returns, commission_bps=0, slippage_bps=0):
    data = pd.DataFrame(
        {"signal": positions, "return": returns},
        index=pd.date_range("2024-01-02", periods=len(positions), freq="B", name="date"),
    )
    return run_backtest(data, "signal", commission_bps=commission_bps, slippage_bps=slippage_bps, initial_capital=100.0)


class BacktestTests(unittest.TestCase):
    def test_today_signal_cannot_earn_today_return(self):
        result = synthetic_backtest([1, 0, 0], [0.8, 0.0, 0.0])
        np.testing.assert_array_equal(result["gross_return"], [0, 0, 0])

    def test_t_plus_one_execution_and_final_signal_not_executed(self):
        result = synthetic_backtest([1, -1, 0, 1], [0.1, 0.02, -0.03, 0.04])
        np.testing.assert_array_equal(result["executed_position"], [0, 1, -1, 0])

    def test_long_returns(self):
        result = synthetic_backtest([1, 1, 1], [0.5, 0.1, -0.05])
        np.testing.assert_allclose(result["gross_return"], [0, 0.1, -0.05])
        self.assertAlmostEqual(result["equity"].iloc[-1], 104.5)

    def test_short_returns(self):
        result = synthetic_backtest([-1, -1, -1], [0.5, 0.1, -0.05])
        np.testing.assert_allclose(result["gross_return"], [0, -0.1, 0.05])
        self.assertAlmostEqual(result["equity"].iloc[-1], 94.5)

    def test_neutral_gross_returns(self):
        result = synthetic_backtest([0, 0, 0], [0.4, -0.1, 0.9], 5, 5)
        np.testing.assert_array_equal(result["gross_return"], [0, 0, 0])
        np.testing.assert_array_equal(result["equity"], [100, 100, 100])

    def test_entry_exit_and_reversal_turnover(self):
        result = synthetic_backtest([1, -1, 0, 0], [0, 0, 0, 0])
        np.testing.assert_array_equal(result["turnover"], [0, 1, 2, 1])
        np.testing.assert_array_equal(result["trade_event"], [0, 1, 1, 1])

    def test_commission_and_slippage_use_executed_turnover(self):
        result = synthetic_backtest([1, -1, 0, 0], [0, 0.1, -0.1, 0.3], 5, 5)
        np.testing.assert_allclose(result["transaction_cost"], [0, 0.001, 0.002, 0.001])
        np.testing.assert_allclose(result["net_return"], [0, 0.099, 0.098, -0.001])
        self.assertAlmostEqual(result["equity"].iloc[-1], 100 * 1.099 * 1.098 * 0.999)

    def test_positive_costs_never_increase_daily_or_cumulative_returns(self):
        result = synthetic_backtest([1, -1, 0, 1, 1], [0.1, -0.03, 0.05, -0.01, 0.02], 5, 5)
        self.assertTrue((result["gross_return"] >= result["net_return"]).all())
        self.assertTrue((result["gross_equity"] >= result["equity"]).all())

    def test_default_costs_match_config(self):
        data = pd.DataFrame({"signal": [1, 1], "return": [0.0, 0.01]})
        result = run_backtest(data, "signal")
        self.assertAlmostEqual(result["transaction_cost"].iloc[1], (COMMISSION_BPS + SLIPPAGE_BPS) / 10000)

    def test_invalid_rows_rejected_without_compressing_execution_timeline(self):
        for column in ("signal", "return"):
            for invalid in (np.nan, np.inf, -np.inf, "bad"):
                with self.subTest(column=column, invalid=invalid):
                    data = pd.DataFrame({"signal": [1, invalid, 0] if column == "signal" else [1, 0, 0],
                                         "return": [0, invalid, 0.2] if column == "return" else [0, 0.1, 0.2]})
                    with self.assertRaises((ValueError, TypeError)):
                        run_backtest(data, "signal")

    def test_date_column_orders_execution_instead_of_range_index(self):
        data = pd.DataFrame({"date": pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"]),
                             "signal": [0, 1, -1], "return": [0.02, 0.9, 0.1]})
        result = run_backtest(data, "signal", commission_bps=0, slippage_bps=0)
        self.assertTrue(result["date"].is_monotonic_increasing)
        np.testing.assert_allclose(result["gross_return"], [0, 0.1, -0.02])

    def test_duplicate_dates_rejected(self):
        data = pd.DataFrame({"signal": [1, 0], "return": [0, 0.1]}, index=[0, 0])
        with self.assertRaisesRegex(ValueError, "unique"):
            run_backtest(data, "signal")

    def test_invalid_settings_rejected(self):
        data = pd.DataFrame({"signal": [1], "return": [0.1]})
        for setting in ("commission_bps", "slippage_bps", "initial_capital"):
            for value in (np.nan, np.inf, -1):
                with self.subTest(setting=setting, value=value), self.assertRaises(ValueError):
                    run_backtest(data, "signal", **{setting: value})

    def test_insolvency_rejected_instead_of_compounding_negative_wealth(self):
        with self.assertRaisesRegex(ValueError, "insolvency"):
            synthetic_backtest([-1, -1], [0, 1.5])

    def test_outputs_finite_with_no_unexpected_nan(self):
        result = synthetic_backtest([1, -1, 0, 0], [0.1, 0.02, -0.03, 0.01], 5, 5)
        self.assertFalse(result.isna().any().any())
        self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())


class PerformanceTests(unittest.TestCase):
    def test_known_three_period_metrics(self):
        result = synthetic_backtest([1, 1, 1], [0.8, 0.1, -0.05])
        metrics = performance_metrics(result, annualization_factor=3)
        returns = np.array([0, 0.1, -0.05])
        self.assertAlmostEqual(metrics["total_return"], 0.045)
        self.assertAlmostEqual(metrics["gross_total_return"], 0.045)
        self.assertAlmostEqual(metrics["annualized_return"], 0.045)
        self.assertAlmostEqual(metrics["annualized_volatility"], returns.std(ddof=1) * np.sqrt(3))
        self.assertAlmostEqual(metrics["sharpe_ratio"], returns.mean() / returns.std(ddof=1) * np.sqrt(3))
        self.assertAlmostEqual(metrics["max_drawdown"], -0.05)
        self.assertEqual(metrics["win_rate"], 0.5)
        self.assertEqual(metrics["number_of_trades"], 1)
        self.assertEqual(metrics["total_turnover"], 1.0)
        self.assertEqual(metrics["total_transaction_cost"], 0.0)

    def test_drawdown_includes_initial_capital_before_first_loss(self):
        frame = pd.DataFrame({"net_return": [-0.1, 0.05]})
        self.assertAlmostEqual(performance_metrics(frame)["max_drawdown"], -0.1)
        self.assertIsNone(performance_metrics(frame)["gross_total_return"])

    def test_gross_and_net_totals_are_compounded_separately(self):
        result = synthetic_backtest([1, -1, 0, 0], [0, 0.1, -0.1, 0], 5, 5)
        metrics = performance_metrics(result)
        self.assertAlmostEqual(metrics["gross_total_return"], 0.21)
        self.assertAlmostEqual(metrics["total_return"], 1.099 * 1.098 * 0.999 - 1)
        self.assertAlmostEqual(metrics["total_transaction_cost"], 0.004)
        self.assertEqual(metrics["number_of_trades"], 3)
        self.assertEqual(metrics["total_turnover"], 4)

    def test_flat_and_single_period_metrics_remain_finite(self):
        for count in (1, 4):
            metrics = performance_metrics(synthetic_backtest([0] * count, [0.1] * count))
            self.assertTrue(np.isfinite(list(metrics.values())).all())
            self.assertEqual(metrics["sharpe_ratio"], 0)
            self.assertEqual(metrics["win_rate"], 0)

    def test_missing_or_infinite_returns_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                performance_metrics(pd.DataFrame({"net_return": [0, value]}))

    def test_compare_strategies_includes_gross_return(self):
        data = synthetic_backtest([1, 1], [0, 0.1], 5, 5)
        table = compare_strategies({"first": data, "second": data})
        self.assertEqual(table["strategy"].tolist(), ["first", "second"])
        np.testing.assert_allclose(table["gross_total_return"], [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
