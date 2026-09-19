import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.run_ablation import load_cached_model, write_outputs
from src.ablation import STRATEGY_COLUMNS, run_ablation
from src.backtest import run_backtest
from tests.test_news import writable_test_directory


def synthetic_model_data():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=8, freq="B"),
        "dataset_split": ["train", "train", "validation", "validation", "test", "test", "test", "test"],
        "return": [0.9, 0.8, 0.7, 0.6, 0.5, 0.1, -0.05, 0.02],
        "sentiment_signal": [1, -1, 0, 1, 1, 0, -1, 1],
        "regime_signal": [-1, 1, 0, -1, -1, 1, 1, -1],
        "combined_signal": [1, 1, -1, 1, 0, 1, -1, 0],
    })


class AblationTests(unittest.TestCase):
    def test_exactly_four_named_strategies(self):
        result = run_ablation(synthetic_model_data())
        self.assertEqual(list(result.backtests), list(STRATEGY_COLUMNS))
        self.assertEqual(result.comparison["strategy"].tolist(), list(STRATEGY_COLUMNS))

    def test_identical_test_dates_and_market_returns_for_all_strategies(self):
        data = synthetic_model_data()
        test = data.loc[data["dataset_split"] == "test"].set_index("date")
        result = run_ablation(data)
        for frame in result.backtests.values():
            self.assertTrue(frame.index.equals(test.index))
            pd.testing.assert_series_equal(frame["return"], test["return"])
            self.assertTrue(frame["dataset_split"].eq("test").all())

    def test_first_test_row_flat_for_every_strategy_without_validation_carry(self):
        result = run_ablation(synthetic_model_data())
        for frame in result.backtests.values():
            self.assertEqual(frame["executed_position"].iloc[0], 0)
            self.assertEqual(frame["net_return"].iloc[0], 0)
            self.assertEqual(frame["turnover"].iloc[0], 0)

    def test_buy_and_hold_constant_targets_use_same_delay(self):
        result = run_ablation(synthetic_model_data())
        np.testing.assert_array_equal(result.backtests["Buy & Hold"]["target_position"], [1, 1, 1, 1])
        np.testing.assert_array_equal(result.backtests["Buy & Hold"]["executed_position"], [0, 1, 1, 1])

    def test_existing_signals_used_without_double_shifting_or_sizing(self):
        data = synthetic_model_data()
        test = data.loc[data["dataset_split"] == "test"]
        result = run_ablation(data)
        for name, column in STRATEGY_COLUMNS.items():
            if column:
                np.testing.assert_array_equal(result.backtests[name]["target_position"], test[column])
                np.testing.assert_array_equal(result.backtests[name]["executed_position"], test[column].shift(1).fillna(0))

    def test_only_existing_backtester_is_used_with_identical_costs(self):
        with patch("src.ablation.run_backtest", wraps=run_backtest) as mocked:
            run_ablation(synthetic_model_data(), commission_bps=5, slippage_bps=5)
        self.assertEqual(mocked.call_count, 4)
        for call in mocked.call_args_list:
            self.assertTrue(call.args[0]["dataset_split"].eq("test").all())
            self.assertEqual(call.kwargs["commission_bps"], 5)
            self.assertEqual(call.kwargs["slippage_bps"], 5)

    def test_train_validation_values_cannot_influence_test_performance(self):
        data = synthetic_model_data()
        expected = run_ablation(data)
        data.loc[data["dataset_split"] != "test", ["return", "sentiment_signal", "regime_signal", "combined_signal"]] = 500
        changed = run_ablation(data)
        pd.testing.assert_frame_equal(expected.comparison, changed.comparison)
        pd.testing.assert_frame_equal(expected.equity_curves, changed.equity_curves)

    def test_future_return_columns_are_ignored_and_not_propagated(self):
        data = synthetic_model_data()
        expected = run_ablation(data)
        data["future_return"] = np.inf
        data["tomorrow_return"] = np.nan
        result = run_ablation(data)
        for name in result.backtests:
            pd.testing.assert_frame_equal(result.backtests[name], expected.backtests[name])

    def test_later_target_and_return_changes_do_not_change_earlier_output(self):
        data = synthetic_model_data()
        expected = run_ablation(data)
        data.loc[7, ["return", "sentiment_signal", "regime_signal", "combined_signal"]] = [0.4, -1, 1, 1]
        result = run_ablation(data)
        for name in result.backtests:
            pd.testing.assert_frame_equal(result.backtests[name].iloc[:-1], expected.backtests[name].iloc[:-1])

    def test_invalid_test_signals_or_returns_rejected(self):
        for column in ("sentiment_signal", "regime_signal", "combined_signal", "return"):
            data = synthetic_model_data()
            data[column] = data[column].astype(float)
            data.loc[5, column] = np.nan
            with self.subTest(column=column), self.assertRaises(ValueError):
                run_ablation(data)
        data = synthetic_model_data()
        data.loc[5, "sentiment_signal"] = 2
        with self.assertRaises(ValueError):
            run_ablation(data)

    def test_absent_test_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "No test rows"):
            run_ablation(synthetic_model_data().iloc[:4])

    def test_duplicate_or_shuffled_dates_rejected(self):
        data = synthetic_model_data()
        for invalid in (pd.concat([data, data.iloc[-1:]]), data.iloc[::-1]):
            with self.assertRaisesRegex(ValueError, "sorted, unique"):
                run_ablation(invalid)

    def test_split_reordering_rejected(self):
        data = synthetic_model_data()
        data.loc[6, "dataset_split"] = "train"
        with self.assertRaisesRegex(ValueError, "chronological"):
            run_ablation(data)

    def test_equity_curves_and_position_counts_match_backtests(self):
        result = run_ablation(synthetic_model_data())
        counts = result.position_counts.set_index("strategy")
        for name, frame in result.backtests.items():
            np.testing.assert_array_equal(result.equity_curves[name], frame["equity"])
            self.assertEqual(counts.loc[name, ["executed_long_days", "executed_neutral_days", "executed_short_days"]].sum(), 4)
            self.assertEqual(counts.loc[name, ["target_long_days", "target_neutral_days", "target_short_days"]].sum(), 4)
        np.testing.assert_array_equal(counts.loc["Combined", ["executed_long_days", "executed_neutral_days", "executed_short_days"]], [1, 2, 1])

    def test_outputs_are_finite_and_do_not_mutate_input(self):
        data = synthetic_model_data()
        original = data.copy(deep=True)
        result = run_ablation(data)
        pd.testing.assert_frame_equal(data, original)
        for frame in [*result.backtests.values(), result.comparison, result.equity_curves]:
            self.assertFalse(frame.isna().any().any())
            self.assertTrue(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all())

    def test_report_files_round_trip_and_record_assumptions(self):
        with writable_test_directory() as root:
            data = synthetic_model_data()
            model_path = root / "model.parquet"
            data.to_parquet(model_path, index=False)
            result = run_ablation(data)
            directory = write_outputs(result, model_path, root / "reports")
            pd.testing.assert_frame_equal(pd.read_parquet(directory / "comparison.parquet"), result.comparison)
            pd.testing.assert_frame_equal(pd.read_parquet(directory / "equity_curves.parquet"), result.equity_curves)
            self.assertTrue((directory / "combined.parquet").is_file())
            self.assertIn('"start_flat": true', (directory / "metadata.json").read_text())

    def test_missing_cache_never_triggers_network_or_model_fitting(self):
        with writable_test_directory() as root:
            with patch("requests.sessions.Session.request", side_effect=AssertionError("No network")) as network:
                with patch("src.regime.CausalHMMRegimeModel.fit", side_effect=AssertionError("No fitting")) as fit:
                    with self.assertRaisesRegex(FileNotFoundError, "No downloading or refitting"):
                        load_cached_model(root)
                    network.assert_not_called()
                    fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
