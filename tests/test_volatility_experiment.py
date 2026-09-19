import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import TARGET_VOLATILITY
from scripts.run_volatility_sizing import write_outputs
from src.ablation import run_ablation
from src.backtest import run_backtest
from src.volatility_experiment import run_volatility_comparison, validation_preflight
from tests.test_ablation import synthetic_model_data
from tests.test_news import writable_test_directory


def sizing_data():
    data = synthetic_model_data()
    data["volatility"] = [0.1, 0.2, 0.3, 0.6, 0.15, 0.3, 0.6, 0.2]
    return data


class VolatilityExperimentTests(unittest.TestCase):
    def test_seven_named_strategies(self):
        result = run_volatility_comparison(sizing_data())
        self.assertEqual(list(result.backtests), [
            "Buy & Hold", "Sentiment Fixed", "Sentiment Vol Sized", "Regime Fixed",
            "Regime Vol Sized", "Combined Fixed", "Combined Vol Sized",
        ])
        self.assertEqual(result.comparison["strategy"].tolist(), list(result.backtests))

    def test_fixed_outputs_equal_existing_ablation(self):
        baseline = run_ablation(sizing_data())
        result = run_volatility_comparison(sizing_data())
        for name, original in (("Buy & Hold", "Buy & Hold"), ("Sentiment Fixed", "Sentiment Only"),
                               ("Regime Fixed", "Regime Only"), ("Combined Fixed", "Combined")):
            pd.testing.assert_frame_equal(result.backtests[name], baseline.backtests[original])

    def test_identical_dates_returns_costs_and_single_delay(self):
        result = run_volatility_comparison(sizing_data(), commission_bps=3, slippage_bps=7)
        for frame in result.backtests.values():
            self.assertTrue(frame.index.equals(result.test_data.index))
            pd.testing.assert_series_equal(frame["return"], result.test_data["return"])
            np.testing.assert_array_equal(frame["executed_position"], frame["target_position"].shift(1).fillna(0))
            np.testing.assert_allclose(frame["transaction_cost"], frame["turnover"] * 0.001)
            self.assertEqual(frame["net_return"].iloc[0], 0)

    def test_sized_directions_preserved_without_rounding(self):
        result = run_volatility_comparison(sizing_data())
        for prefix in ("Sentiment", "Regime", "Combined"):
            signal = f"{prefix.lower()}_signal"
            sized = result.backtests[f"{prefix} Vol Sized"]
            pd.testing.assert_series_equal(sized[signal], result.test_data[signal])
            np.testing.assert_allclose(sized["target_position"], sized[signal] * sized["position_size"])
            self.assertTrue(sized["position_size"].between(0, 1).all())
        self.assertEqual(result.backtests["Combined Vol Sized"]["target_position"].iloc[1], 0.5)

    def test_exposure_metrics_use_executed_positions_including_flat_rows(self):
        result = run_volatility_comparison(sizing_data())
        metrics = result.comparison.set_index("strategy")
        for name, frame in result.backtests.items():
            self.assertEqual(metrics.loc[name, "average_absolute_exposure"], frame["executed_position"].abs().mean())
            self.assertEqual(metrics.loc[name, "maximum_absolute_exposure"], frame["executed_position"].abs().max())
        self.assertEqual(metrics.loc["Combined Vol Sized", "average_absolute_exposure"], 0.1875)

    def test_future_volatility_and_returns_do_not_change_earlier_output(self):
        data = sizing_data()
        expected = run_volatility_comparison(data)
        data.loc[7, ["volatility", "return"]] = [1.5, 0.4]
        changed = run_volatility_comparison(data)
        for name in expected.backtests:
            pd.testing.assert_frame_equal(expected.backtests[name].iloc[:-1], changed.backtests[name].iloc[:-1])

    def test_prespecified_target_cannot_be_selected_by_test_data(self):
        data = sizing_data()
        expected = run_volatility_comparison(data)
        data.loc[data["dataset_split"].eq("test"), ["return", "volatility"]] = [0.01, 0.9]
        changed = run_volatility_comparison(data)
        self.assertEqual(expected.settings, changed.settings)
        self.assertEqual(changed.settings["target_volatility"], TARGET_VOLATILITY)
        self.assertEqual(changed.settings["target_selection"], "prespecified_no_search")

    def test_validation_preflight_accepts_only_validation(self):
        data = sizing_data()
        for invalid in (data, data.loc[data["dataset_split"].eq("test")], data.iloc[:0]):
            with self.assertRaisesRegex(ValueError, "validation rows only"):
                validation_preflight(invalid)
        result = validation_preflight(data.loc[data["dataset_split"].eq("validation")])
        self.assertTrue(result["dataset_split"].eq("validation").all())
        np.testing.assert_allclose(result["executed_position"], [0, -0.5])

    def test_test_changes_cannot_influence_validation_preflight(self):
        data = sizing_data()
        data["combined_signal"] = data["combined_signal"].astype(float)
        expected = validation_preflight(data.loc[data["dataset_split"].eq("validation")])
        data.loc[data["dataset_split"].eq("test"), ["return", "volatility", "combined_signal"]] = np.nan
        actual = validation_preflight(data.loc[data["dataset_split"].eq("validation")])
        pd.testing.assert_frame_equal(expected, actual)

    def test_invalid_cached_volatility_rejected_before_any_backtest(self):
        for value in (np.nan, np.inf, -np.inf, -0.1):
            data = sizing_data()
            data.loc[5, "volatility"] = value
            with self.subTest(value=value), patch("src.ablation.run_backtest") as baseline:
                with patch("src.volatility_experiment.run_backtest") as sized:
                    with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                        run_volatility_comparison(data)
                    baseline.assert_not_called()
                    sized.assert_not_called()

    def test_zero_volatility_is_finite_and_capped(self):
        data = sizing_data()
        data["volatility"] = 0
        result = run_volatility_comparison(data)
        for prefix in ("Sentiment", "Regime", "Combined"):
            frame = result.backtests[f"{prefix} Vol Sized"]
            self.assertTrue(frame["position_size"].eq(1).all())
            np.testing.assert_array_equal(frame["net_return"], result.backtests[f"{prefix} Fixed"]["net_return"])

    def test_fractional_rebalancing_without_direction_change_has_cost(self):
        data = sizing_data()
        data["combined_signal"] = 1
        frame = run_volatility_comparison(data).backtests["Combined Vol Sized"]
        np.testing.assert_allclose(frame["turnover"], [0, 1, 0.5, 0.25])
        np.testing.assert_allclose(frame["transaction_cost"], [0, 0.001, 0.0005, 0.00025])

    def test_existing_backtester_used_for_three_sized_strategies(self):
        with patch("src.volatility_experiment.run_backtest", wraps=run_backtest) as mocked:
            run_volatility_comparison(sizing_data())
        self.assertEqual(mocked.call_count, 3)
        for call in mocked.call_args_list:
            self.assertTrue(call.args[0]["dataset_split"].eq("test").all())
            self.assertEqual(call.args[1], "position")

    def test_outputs_finite_input_unchanged_and_future_columns_ignored(self):
        data = sizing_data()
        original = data.copy(deep=True)
        expected = run_volatility_comparison(data)
        pd.testing.assert_frame_equal(data, original)
        data["future_return"] = np.inf
        data["future_volatility"] = np.nan
        actual = run_volatility_comparison(data)
        for name, frame in actual.backtests.items():
            pd.testing.assert_frame_equal(frame, expected.backtests[name])
            self.assertFalse(frame.isna().any().any())
            self.assertTrue(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all())

    def test_datetime_index_supported_and_bad_dates_rejected(self):
        data = sizing_data()
        expected = run_volatility_comparison(data)
        actual = run_volatility_comparison(data.set_index("date"))
        pd.testing.assert_frame_equal(expected.comparison, actual.comparison)
        for invalid in (data.iloc[::-1], pd.concat([data, data.iloc[-1:]])):
            with self.assertRaisesRegex(ValueError, "sorted, unique"):
                run_volatility_comparison(invalid)

    def test_reports_round_trip_without_overwriting_baseline(self):
        from scripts.run_ablation import write_outputs as write_baseline

        with writable_test_directory() as root:
            data = sizing_data()
            model_path = root / "model.parquet"
            data.to_parquet(model_path, index=False)
            baseline_directory = write_baseline(run_ablation(data), model_path, root / "reports")
            before = {path.name: path.read_bytes() for path in baseline_directory.iterdir()}
            result = run_volatility_comparison(data)
            validation = validation_preflight(data.loc[data["dataset_split"].eq("validation")])
            directory = write_outputs(result, validation, model_path, root / "reports")
            self.assertNotEqual(directory, baseline_directory)
            self.assertEqual(before, {path.name: path.read_bytes() for path in baseline_directory.iterdir()})
            pd.testing.assert_frame_equal(pd.read_parquet(directory / "comparison.parquet"), result.comparison)
            pd.testing.assert_frame_equal(pd.read_parquet(directory / "combined_vol_sized.parquet"), result.backtests["Combined Vol Sized"])
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["settings"]["target_selection"], "prespecified_no_search")
            self.assertEqual(metadata["validation_rows"], 2)
            self.assertTrue((directory / "combined_last10.txt").is_file())


if __name__ == "__main__":
    unittest.main()
