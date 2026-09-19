import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ablation import run_ablation
from src.regime import CausalHMMRegimeModel
from src.tuning import (
    evaluate_selected_test, generate_parameter_grid, get_validation_regimes,
    robustness_summary, search_validation_parameters, select_best_validation_config,
    validate_signal_config, validation_regime_cache_path,
)
from tests.test_news import writable_test_directory
from tests.test_regime import synthetic_prices


def validation_data():
    return pd.DataFrame({
        "date": pd.date_range("2024-08-01", periods=6, freq="B"),
        "return": [0, 0, 0.1, -0.1, 0.1, -0.1],
        "sentiment_score": [0, 0.3, -0.3, 0.3, -0.3, 0.3],
        "regime": "neutral", "dataset_split": "validation",
    })


def small_grid():
    return [
        {"buy_threshold": 0.2, "sell_threshold": -0.2, "sentiment_weight": 1,
         "regime_weight": 0, "decision_threshold": 0.5},
        {"buy_threshold": 0.4, "sell_threshold": -0.4, "sentiment_weight": 1,
         "regime_weight": 0, "decision_threshold": 0.5},
    ]


class TuningTests(unittest.TestCase):
    def test_default_grid_is_deterministic_and_has_225_candidates(self):
        first, second = generate_parameter_grid(), generate_parameter_grid()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 225)
        self.assertEqual(first[0], {"buy_threshold": 0.2, "sell_threshold": -0.2,
                                   "sentiment_weight": 0.25, "regime_weight": 0.75,
                                   "decision_threshold": 0.25})

    def test_invalid_threshold_pairs_rejected(self):
        for buy, sell in ((-0.1, -0.2), (0.2, 0.2), (1.1, -0.2)):
            with self.subTest(buy=buy, sell=sell), self.assertRaises(ValueError):
                validate_signal_config({"buy_threshold": buy, "sell_threshold": sell,
                                        "sentiment_weight": 0.5, "regime_weight": 0.5,
                                        "decision_threshold": 0.5})

    def test_weights_must_sum_to_one(self):
        for weights in ((0.5, 0.6), (-0.1, 1.1), (np.inf, -np.inf)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                validate_signal_config({"buy_threshold": 0.2, "sell_threshold": -0.2,
                                        "sentiment_weight": weights[0], "regime_weight": weights[1],
                                        "decision_threshold": 0.5})

    def test_tuner_rejects_train_or_test_rows(self):
        for split in ("train", "test"):
            data = validation_data()
            data.loc[0, "dataset_split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "validation rows only"):
                search_validation_parameters(data, 2, small_grid())

    def test_known_synthetic_example_selects_active_profitable_candidate(self):
        results = search_validation_parameters(validation_data(), 2, small_grid(), minimum_trades=2,
                                               commission_bps=0, slippage_bps=0)
        selected = select_best_validation_config(results)
        self.assertEqual(selected["buy_threshold"], 0.2)
        self.assertGreater(selected["validation_sharpe_ratio"], 0)
        inactive = results.loc[results["buy_threshold"] == 0.4].iloc[0]
        self.assertFalse(inactive["eligible"])
        self.assertEqual(inactive["rejection_reason"], "insufficient_trading_activity")

    def test_changing_unpassed_test_returns_cannot_change_selection(self):
        validation = validation_data()
        first = select_best_validation_config(search_validation_parameters(validation, 2, small_grid()))
        test = pd.DataFrame({"return": [0.2, -0.9]})
        test["return"] *= -100
        second = select_best_validation_config(search_validation_parameters(validation, 2, small_grid()))
        self.assertEqual(first, second)

    def test_nonfinite_candidate_metrics_are_rejected_and_recorded(self):
        metrics = {"total_return": 0, "gross_total_return": 0, "annualized_return": 0,
                   "annualized_volatility": 0, "sharpe_ratio": np.inf, "max_drawdown": 0,
                   "win_rate": 0, "number_of_trades": 3, "total_turnover": 3,
                   "total_transaction_cost": 0}
        with patch("src.tuning.performance_metrics", return_value=metrics):
            with self.assertRaisesRegex(ValueError, "No eligible"):
                search_validation_parameters(validation_data(), 2, small_grid()[:1])

    def test_hmm_state_count_cache_names_do_not_collide(self):
        first = validation_regime_cache_path("cache", "AAPL", "2024-01-01", "2025-01-01", 2, "a" * 64)
        second = validation_regime_cache_path("cache", "AAPL", "2024-01-01", "2025-01-01", 3, "a" * 64)
        self.assertNotEqual(first, second)
        self.assertIn("_s2_", first.name)
        self.assertIn("_s3_", second.name)

    def test_validation_hmm_fits_train_only_and_never_infers_test(self):
        prices = synthetic_prices(150)
        with writable_test_directory() as root:
            original_fit = CausalHMMRegimeModel.fit
            original_infer = CausalHMMRegimeModel.infer
            with patch.object(CausalHMMRegimeModel, "fit", autospec=True, side_effect=original_fit) as fitted:
                with patch.object(CausalHMMRegimeModel, "infer", autospec=True, side_effect=original_infer) as inferred:
                    _, result = get_validation_regimes(
                        prices, "AAPL", prices.index.min(), prices.index.max() + pd.Timedelta(days=1), 2,
                        cache_dir=root, volatility_window=5, n_iter=30,
                    )
            fit_rows = fitted.call_args.args[1]
            inferred_rows = [call.args[1] for call in inferred.call_args_list]
            self.assertEqual(len(inferred_rows), 2)
            self.assertTrue(fit_rows.index.equals(inferred_rows[0].index))
            self.assertTrue(result["dataset_split"].eq("validation").all())
            metadata = result.attrs["methodology"]
            self.assertEqual(metadata["test_rows_inferred"], 0)
            self.assertGreater(metadata["excluded_test_rows"], 0)
            self.assertLess(fit_rows.index.max(), result["date"].min())

    def test_validation_hmm_cache_reuse_avoids_refit(self):
        prices = synthetic_prices(120)
        with writable_test_directory() as root:
            path, first = get_validation_regimes(prices, "AAPL", "2022-01-01", "2023-01-01", 2,
                                                 cache_dir=root, volatility_window=5, n_iter=30)
            with patch.object(CausalHMMRegimeModel, "fit", side_effect=AssertionError("refit")):
                reused_path, second = get_validation_regimes(prices, "AAPL", "2022-01-01", "2023-01-01", 2,
                                                             cache_dir=root, volatility_window=5, n_iter=30)
            self.assertEqual(path, reused_path)
            pd.testing.assert_frame_equal(first, second)

    def test_robustness_includes_neighbor_and_alternate_hmm(self):
        frames = [search_validation_parameters(validation_data(), states, generate_parameter_grid(
            buy_thresholds=(0.2, 0.25), sell_thresholds=(-0.2,), weight_pairs=((1, 0),),
            decision_thresholds=(0.25, 0.5)), minimum_trades=1, commission_bps=0, slippage_bps=0)
            for states in (2, 3)]
        results = pd.concat(frames, ignore_index=True).sort_values(
            ["sharpe_ratio", "config_id"], ascending=[False, True]).reset_index(drop=True)
        selected = select_best_validation_config(results)
        nearby, summary = robustness_summary(results, selected)
        self.assertIn("alternate_hmm_state_count", nearby["relationship"].tolist())
        self.assertTrue(any(value.startswith("nearby_") for value in nearby["relationship"]))
        self.assertIn(summary["classification"], ["stable", "moderately sensitive", "highly sensitive"])

    def test_final_test_evaluation_requires_validation_lock_and_runs_once(self):
        data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3), "dataset_split": "test",
            "return": [0, 0.1, -0.1], "sentiment_score": [0, 0.5, -0.5], "regime": ["bull", "bear", "bull"],
        })
        config = {**small_grid()[0], "hmm_state_count": 2, "selection_split": "validation"}
        with patch("src.tuning.run_ablation", wraps=run_ablation) as evaluator:
            result = evaluate_selected_test(data, config, commission_bps=0, slippage_bps=0)
        self.assertEqual(evaluator.call_count, 1)
        self.assertIn("Combined Tuned", result.backtests)
        bad = {**config, "selection_split": "test"}
        with self.assertRaisesRegex(ValueError, "validation-locked"):
            evaluate_selected_test(data, bad)


if __name__ == "__main__":
    unittest.main()
