import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.integration import (
    ModelDatasetPipeline,
    SIGNAL_COLUMNS,
    build_model_dataset,
    validate_model_dataset,
)
from src.regime_pipeline import CausalHMMRegimePipeline
from tests.test_news import writable_test_directory


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime([
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
            "2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
        ])
        self.prices = pd.DataFrame({"Close": np.arange(100.0, 109.0)}, index=self.dates)
        scores = np.array([0, 0, 0.4, -0.4, 0, 0.2, -0.2, 0.7, -0.7])
        self.sentiment = pd.DataFrame({
            "date": self.dates,
            "sentiment_score": scores,
            "positive": np.maximum(scores, 0),
            "neutral": 1 - np.abs(scores),
            "negative": np.maximum(-scores, 0),
            "headline_count": (scores != 0).astype(int),
        })
        states = np.array([1, 0, 1, 0, 1, 0, 1])
        labels = np.where(states == 1, "bull", "bear")
        self.regimes = pd.DataFrame({
            "date": self.dates[2:],
            "return": self.prices["Close"].pct_change().iloc[2:].to_numpy(),
            "volatility": 0.15,
            "state": states,
            "state_probability": 0.8,
            "state_0_probability": np.where(states == 0, 0.8, 0.2),
            "state_1_probability": np.where(states == 1, 0.8, 0.2),
            "regime": labels,
            "volatility_regime": "low_volatility",
            "regime_label": [f"{label}_low_volatility" for label in labels],
            "dataset_split": ["train"] * 3 + ["validation"] * 2 + ["test"] * 2,
        })
        self.range = dict(ticker="AAPL", start_date="2024-01-08", end_date="2024-01-20")

    def build(self, **settings):
        return build_model_dataset(
            self.prices, self.sentiment, self.regimes, **self.range, **settings,
        )

    def test_exact_trading_date_alignment(self):
        result = self.build()
        self.assertEqual(result["date"].tolist(), self.regimes["date"].tolist())
        self.assertTrue(result["date"].isin(self.prices.index).all())
        np.testing.assert_allclose(result["sentiment_score"], self.sentiment["sentiment_score"].iloc[2:])
        np.testing.assert_allclose(result["return"], self.regimes["return"])
        np.testing.assert_allclose(result["close"], self.prices["Close"].iloc[2:])

    def test_non_trading_news_and_regimes_do_not_create_rows(self):
        original = self.build()
        for name in ("sentiment", "regimes"):
            data = getattr(self, name)
            extra = data.iloc[[0]].copy()
            extra["date"] = pd.Timestamp("2024-01-13")
            setattr(self, name, pd.concat([data, extra], ignore_index=True))
        pd.testing.assert_frame_equal(original, self.build())
        self.assertNotIn(pd.Timestamp("2024-01-15"), original["date"].tolist())

    def test_no_news_rows_stay_neutral_without_filling(self):
        result = self.build().set_index("date")
        values = result.loc["2024-01-12", ["sentiment_score", "positive", "neutral", "negative", "headline_count", "sentiment_signal"]]
        np.testing.assert_allclose(values.to_numpy(dtype=float), [0, 0, 1, 0, 0, 0])

    def test_missing_sentiment_is_not_backfilled_or_forward_filled(self):
        self.sentiment = self.sentiment[self.sentiment["date"] != pd.Timestamp("2024-01-12")]
        with self.assertRaisesRegex(ValueError, "Missing daily sentiment"):
            self.build()

    def test_missing_regimes_are_not_backfilled(self):
        missing_date = self.regimes.iloc[0]["date"]
        self.regimes = self.regimes.iloc[1:]
        self.assertNotIn(missing_date, self.build()["date"].tolist())

    def test_rolling_warmup_rows_are_dropped(self):
        warmup = self.regimes.iloc[[0]].copy()
        warmup["date"] = self.dates[1]
        warmup[["return", "volatility", "state"]] = np.nan
        self.regimes = pd.concat([warmup, self.regimes], ignore_index=True)
        result = self.build()
        self.assertEqual(result["date"].tolist(), self.dates[2:].tolist())
        self.assertFalse(result.isna().any().any())

    def test_original_splits_are_preserved_even_for_a_test_only_slice(self):
        result = self.build()
        self.assertEqual(result["dataset_split"].tolist(), self.regimes["dataset_split"].tolist())
        self.range["start_date"] = "2024-01-18"
        self.assertEqual(self.build()["dataset_split"].tolist(), ["test", "test"])

    def test_signal_domains_and_same_day_targets(self):
        result = self.build()
        for column in SIGNAL_COLUMNS:
            self.assertTrue(result[column].isin([-1, 0, 1]).all())
        self.assertEqual(result["sentiment_signal"].tolist(), [1, -1, 0, 0, 0, 1, -1])
        self.assertEqual(result["regime_signal"].tolist(), [1, -1, 1, -1, 1, -1, 1])
        self.assertEqual(result["combined_signal"].tolist(), [1, -1, 1, -1, 1, 0, 0])
        self.assertNotIn("executed_position", result)
        self.assertNotIn("net_return", result)

    def test_threshold_changes_change_sentiment_but_not_regime_signal(self):
        baseline = self.build()
        changed = self.build(buy_threshold=0.5, sell_threshold=-0.5)
        self.assertEqual(changed["sentiment_signal"].tolist(), [0, 0, 0, 0, 0, 1, -1])
        pd.testing.assert_series_equal(changed["regime_signal"], baseline["regime_signal"])

    def test_future_return_columns_are_not_used_or_propagated(self):
        expected = self.build()
        for frame in (self.prices, self.sentiment, self.regimes):
            frame["future_return"] = np.inf
        pd.testing.assert_frame_equal(expected, self.build())

    def test_later_sentiment_and_regimes_do_not_change_earlier_signals(self):
        expected = self.build().iloc[:-1]
        self.sentiment.loc[self.sentiment.index[-1], ["sentiment_score", "positive", "neutral", "negative"]] = [0.9, 0.9, 0.1, 0]
        self.regimes.loc[self.regimes.index[-1], ["state", "regime", "regime_label", "state_0_probability", "state_1_probability"]] = [0, "bear", "bear_low_volatility", 0.8, 0.2]
        pd.testing.assert_frame_equal(expected, self.build().iloc[:-1])

    def test_duplicate_dates_are_rejected(self):
        self.sentiment = pd.concat([self.sentiment, self.sentiment.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "duplicate trading dates"):
            self.build()

    def test_missing_and_infinite_required_values_are_rejected(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                self.sentiment.loc[2, "sentiment_score"] = value
                with self.assertRaises(ValueError):
                    self.build()

    def test_intraday_sentiment_is_not_mapped_back_to_a_daily_row(self):
        self.sentiment.loc[2, "date"] += pd.Timedelta(hours=23)
        with self.assertRaisesRegex(ValueError, "intraday"):
            self.build()

    def test_multiindex_prices_select_requested_ticker(self):
        expected = self.build()
        self.prices = pd.concat({"AAPL": self.prices, "MSFT": self.prices * 2}, axis=1)
        pd.testing.assert_frame_equal(expected, self.build())

    def test_cache_is_reused_without_regenerating_signals(self):
        with writable_test_directory() as directory:
            pipeline = ModelDatasetPipeline(CausalHMMRegimePipeline(), processed_dir=directory)
            arguments = dict(**self.range, price_data=self.prices, sentiment_data=self.sentiment, regime_data=self.regimes)
            first = pipeline.get_model_dataset(**arguments)
            with patch("src.integration.build_model_dataset", side_effect=AssertionError("Unexpected regeneration")):
                second = pipeline.get_model_dataset(**arguments)
            self.assertTrue(pipeline.loaded_from_cache_)
            self.assertTrue(pipeline.cache_path_.exists())
            pd.testing.assert_frame_equal(first, second)

    def test_material_signal_settings_produce_different_cache_paths(self):
        arguments = dict(**self.range, price_data=self.prices, sentiment_data=self.sentiment, regime_data=self.regimes)
        baseline = ModelDatasetPipeline(CausalHMMRegimePipeline()).cache_path(**arguments)
        for setting in (
            {"buy_threshold": 0.5}, {"sell_threshold": -0.5},
            {"sentiment_weight": 0.7}, {"regime_weight": 0.7},
            {"decision_threshold": 0.75},
        ):
            with self.subTest(setting=setting):
                other = ModelDatasetPipeline(CausalHMMRegimePipeline(), **setting)
                self.assertNotEqual(baseline, other.cache_path(**arguments))

    def test_changed_hmm_config_or_source_content_invalidates_cache(self):
        arguments = dict(**self.range, price_data=self.prices, sentiment_data=self.sentiment, regime_data=self.regimes)
        pipeline = ModelDatasetPipeline(CausalHMMRegimePipeline())
        baseline = pipeline.cache_path(**arguments)
        for setting in (
            {"volatility_window": 21}, {"random_state": 7}, {"n_iter": 100},
            {"train_ratio": 0.5}, {"validation_ratio": 0.1},
        ):
            other = ModelDatasetPipeline(CausalHMMRegimePipeline(**setting))
            self.assertNotEqual(baseline, other.cache_path(**arguments))
        self.sentiment.loc[2, ["sentiment_score", "positive", "neutral"]] = [0.6, 0.6, 0.4]
        self.assertNotEqual(baseline, pipeline.cache_path(**arguments))
        three_state_data = self.regimes.assign(state_2_probability=0.0)
        three_state_pipeline = ModelDatasetPipeline(CausalHMMRegimePipeline(n_states=3))
        arguments["regime_data"] = three_state_data
        self.assertNotEqual(baseline, three_state_pipeline.cache_path(**arguments))

    def test_full_year_does_not_silently_use_partial_sentiment_coverage(self):
        self.sentiment = self.sentiment.tail(2)
        with self.assertRaisesRegex(ValueError, "Missing daily sentiment for 5"):
            self.build()

    def test_validation_rejects_non_ternary_signals(self):
        result = self.build()
        result.loc[0, "combined_signal"] = 2
        with self.assertRaisesRegex(ValueError, "only -1, 0, 1"):
            validate_model_dataset(result)


if __name__ == "__main__":
    unittest.main()
