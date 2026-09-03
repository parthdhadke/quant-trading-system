import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from src.evaluation import chronological_split
from src.regime import CausalHMMRegimeModel
from src.regime_pipeline import CausalHMMRegimePipeline


@contextmanager
def writable_regime_test_directory():
    temporary_root = (Path(__file__).resolve().parent / ".tmp").resolve()
    temporary_directory = (temporary_root / uuid.uuid4().hex).resolve()

    if temporary_directory.parent != temporary_root:
        raise RuntimeError("Invalid temporary test directory")

    temporary_directory.mkdir(parents=True)

    try:
        yield temporary_directory
    finally:
        shutil.rmtree(temporary_directory)


def synthetic_prices(periods=320):
    index = pd.date_range("2022-01-03", periods=periods, freq="B")
    rng = np.random.default_rng(2026)
    first = 0.0018 + rng.normal(0, 0.003, periods // 3)
    second = -0.0015 + rng.normal(0, 0.012, periods // 3)
    third_length = periods - len(first) - len(second)
    third = 0.0006 + rng.normal(0, 0.006, third_length)
    returns = np.concatenate([first, second, third])
    close = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame({"Close": close}, index=index)


def feature_splits(volatility_window=10):
    model = CausalHMMRegimeModel(
        n_states=2,
        volatility_window=volatility_window,
        random_state=42,
        n_iter=100,
    )
    features = model.prepare_features(synthetic_prices())
    train, validation, test = chronological_split(features)
    return model, features, train, validation, test


class RegimeTests(unittest.TestCase):
    def test_features_use_only_current_and_past_prices(self):
        prices = synthetic_prices(periods=80)
        model = CausalHMMRegimeModel(volatility_window=5, n_iter=50)
        original = model.prepare_features(prices)
        changed = prices.copy()
        change_date = changed.index[60]
        changed.loc[change_date:, "Close"] *= np.linspace(
            1.5,
            2.0,
            len(changed.loc[change_date:]),
        )
        recalculated = model.prepare_features(changed)

        pd.testing.assert_frame_equal(
            original.loc[original.index < change_date],
            recalculated.loc[recalculated.index < change_date],
        )

        returns = prices["Close"].pct_change()
        expected = returns.iloc[16:21].std() * np.sqrt(252)
        self.assertAlmostEqual(original.loc[prices.index[20], "volatility"], expected)

    def test_chronological_split_preserves_order(self):
        _, features, train, validation, test = feature_splits()

        self.assertEqual(len(train), int(len(features) * 0.60))
        self.assertEqual(len(validation), int(len(features) * 0.80) - len(train))
        self.assertLess(train.index.max(), validation.index.min())
        self.assertLess(validation.index.max(), test.index.min())

    def test_scaler_and_hmm_receive_training_data_only(self):
        model, _, train, validation, test = feature_splits()
        original_fit = model.model.fit

        with patch.object(model.model, "fit", wraps=original_fit) as hmm_fit:
            model.fit(train)

        hmm_observations = hmm_fit.call_args.args[0]
        expected_mean = train[model.feature_columns].to_numpy().mean(axis=0)

        self.assertEqual(len(hmm_observations), len(train))
        self.assertEqual(model.training_observation_count_, len(train))
        self.assertEqual(model.scaler.n_samples_seen_, len(train))
        self.assertTrue(model.training_index_.equals(train.index))
        self.assertTrue(np.allclose(model.scaler.mean_, expected_mean))
        self.assertTrue(model.training_index_.intersection(validation.index).empty)
        self.assertTrue(model.training_index_.intersection(test.index).empty)

    def test_state_labels_and_summary_use_causal_training_states(self):
        model, _, train, _, _ = feature_splits()
        model.fit(train)
        inferred, _ = model.infer(train)
        expected = (
            inferred.groupby("state")
            .agg(
                mean_return=("return", "mean"),
                mean_volatility=("volatility", "mean"),
                observations=("return", "size"),
            )
            .reindex(range(model.n_states))
        )
        actual = model.get_state_summary()

        self.assertTrue(
            np.allclose(
                actual["mean_return"],
                expected["mean_return"],
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                actual["mean_volatility"],
                expected["mean_volatility"],
                equal_nan=True,
            )
        )
        self.assertTrue(
            np.allclose(
                actual["observations"],
                expected["observations"],
                equal_nan=True,
            )
        )
        ordered = actual["mean_return"].sort_values().index.tolist()
        self.assertEqual(actual.loc[ordered[0], "regime"], "bear")
        self.assertEqual(actual.loc[ordered[-1], "regime"], "bull")

    def test_validation_and_test_filtering_is_causal_and_continuous(self):
        model, features, train, validation, test = feature_splits()
        model.fit(train)
        train_result, train_posterior = model.infer(train)
        validation_result, validation_posterior = model.infer(
            validation,
            initial_probs=train_posterior,
        )
        test_result, _ = model.infer(
            test,
            initial_probs=validation_posterior,
        )
        prefix_length = len(validation) // 2
        validation_prefix, _ = model.infer(
            validation.iloc[:prefix_length],
            initial_probs=train_posterior,
        )
        continuous_result, _ = model.infer(features)
        probability_columns = [
            "state_0_probability",
            "state_1_probability",
        ]

        self.assertTrue(
            np.allclose(
                validation_prefix[probability_columns],
                validation_result.iloc[:prefix_length][probability_columns],
            )
        )
        sequential = pd.concat(
            [train_result, validation_result, test_result]
        )
        self.assertTrue(
            np.allclose(
                sequential[probability_columns],
                continuous_result[probability_columns],
            )
        )
        self.assertTrue(
            np.allclose(
                sequential[probability_columns].sum(axis=1),
                1.0,
            )
        )

    def test_processed_cache_is_reused(self):
        prices = synthetic_prices()
        start = prices.index.min()
        end = prices.index.max() + pd.Timedelta(days=1)

        with writable_regime_test_directory() as directory:
            pipeline = CausalHMMRegimePipeline(
                n_states=2,
                volatility_window=10,
                n_iter=100,
                processed_dir=directory,
            )
            first = pipeline.get_regimes("AAPL", start, end, prices)

            with patch.object(
                pipeline,
                "_compute_regimes",
                side_effect=AssertionError("Cache was not reused"),
            ):
                second = pipeline.get_regimes("AAPL", start, end, prices)

            pd.testing.assert_frame_equal(first, second)
            self.assertTrue(pipeline.loaded_from_cache_)
            self.assertEqual(
                set(second["dataset_split"]),
                {"train", "validation", "test"},
            )

    def test_material_configuration_changes_cache_path(self):
        common = {
            "ticker": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
        }
        two_states = CausalHMMRegimePipeline(n_states=2)
        three_states = CausalHMMRegimePipeline(n_states=3)
        wider_window = CausalHMMRegimePipeline(
            n_states=2,
            volatility_window=21,
        )

        two_state_path = two_states.cache_path(**common)

        self.assertNotEqual(two_state_path, three_states.cache_path(**common))
        self.assertNotEqual(two_state_path, wider_window.cache_path(**common))


if __name__ == "__main__":
    unittest.main()
