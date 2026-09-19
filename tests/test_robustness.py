import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline import get_price_data
from src.news import AlphaVantageNewsClient, NewsSentimentPipeline
from src.robustness import MultiTickerRobustness, load_global_settings, summarize_results
from src.robustness_data import RobustnessDataProvider
from tests.test_news import FakeSentimentModel, writable_test_directory
from tests.test_news_backfill import RangeSession
from tests.test_volatility_experiment import sizing_data


def settings():
    return {
        "signals": {"buy_threshold": 0.2, "sell_threshold": -0.25, "sentiment_weight": 0.75,
                    "regime_weight": 0.25, "decision_threshold": 0.5},
        "hmm": {"n_states": 3, "volatility_window": 20, "random_state": 42, "n_iter": 500,
                "train_ratio": 0.6, "validation_ratio": 0.2},
        "evaluation": {"commission_bps": 5, "slippage_bps": 5, "initial_capital": 100000,
                       "target_volatility": 0.15, "max_position_size": 1, "min_position_size": 0},
    }


class FakeProvider:
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = outcomes or {}

    def inspect(self, ticker):
        return {"missing_news_ranges": [["2024-01-01", "2025-01-01"]], "price_cache_exists": False}

    def safe_error(self, error):
        return str(error)

    def prepare(self, ticker):
        self.calls.append(ticker)
        outcome = self.outcomes.get(ticker)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, dict):
            return outcome
        data = sizing_data()
        if ticker != "AAPL":
            data["date"] += pd.Timedelta(days=20)
            data["return"] *= -0.25
        return {"status": "ready", "model": data, "provenance": {"ticker": ticker}}


class RobustnessTests(unittest.TestCase):
    def run_basket(self, root, provider=None, tickers=("AAPL", "MSFT")):
        runner = MultiTickerRobustness(root, settings(), tickers)
        result = runner.run(provider or FakeProvider())
        return runner, result

    def test_multiple_tickers_independent_and_dates_preserved(self):
        with writable_test_directory() as root:
            runner, (results, _, states) = self.run_basket(root)
            self.assertEqual(len(results), 10)
            self.assertEqual(set(states), {"AAPL", "MSFT"})
            self.assertNotEqual(states["AAPL"]["test_start"], states["MSFT"]["test_start"])
            for ticker in states:
                saved = pd.read_parquet(runner.directory / ticker / "combined.parquet")
                self.assertTrue(saved["date"].is_monotonic_increasing)
                self.assertTrue(saved["dataset_split"].eq("test").all())
                self.assertEqual(saved["executed_position"].iloc[0], 0)

    def test_ticker_failure_preserves_others(self):
        with writable_test_directory() as root:
            runner, (results, _, states) = self.run_basket(root, FakeProvider({"MSFT": ValueError("bad source")}))
            self.assertEqual(states["AAPL"]["status"], "completed")
            self.assertEqual(states["MSFT"]["status"], "failed")
            self.assertEqual(set(results["ticker"]), {"AAPL"})
            self.assertTrue((runner.directory / "AAPL/comparison.parquet").is_file())

    def test_quota_stops_remaining_tickers_and_lists_whole_basket(self):
        with writable_test_directory() as root:
            provider = FakeProvider({"MSFT": {"status": "incomplete", "reason": "NewsRequestLimit: quota",
                                               "stop_basket": True, "missing_news_ranges": [["2024-02-01", "2025-01-01"]]}})
            _, (_, _, states) = self.run_basket(root, provider, ("AAPL", "MSFT", "NVDA", "JPM", "XOM"))
            self.assertEqual(provider.calls, ["AAPL", "MSFT"])
            self.assertEqual(len(states), 5)
            self.assertEqual(states["XOM"]["missing_news_ranges"], [["2024-01-01", "2025-01-01"]])

    def test_completed_tickers_skipped_on_resume_and_reports_unchanged(self):
        with writable_test_directory() as root:
            first, _ = self.run_basket(root, FakeProvider({"MSFT": ValueError("interruption")}))
            path = first.directory / "AAPL/comparison.parquet"
            before = path.read_bytes()
            provider = FakeProvider()
            _, (results, _, states) = self.run_basket(root, provider)
            self.assertEqual(provider.calls, ["MSFT"])
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(len(results), 10)
            self.assertTrue(all(state["status"] == "completed" for state in states.values()))

    def test_transaction_costs_identical_across_tickers(self):
        with writable_test_directory() as root:
            runner, _ = self.run_basket(root)
            for ticker in ("AAPL", "MSFT"):
                for name in ("combined", "combined_volatility_sized", "sentiment_only"):
                    frame = pd.read_parquet(runner.directory / ticker / f"{name}.parquet")
                    np.testing.assert_allclose(frame["transaction_cost"], frame["turnover"] * 0.001)

    def test_invalid_dates_or_split_order_never_enter_aggregate(self):
        for change in ("dates", "splits"):
            with self.subTest(change=change), writable_test_directory() as root:
                data = sizing_data()
                if change == "dates":
                    data = data.iloc[::-1]
                else:
                    data.loc[6, "dataset_split"] = "train"
                provider = FakeProvider({"MSFT": {"status": "ready", "model": data, "provenance": {}}})
                _, (results, summary, states) = self.run_basket(root, provider)
                self.assertEqual(set(results["ticker"]), {"AAPL"})
                self.assertTrue(summary["tickers_completed"].eq(1).all())
                self.assertEqual(states["MSFT"]["status"], "failed")

    def test_global_settings_not_changed_by_test_returns(self):
        with writable_test_directory() as root:
            chosen = settings()
            snapshot = json.dumps(chosen, sort_keys=True)
            runner = MultiTickerRobustness(root, chosen)
            runner.run(FakeProvider())
            self.assertEqual(json.dumps(chosen, sort_keys=True), snapshot)
            self.assertEqual(runner.settings, chosen)

    def test_aggregate_means_medians_and_benchmark_counts(self):
        results = pd.DataFrame({
            "ticker": ["A", "A", "B", "B", "C", "C"],
            "strategy": ["Buy & Hold", "Combined"] * 3,
            "total_return": [0.1, 0.2, 0.2, -0.1, -0.1, 0.0],
            "sharpe_ratio": [1, 2, 2, -1, -1, 0], "max_drawdown": [-0.1] * 6,
        })
        combined = summarize_results(results, 5).set_index("strategy").loc["Combined"]
        self.assertAlmostEqual(combined["mean_net_return"], 0.1 / 3)
        self.assertEqual(combined["median_net_return"], 0)
        self.assertEqual(combined["positive_tickers"], 1)
        self.assertEqual(combined["beating_buy_hold"], 2)
        self.assertEqual(combined["classification"], "highly ticker-dependent")

    def test_fewer_than_three_tickers_never_claim_generalization(self):
        with writable_test_directory() as root:
            _, (_, summary, _) = self.run_basket(root)
            self.assertTrue(summary["classification"].eq("insufficient coverage").all())

    def test_empty_basket_results_saved_with_missing_assets(self):
        with writable_test_directory() as root:
            provider = FakeProvider({ticker: ValueError("missing data") for ticker in ("AAPL", "MSFT")})
            runner, (results, summary, states) = self.run_basket(root, provider)
            self.assertTrue(results.empty and summary.empty)
            document = json.loads((runner.directory / "metadata.json").read_text())
            self.assertEqual(document["incomplete"], ["AAPL", "MSFT"])
            self.assertEqual(set(states), {"AAPL", "MSFT"})

    def test_settings_change_creates_separate_report_identity(self):
        with writable_test_directory() as root:
            first = MultiTickerRobustness(root, settings())
            changed = settings()
            changed["evaluation"]["commission_bps"] = 10
            self.assertNotEqual(first.directory, MultiTickerRobustness(root, changed).directory)

    def test_unsafe_or_duplicate_tickers_rejected(self):
        for tickers in ([], ["AAPL", "aapl"], ["../AAPL"]):
            with self.subTest(tickers=tickers), self.assertRaises(ValueError):
                MultiTickerRobustness(".", settings(), tickers)

    def test_only_locked_validation_selection_accepted(self):
        with writable_test_directory() as root:
            path = root / "selection.json"
            path.write_text(json.dumps({"selection_split": "test"}))
            with self.assertRaisesRegex(ValueError, "validation-only"):
                load_global_settings(root, path)

    def test_ticker_specific_news_cache_not_reused_for_another(self):
        with writable_test_directory() as root:
            client = AlphaVantageNewsClient(api_key="test-key", cache_dir=root / "raw")
            start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
            client._save_cache(client.cache_path("AAPL", start, end), "AAPL", start, end, {"feed": []})
            provider = RobustnessDataProvider(root, settings(), start, end, pipeline=NewsSentimentPipeline(client))
            self.assertEqual(provider.inspect("AAPL")["missing_news_ranges"], [])
            self.assertEqual(provider.inspect("MSFT")["missing_news_ranges"], [["2024-01-01", "2025-01-01"]])

    def test_actual_news_quota_checkpoint_stops_cleanly(self):
        with writable_test_directory() as root:
            session = RangeSession(fail_call=1, quota=True)
            client = AlphaVantageNewsClient(api_key="test-key", cache_dir=root / "news", session=session)
            pipeline = NewsSentimentPipeline(client, FakeSentimentModel(), processed_dir=root / "sentiment")
            prices = pd.DataFrame({"Close": [100, 101]}, index=pd.to_datetime(["2024-01-05", "2024-01-08"]))
            provider = RobustnessDataProvider(root, settings(), "2024-01-05", "2024-01-09", pipeline=pipeline,
                                              price_loader=lambda *args, **kwargs: prices)
            result = provider.prepare("MSFT")
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["stop_basket"])
            self.assertEqual(provider.used_requests, 1)
            second = provider.prepare("NVDA")
            self.assertTrue(second["stop_basket"])
            self.assertEqual(len(session.calls), 1)

    def test_shared_per_run_budget_passed_as_remaining_not_reset(self):
        with writable_test_directory() as root:
            prices = pd.DataFrame({"Close": [100, 101]}, index=pd.to_datetime(["2024-01-05", "2024-01-08"]))
            provider = RobustnessDataProvider(root, settings(), "2024-01-05", "2024-01-09", max_requests=5,
                                              price_loader=lambda *args, **kwargs: prices)
            provider.used_requests = 4
            with patch("src.robustness_data.HistoricalSentimentBackfill") as factory:
                factory.return_value.ledger_path = root / "ledger.json"
                factory.return_value.run.side_effect = ValueError("stop before network")
                with self.assertRaises(ValueError):
                    provider.prepare("MSFT")
                self.assertEqual(factory.call_args.kwargs["max_requests"], 1)

    def test_price_cache_independent_and_reused_without_download(self):
        with writable_test_directory() as root:
            frames = [pd.DataFrame({"Close": [value]}, index=pd.to_datetime(["2024-01-02"])) for value in (100, 200)]
            with patch("pipeline.yf.download", side_effect=frames) as download:
                apple = get_price_data("AAPL", "2024-01-01", "2025-01-01", root)
                microsoft = get_price_data("MSFT", "2024-01-01", "2025-01-01", root)
                reused = get_price_data("AAPL", "2024-01-01", "2025-01-01", root, ensure_range=True)
            self.assertEqual(download.call_count, 2)
            pd.testing.assert_frame_equal(apple, reused)
            self.assertNotEqual(apple["Close"].iloc[0], microsoft["Close"].iloc[0])

    def test_price_extensions_request_only_missing_ranges_and_preserve_old_rows(self):
        with writable_test_directory() as root:
            first = pd.DataFrame({"Close": [100.]}, index=pd.to_datetime(["2024-01-03"]))
            extension = pd.DataFrame({"Close": [101.]}, index=pd.to_datetime(["2024-01-05"]))
            with patch("pipeline.yf.download", side_effect=[first, extension]) as download:
                get_price_data("MSFT", "2024-01-02", "2024-01-05", root)
                result = get_price_data("MSFT", "2024-01-02", "2024-01-08", root, ensure_range=True)
                get_price_data("MSFT", "2024-01-02", "2024-01-08", root, ensure_range=True)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(download.call_args.kwargs, {"start": "2024-01-05", "end": "2024-01-08"})
            self.assertEqual(result["Close"].tolist(), [100, 101])

    def test_price_cache_ticker_metadata_mismatch_rejected(self):
        with writable_test_directory() as root:
            data = pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2024-01-02"]))
            data.attrs["ticker"] = "AAPL"
            data.to_parquet(root / "MSFT.parquet")
            with self.assertRaisesRegex(ValueError, "ticker metadata"):
                get_price_data("MSFT", "2024-01-01", "2025-01-01", root)

    def test_changed_completed_report_not_overwritten_or_reevaluated(self):
        with writable_test_directory() as root:
            runner, _ = self.run_basket(root)
            path = runner.directory / "AAPL/comparison.parquet"
            data = pd.read_parquet(path)
            data["total_return"] = 0
            data.to_parquet(path)
            changed = path.read_bytes()
            for _ in range(2):
                provider = FakeProvider()
                _, (results, _, states) = self.run_basket(root, provider)
                self.assertEqual(provider.calls, [])
                self.assertEqual(states["AAPL"]["status"], "invalidated")
                self.assertEqual(set(results["ticker"]), {"MSFT"})
                self.assertEqual(path.read_bytes(), changed)

    def test_unproven_legacy_price_range_fails_without_redownload(self):
        with writable_test_directory() as root:
            pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2024-01-02"])).to_parquet(root / "MSFT.parquet")
            with patch("pipeline.yf.download") as download:
                with self.assertRaisesRegex(ValueError, "provenance"):
                    get_price_data("MSFT", "2024-01-01", "2025-01-01", root, ensure_range=True)
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
