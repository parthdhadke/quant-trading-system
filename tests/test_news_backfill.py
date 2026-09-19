import json
import unittest

import numpy as np
import pandas as pd

from src.news import AlphaVantageNewsClient, NewsSentimentPipeline, aggregate_daily_sentiment, normalize_alpha_vantage_news
from src.news_backfill import HistoricalSentimentBackfill
from src.news_budget import BudgetedNewsSession, NewsRequestLimit
from src.sentiment_coverage import (
    chunk_ranges, date_range, inspect_raw_coverage, merge_ranges, missing_ranges,
    trading_calendar, validate_daily_sentiment,
)
from tests.test_news import FakeResponse, FakeSentimentModel, sample_payload, writable_test_directory


class RangeSession:
    def __init__(self, feed=None, fail_call=None, quota=False, cap=False):
        self.feed = sample_payload()["feed"] if feed is None else feed
        self.fail_call = fail_call
        self.quota = quota
        self.cap = cap
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params.copy())
        if len(self.calls) == self.fail_call:
            if self.quota:
                return FakeResponse({"Information": "API rate limit is 25 requests per day"})
            raise ConnectionError("deterministic interrupted download")
        start = pd.to_datetime(params["time_from"], format="%Y%m%dT%H%M")
        end = pd.to_datetime(params["time_to"], format="%Y%m%dT%H%M") + pd.Timedelta(minutes=1)
        feed = [item for item in self.feed if start <= pd.to_datetime(item["time_published"], format="%Y%m%dT%H%M%S") < end]
        if self.cap:
            feed = feed[:params["limit"]]
        return FakeResponse({"feed": feed})


class FailingModel(FakeSentimentModel):
    def __init__(self, fail_call=1):
        super().__init__()
        self.fail_call = fail_call

    def score_texts(self, texts, batch_size=16):
        if self.calls + 1 == self.fail_call:
            self.calls += 1
            raise RuntimeError("deterministic inference failure")
        return super().score_texts(texts, batch_size)


class BackfillTests(unittest.TestCase):
    calendar = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])

    def setup_runner(self, root, session=None, model=None, chunk_days=3, max_requests=20, limit=1000):
        session = session if session is not None else RangeSession()
        model = model if model is not None else FakeSentimentModel()
        client = AlphaVantageNewsClient(api_key="test-key", cache_dir=root / "raw", session=session, limit=limit)
        pipeline = NewsSentimentPipeline(client, model, processed_dir=root / "processed")
        runner = HistoricalSentimentBackfill(pipeline, chunk_days=chunk_days, max_requests=max_requests, min_request_interval=0)
        return runner, client, model, session

    def cache(self, client, start, end, feed=None):
        left, right = date_range(start, end)
        payload = {"feed": [] if feed is None else feed}
        client._save_cache(client.cache_path("AAPL", left, right), "AAPL", left, right, payload)

    def run_range(self, runner):
        return runner.run("AAPL", "2024-01-05", "2024-01-11", self.calendar)

    def test_fully_cached_range_performs_zero_api_calls_and_no_rescoring_on_rerun(self):
        with writable_test_directory() as root:
            runner, client, model, session = self.setup_runner(root)
            self.cache(client, "2024-01-05", "2024-01-11", sample_payload()["feed"])
            first = self.run_range(runner)
            calls = model.calls
            second = self.run_range(runner)
            self.assertTrue(first.complete and second.complete)
            self.assertEqual(session.calls, [])
            self.assertEqual(model.calls, calls)
            pd.testing.assert_frame_equal(first.daily, second.daily)

    def test_partially_cached_range_requests_only_missing_intervals(self):
        with writable_test_directory() as root:
            runner, client, _, session = self.setup_runner(root, chunk_days=6)
            self.cache(client, "2024-01-07", "2024-01-09")
            result = self.run_range(runner)
            self.assertTrue(result.complete)
            self.assertEqual([(call["time_from"], call["time_to"]) for call in session.calls], [
                ("20240105T0000", "20240106T2359"), ("20240109T0000", "20240110T2359"),
            ])

    def test_adjacent_cached_chunks_combine_without_gaps(self):
        with writable_test_directory() as root:
            runner, client, _, session = self.setup_runner(root)
            self.cache(client, "2024-01-05", "2024-01-08", sample_payload()["feed"])
            self.cache(client, "2024-01-08", "2024-01-11")
            result = self.run_range(runner)
            self.assertTrue(result.complete)
            self.assertEqual(result.daily["headline_count"].tolist(), [1, 2, 0, 0])
            self.assertEqual(session.calls, [])

    def test_overlapping_cached_chunks_deduplicate_article_identities(self):
        with writable_test_directory() as root:
            runner, client, _, _ = self.setup_runner(root)
            self.cache(client, "2024-01-05", "2024-01-09", sample_payload()["feed"])
            self.cache(client, "2024-01-06", "2024-01-11", sample_payload()["feed"][3:])
            result = self.run_range(runner)
            self.assertEqual(result.article_count, 3)
            self.assertEqual(result.daily["headline_count"].sum(), 3)

    def test_later_failure_preserves_checkpoints_and_resume_fetches_missing_only(self):
        with writable_test_directory() as root:
            runner, client, _, session = self.setup_runner(root, RangeSession(fail_call=2))
            first = self.run_range(runner)
            self.assertFalse(first.complete)
            self.assertTrue(client.cache_path("AAPL", "2024-01-05", "2024-01-08").exists())
            self.assertTrue(runner.progress_path.exists())
            self.assertEqual(first.daily["date"].tolist(), [self.calendar[0]])
            self.assertFalse(runner.final_path.exists())
            session.fail_call = None
            second = self.run_range(runner)
            self.assertTrue(second.complete)
            self.assertEqual([call["time_from"] for call in session.calls], ["20240105T0000", "20240108T0000", "20240108T0000"])

    def test_capped_responses_use_existing_recursive_split_merge(self):
        feed = [sample_payload()["feed"][0], sample_payload()["feed"][3]]
        with writable_test_directory() as root:
            runner, client, _, session = self.setup_runner(root, RangeSession(feed, cap=True), chunk_days=6, limit=2)
            result = self.run_range(runner)
            self.assertTrue(result.complete)
            self.assertEqual(len(session.calls), 5)
            self.assertEqual(result.article_count, 2)
            self.assertTrue(client.cache_path("AAPL", "2024-01-05", "2024-01-11").exists())

    def test_interrupted_split_resumes_missing_children_without_parent_request(self):
        feed = [sample_payload()["feed"][0], {**sample_payload()["feed"][3], "time_published": "20240109T150000"}]
        with writable_test_directory() as root:
            runner, _, _, session = self.setup_runner(root, RangeSession(feed, fail_call=3, cap=True), chunk_days=6, limit=2)
            first = self.run_range(runner)
            self.assertFalse(first.complete)
            session.fail_call = None
            second = self.run_range(runner)
            self.assertTrue(second.complete)
            self.assertEqual(len(session.calls), 4)
            self.assertEqual(session.calls[-1]["time_from"], "20240108T0000")

    def test_weekend_and_after_close_news_map_forward_across_chunk_boundary(self):
        with writable_test_directory() as root:
            runner, _, _, _ = self.setup_runner(root)
            result = self.run_range(runner)
            monday = result.daily.set_index("date").loc["2024-01-08"]
            self.assertEqual(monday["headline_count"], 2)
            self.assertAlmostEqual(monday["sentiment_score"], (-0.87 + 0.1) / 2)

    def test_future_articles_do_not_map_backward_or_create_holiday_rows(self):
        feed = sample_payload()["feed"] + [{**sample_payload()["feed"][0], "time_published": "20240110T220000"}]
        with writable_test_directory() as root:
            runner, _, _, _ = self.setup_runner(root, RangeSession(feed))
            result = self.run_range(runner)
            self.assertEqual(result.article_count, 4)
            self.assertEqual(result.daily["headline_count"].sum(), 3)
            self.assertEqual(pd.DatetimeIndex(result.daily["date"]).tolist(), self.calendar.tolist())

    def test_no_news_days_are_neutral_without_loading_finbert(self):
        with writable_test_directory() as root:
            runner, _, model, _ = self.setup_runner(root, RangeSession([]))
            result = self.run_range(runner)
            self.assertTrue(result.complete)
            self.assertEqual(model.calls, 0)
            self.assertEqual(result.daily["headline_count"].tolist(), [0, 0, 0, 0])
            self.assertEqual(result.daily["neutral"].tolist(), [1, 1, 1, 1])

    def test_zero_budget_keeps_partial_coverage_explicit_without_network(self):
        with writable_test_directory() as root:
            runner, client, _, session = self.setup_runner(root, max_requests=0)
            self.cache(client, "2024-01-05", "2024-01-08", sample_payload()["feed"])
            result = self.run_range(runner)
            self.assertFalse(result.complete)
            self.assertEqual(len(result.daily), 1)
            self.assertEqual(len(result.missing_dates), 3)
            self.assertEqual(session.calls, [])
            self.assertIn("budget", result.stop_reason)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_daily_sentiment(result.daily, self.calendar)

    def test_inference_failure_preserves_raw_data_and_successful_processed_chunks(self):
        with writable_test_directory() as root:
            runner, client, _, session = self.setup_runner(root, model=FailingModel(fail_call=2))
            first = self.run_range(runner)
            self.assertFalse(first.complete)
            self.assertEqual(len(first.daily), 1)
            self.assertTrue(client.cache_path("AAPL", "2024-01-08", "2024-01-11").exists())
            call_count = len(session.calls)
            runner.pipeline.sentiment_model = FakeSentimentModel()
            second = self.run_range(runner)
            self.assertTrue(second.complete)
            self.assertEqual(len(session.calls), call_count)

    def test_legacy_processed_cache_reuses_only_context_valid_days(self):
        with writable_test_directory() as root:
            runner, client, model, _ = self.setup_runner(root)
            self.cache(client, "2024-01-05", "2024-01-11", sample_payload()["feed"])
            old = aggregate_daily_sentiment(normalize_alpha_vantage_news({"feed": []}, "AAPL"), None, self.calendar[1:])
            path = runner.pipeline.processed_path("AAPL", "2024-01-07", "2024-01-11")
            path.parent.mkdir(parents=True)
            old.to_parquet(path, index=False)
            result = self.run_range(runner)
            self.assertEqual(result.processed_before.tolist(), self.calendar[2:].tolist())
            self.assertEqual(result.daily["headline_count"].tolist(), [1, 2, 0, 0])
            self.assertEqual(model.calls, 2)

    def test_corrupt_raw_cache_cannot_prove_coverage(self):
        with writable_test_directory() as root:
            runner, client, _, _ = self.setup_runner(root, max_requests=0)
            self.cache(client, "2024-01-05", "2024-01-11", [{"title": "Broken timestamp", "time_published": "bad"}])
            result = self.run_range(runner)
            self.assertFalse(result.complete)
            self.assertTrue(result.daily.empty)
            self.assertTrue(result.warnings)

    def test_quota_refusal_stops_once_and_persists_cooldown(self):
        with writable_test_directory() as root:
            runner, _, _, session = self.setup_runner(root, RangeSession(fail_call=2, quota=True))
            first = self.run_range(runner)
            self.assertEqual(len(session.calls), 2)
            self.assertEqual(len(first.daily), 1)
            second = self.run_range(runner)
            self.assertEqual(len(session.calls), 2)
            self.assertIn("cooldown", second.stop_reason)
            self.assertNotIn("test-key", runner.ledger_path.read_text(encoding="utf-8"))

    def test_duplicate_input_calendar_rejected_before_any_requests(self):
        with writable_test_directory() as root:
            runner, _, _, session = self.setup_runner(root)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                runner.run("AAPL", "2024-01-05", "2024-01-11", self.calendar.append(self.calendar[-1:]))
            self.assertEqual(session.calls, [])

    def test_completed_cache_fingerprints_prevent_stale_scores_with_same_article_count(self):
        with writable_test_directory() as root:
            runner, client, model, session = self.setup_runner(root)
            original_feed = sample_payload()["feed"][:1]
            self.cache(client, "2024-01-05", "2024-01-11", original_feed)
            first = self.run_range(runner)
            old_calls = model.calls
            changed = [{**original_feed[0], "title": "Apple disappointing guidance"}]
            self.cache(client, "2024-01-05", "2024-01-11", changed)
            with self.assertRaisesRegex(ValueError, "Existing final cache disagrees"):
                self.run_range(runner)
            self.assertGreater(model.calls, old_calls)
            self.assertEqual(session.calls, [])
            pd.testing.assert_frame_equal(pd.read_parquet(runner.final_path), first.daily)

    def test_equivalent_publication_timestamp_formats_deduplicate(self):
        with writable_test_directory() as root:
            runner, client, _, session = self.setup_runner(root)
            article = sample_payload()["feed"][0]
            self.cache(client, "2024-01-05", "2024-01-11", [article, {**article, "time_published": "20240105T2000"}])
            result = self.run_range(runner)
            self.assertTrue(result.complete)
            self.assertEqual(result.article_count, 1)
            self.assertFalse(result.warnings)
            self.assertEqual(session.calls, [])

    def test_single_day_response_cap_is_not_claimed_complete(self):
        with writable_test_directory() as root:
            runner, _, _, session = self.setup_runner(root, RangeSession(cap=True), chunk_days=1, limit=1)
            result = runner.run("AAPL", "2024-01-05", "2024-01-06", self.calendar)
            self.assertFalse(result.complete)
            self.assertEqual(len(session.calls), 1)
            self.assertIn("completeness cannot be guaranteed", result.stop_reason)
            self.assertTrue(result.daily.empty)
            self.assertFalse(runner.final_path.exists())

    def test_market_holiday_news_rolls_to_next_cached_trading_date(self):
        with writable_test_directory() as root:
            feed = [{**sample_payload()["feed"][0], "time_published": "20240115T160000"}]
            runner, _, _, _ = self.setup_runner(root, RangeSession(feed), chunk_days=2)
            calendar = pd.DatetimeIndex(["2024-01-12", "2024-01-16", "2024-01-17"])
            result = runner.run("AAPL", "2024-01-12", "2024-01-18", calendar)
            self.assertTrue(result.complete)
            self.assertEqual(result.daily["headline_count"].tolist(), [0, 1, 0])


class CoverageValidationTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.date_range("2024-01-08", periods=3, freq="B")
        self.daily = aggregate_daily_sentiment(normalize_alpha_vantage_news({"feed": []}, "AAPL"), None, self.calendar)

    def test_ranges_merge_adjacent_and_overlapping_half_open_intervals(self):
        coverage = [("2024-01-01", "2024-01-04"), ("2024-01-04", "2024-01-06"), ("2024-01-03", "2024-01-05")]
        self.assertEqual(merge_ranges(coverage), [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-06"))])
        self.assertEqual(missing_ranges("2024-01-01", "2024-01-08", coverage), [(pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-08"))])

    def test_chunk_boundaries_are_deterministic_and_cover_leap_day(self):
        chunks = list(chunk_ranges("2024-01-01", "2025-01-01", 30))
        self.assertEqual(len(chunks), 13)
        self.assertEqual(sum((right - left).days for left, right in chunks), 366)
        self.assertEqual(merge_ranges(chunks), [date_range("2024-01-01", "2025-01-01")])

    def test_duplicate_sentiment_dates_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_daily_sentiment(pd.concat([self.daily, self.daily.iloc[-1:]]), self.calendar)

    def test_missing_dates_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_daily_sentiment(self.daily.iloc[:-1], self.calendar)

    def test_unexpected_dates_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_daily_sentiment(self.daily, self.calendar[:-1])

    def test_unsorted_dates_rejected(self):
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_daily_sentiment(self.daily.iloc[::-1], self.calendar)

    def test_nonfinite_values_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            for column in ("positive", "neutral", "negative", "sentiment_score", "headline_count"):
                with self.subTest(column=column, value=value):
                    invalid = self.daily.astype({"headline_count": float}).copy()
                    invalid.loc[0, column] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        validate_daily_sentiment(invalid, self.calendar)

    def test_incorrect_neutral_or_probability_values_rejected(self):
        invalid = self.daily.copy()
        invalid.loc[0, ["sentiment_score", "positive", "neutral"]] = [0.5, 0.5, 0.5]
        with self.assertRaisesRegex(ValueError, "neutral"):
            validate_daily_sentiment(invalid, self.calendar)

    def test_intraday_input_boundaries_rejected(self):
        with self.assertRaises(ValueError):
            date_range("2024-01-01 12:00", "2024-01-02")


class RequestBudgetTests(unittest.TestCase):
    def test_rolling_daily_budget_survives_sessions(self):
        with writable_test_directory() as root:
            transport = RangeSession([])
            params = {"tickers": "AAPL", "time_from": "20240105T0000", "time_to": "20240105T2359", "limit": 1000}
            first = BudgetedNewsSession(transport, root / "ledger.json", daily_budget=1, min_interval=0, clock=lambda: 100000)
            first.get("https://example.com", params, 30)
            second = BudgetedNewsSession(transport, root / "ledger.json", daily_budget=1, min_interval=0, clock=lambda: 100001)
            with self.assertRaisesRegex(NewsRequestLimit, "24-hour"):
                second.get("https://example.com", params, 30)
            self.assertEqual(len(transport.calls), 1)
            second.clock = lambda: 200000
            second.get("https://example.com", params, 30)
            self.assertEqual(len(transport.calls), 2)

    def test_failed_requests_count_and_secrets_are_not_recorded(self):
        with writable_test_directory() as root:
            transport = RangeSession(fail_call=1)
            session = BudgetedNewsSession(transport, root / "ledger.json", max_requests=1, min_interval=0)
            params = {"tickers": "AAPL", "time_from": "20240105T0000", "time_to": "20240105T2359", "apikey": "secret"}
            with self.assertRaises(ConnectionError):
                session.get("https://example.com", params, 30)
            with self.assertRaises(NewsRequestLimit):
                session.get("https://example.com", params, 30)
            ledger_text = (root / "ledger.json").read_text(encoding="utf-8")
            self.assertNotIn("secret", ledger_text)
            self.assertEqual(json.loads(ledger_text)["requests"][0]["outcome"], "failed")


if __name__ == "__main__":
    unittest.main()
