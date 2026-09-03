import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.news import (
    AlphaVantageNewsClient,
    NewsSentimentPipeline,
    aggregate_daily_sentiment,
    align_news_to_trading_days,
    normalize_alpha_vantage_news,
)


@contextmanager
def writable_test_directory():
    temporary_root = (Path(__file__).resolve().parent / ".tmp").resolve()
    temporary_directory = (temporary_root / uuid.uuid4().hex).resolve()

    if temporary_directory.parent != temporary_root:
        raise RuntimeError("Invalid temporary test directory")

    temporary_directory.mkdir(parents=True)

    try:
        yield temporary_directory
    finally:
        shutil.rmtree(temporary_directory)

        try:
            temporary_root.rmdir()
        except OSError:
            pass


def sample_payload():
    return {
        "feed": [
            {
                "title": "Apple reports stronger than expected earnings",
                "time_published": "20240105T200000",
                "url": "https://example.com/positive",
                "summary": "Revenue exceeded expectations.",
                "source": "Example News",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.95",
                        "ticker_sentiment_score": "0.62",
                        "ticker_sentiment_label": "Bullish",
                    }
                ],
            },
            {
                "title": "Apple shares fall after disappointing guidance",
                "time_published": "20240105T220000",
                "url": "https://example.com/negative",
                "summary": "Guidance missed expectations.",
                "source": "Example News",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.91",
                        "ticker_sentiment_score": "-0.58",
                        "ticker_sentiment_label": "Bearish",
                    }
                ],
            },
            {
                "title": "Apple shares fall after disappointing guidance",
                "time_published": "20240105T220000",
                "url": "https://example.com/negative",
                "ticker_sentiment": [{"ticker": "AAPL"}],
            },
            {
                "title": "Weekend Apple product update",
                "time_published": "20240106T150000",
                "url": "https://example.com/weekend",
                "ticker_sentiment": [{"ticker": "AAPL"}],
            },
        ]
    }


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "timeout": timeout,
            }
        )
        return FakeResponse(self.payload)


class SplittingSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params.copy())
        time_from = params["time_from"]
        time_to = params["time_to"]

        if time_from == "20240105T0000" and time_to == "20240108T2359":
            feed = sample_payload()["feed"][:2]
        elif time_from == "20240105T0000":
            feed = sample_payload()["feed"][:1]
        else:
            feed = sample_payload()["feed"][1:2]

        return FakeResponse({"feed": feed})


class FakeSentimentModel:
    def __init__(self):
        self.calls = 0

    def score_texts(self, texts, batch_size=16):
        self.calls += 1
        rows = []

        for text in texts:
            if "stronger" in text:
                positive, neutral, negative = 0.9, 0.08, 0.02
            elif "disappointing" in text:
                positive, neutral, negative = 0.03, 0.07, 0.9
            else:
                positive, neutral, negative = 0.2, 0.7, 0.1

            rows.append(
                {
                    "text": text,
                    "positive": positive,
                    "neutral": neutral,
                    "negative": negative,
                    "sentiment_score": positive - negative,
                    "sentiment_label": max(
                        {
                            "positive": positive,
                            "neutral": neutral,
                            "negative": negative,
                        },
                        key={
                            "positive": positive,
                            "neutral": neutral,
                            "negative": negative,
                        }.get,
                    ),
                }
            )

        return pd.DataFrame(rows)


class NewsTests(unittest.TestCase):
    def setUp(self):
        self.trading_dates = pd.to_datetime(
            ["2024-01-05", "2024-01-08", "2024-01-09"]
        )

    def test_fetch_news_uses_range_specific_raw_cache(self):
        with writable_test_directory() as temporary_directory:
            session = FakeSession(sample_payload())
            client = AlphaVantageNewsClient(
                api_key="test-key",
                cache_dir=temporary_directory,
                session=session,
            )

            first = client.fetch_news("aapl", "2024-01-05", "2024-01-09")
            second = client.fetch_news("AAPL", "2024-01-05", "2024-01-09")

            self.assertEqual(first, second)
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(
                session.calls[0]["params"]["time_to"],
                "20240108T2359",
            )
            self.assertTrue(
                client.cache_path(
                    "AAPL", "2024-01-05", "2024-01-09"
                ).exists()
            )

    def test_normalize_deduplicates_articles(self):
        news = normalize_alpha_vantage_news(sample_payload(), "AAPL")

        self.assertEqual(len(news), 3)
        self.assertEqual(news["article_id"].nunique(), 3)
        self.assertTrue(str(news["published_at"].dtype).endswith("UTC]"))
        self.assertAlmostEqual(news.iloc[0]["relevance_score"], 0.95)

    def test_capped_response_is_split_and_merged(self):
        with writable_test_directory() as temporary_directory:
            session = SplittingSession()
            client = AlphaVantageNewsClient(
                api_key="test-key",
                cache_dir=temporary_directory,
                limit=2,
                session=session,
            )

            first = client.fetch_news("AAPL", "2024-01-05", "2024-01-09")
            second = client.fetch_news("AAPL", "2024-01-05", "2024-01-09")

            self.assertEqual(len(first["feed"]), 2)
            self.assertEqual(first, second)
            self.assertEqual(len(session.calls), 3)

    def test_after_close_and_weekend_news_roll_forward(self):
        news = normalize_alpha_vantage_news(sample_payload(), "AAPL")
        aligned = align_news_to_trading_days(news, self.trading_dates)

        self.assertEqual(
            aligned["trading_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-05", "2024-01-08", "2024-01-08"],
        )

    def test_daily_sentiment_includes_neutral_no_news_days(self):
        news = normalize_alpha_vantage_news(sample_payload(), "AAPL")
        daily = aggregate_daily_sentiment(
            news,
            FakeSentimentModel(),
            self.trading_dates,
        )

        friday = daily.loc[daily["date"] == pd.Timestamp("2024-01-05")].iloc[0]
        monday = daily.loc[daily["date"] == pd.Timestamp("2024-01-08")].iloc[0]
        tuesday = daily.loc[daily["date"] == pd.Timestamp("2024-01-09")].iloc[0]

        self.assertGreater(friday["sentiment_score"], 0)
        self.assertLess(monday["sentiment_score"], 0)
        self.assertEqual(monday["headline_count"], 2)
        self.assertEqual(tuesday["sentiment_score"], 0)
        self.assertEqual(tuesday["neutral"], 1)
        self.assertEqual(tuesday["headline_count"], 0)

    def test_pipeline_reuses_processed_cache(self):
        with writable_test_directory() as root:
            session = FakeSession(sample_payload())
            client = AlphaVantageNewsClient(
                api_key="test-key",
                cache_dir=root / "raw",
                session=session,
            )
            model = FakeSentimentModel()
            pipeline = NewsSentimentPipeline(
                news_client=client,
                sentiment_model=model,
                processed_dir=root / "processed",
            )

            first = pipeline.get_daily_sentiment(
                "AAPL",
                "2024-01-05",
                "2024-01-09",
                self.trading_dates,
            )
            second = pipeline.get_daily_sentiment(
                "AAPL",
                "2024-01-05",
                "2024-01-09",
                self.trading_dates,
            )

            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
