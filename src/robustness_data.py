import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import get_price_data
from .integration import ModelDatasetPipeline
from .news import AlphaVantageNewsClient, NewsSentimentPipeline
from .news_backfill import HistoricalSentimentBackfill
from .regime_pipeline import CausalHMMRegimePipeline
from .sentiment_coverage import inspect_raw_coverage, missing_ranges, trading_calendar


def serialized_ranges(ranges):
    return [[str(left.date()), str(right.date())] for left, right in ranges]


class RobustnessDataProvider:
    def __init__(self, root, settings, start, end, max_requests=20, daily_budget=20, batch_size=16,
                 chunk_days=30, pipeline=None, price_loader=None):
        self.root = Path(root).resolve()
        self.settings = settings
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        self.max_requests, self.daily_budget = max_requests, daily_budget
        self.batch_size, self.chunk_days = batch_size, chunk_days
        if not 0 <= max_requests <= 25 or not 1 <= daily_budget <= 25:
            raise ValueError("Request budgets must be 0..25 per run and 1..25 per day")
        self.used_requests = 0
        self.price_loader = price_loader or get_price_data
        self.pipeline = pipeline or NewsSentimentPipeline(
            AlphaVantageNewsClient(cache_dir=self.root / "data/raw/news"),
            processed_dir=self.root / "data/processed/sentiment",
        )

    def safe_error(self, error):
        text = f"{type(error).__name__}: {error}"
        key = self.pipeline.news_client.api_key
        return text.replace(key, "[redacted]") if key else text

    def inspect(self, ticker):
        raw = inspect_raw_coverage(self.pipeline.news_client, ticker, self.start, self.end)
        price_path = self.root / f"data/raw/prices/{ticker}.parquet"
        sentiment_path = self.pipeline.processed_path(ticker, self.start, self.end)
        price_ranges = []
        if price_path.is_file():
            try:
                cached = pd.read_parquet(price_path)
                price_ranges = cached.attrs.get("requested_ranges", [])
                if not price_ranges and ticker == "AAPL":
                    price_ranges = [("2024-01-01", "2025-01-01")]
            except (ValueError, OSError):
                raw.warnings.append("Price cache could not be inspected")
        return {
            "missing_news_ranges": serialized_ranges(missing_ranges(self.start, self.end, raw.ranges)),
            "price_cache": str(price_path), "price_cache_exists": price_path.is_file(),
            "sentiment_cache_exists": sentiment_path.is_file(),
            "missing_price_ranges": serialized_ranges(missing_ranges(self.start, self.end, price_ranges)),
            "warnings": raw.warnings,
        }

    def prepare(self, ticker):
        prices = self.price_loader(ticker, str(self.start.date()), str(self.end.date()),
                                   cache_dir=self.root / "data/raw/prices", ensure_range=True,
                                   known_cache_range=("2024-01-01", "2025-01-01") if ticker == "AAPL" else None)
        if isinstance(prices.columns, pd.MultiIndex):
            close_columns = [column for column in prices if "Close" in column and ticker in column]
            if len(close_columns) != 1:
                raise ValueError("Price cache does not belong to the requested ticker")
            close = prices[close_columns[0]]
        else:
            if prices.attrs.get("ticker", ticker) != ticker:
                raise ValueError("Price cache ticker metadata mismatch")
            close = prices["Close"]
        calendar = trading_calendar(prices.index, self.start, self.end)
        if not np.isfinite(close.loc[calendar].to_numpy(dtype=float)).all() or close.loc[calendar].le(0).any():
            raise ValueError("Price data contains invalid close values")
        backfill = HistoricalSentimentBackfill(
            pipeline=self.pipeline, chunk_days=self.chunk_days,
            max_requests=max(0, self.max_requests - self.used_requests), daily_budget=self.daily_budget,
            batch_size=self.batch_size,
        )
        before = self._request_identities(backfill.ledger_path)
        try:
            sentiment = backfill.run(ticker, self.start, self.end, prices.index)
        finally:
            self.used_requests += len(self._request_identities(backfill.ledger_path) - before)
        if not sentiment.complete:
            reason = sentiment.stop_reason or "Historical sentiment coverage is incomplete"
            return {
                "status": "incomplete", **self.inspect(ticker), "reason": reason,
                "missing_news_ranges": serialized_ranges(sentiment.missing_raw),
                "missing_sentiment_dates": [str(day.date()) for day in sentiment.missing_dates],
                "processed_days": len(sentiment.daily), "expected_days": len(calendar),
                "requests_this_ticker": len(sentiment.requests), "requests_this_run": self.used_requests,
                "stop_basket": "NewsRequestLimit" in reason or any(
                    term in reason.lower() for term in ("connectionerror", "proxyerror", "sslerror", "keyboardinterrupt")
                ),
            }
        regimes = CausalHMMRegimePipeline(**self.settings["hmm"], processed_dir=self.root / "data/processed/regimes")
        regime_data = regimes.get_regimes(ticker, self.start, self.end, prices)
        integration = ModelDatasetPipeline(regimes, **self.settings["signals"], processed_dir=self.root / "data/processed/model")
        model = integration.get_model_dataset(ticker, self.start, self.end, prices, sentiment.daily, regime_data)
        return {"status": "ready", "model": model, "provenance": {
            "ticker": ticker, "source_paths": [str(path.resolve()) for path in [
                self.root / f"data/raw/prices/{ticker}.parquet",
                self.pipeline.processed_path(ticker, self.start, self.end),
                regimes.cache_path(ticker, self.start, self.end), integration.cache_path_, *sentiment.raw_paths,
            ]],
            "actual_price_start": str(calendar.min().date()), "actual_price_end": str(calendar.max().date()),
            "price_rows": len(calendar), "headline_count": int(sentiment.daily["headline_count"].sum()),
            "no_news_days": int(sentiment.daily["headline_count"].eq(0).sum()),
            "requests_this_ticker": len(sentiment.requests), "warnings": sentiment.warnings,
        }}

    @staticmethod
    def _request_identities(path):
        path = Path(path)
        if not path.exists():
            return set()
        document = json.loads(path.read_text(encoding="utf-8"))
        return {(row["at"], row["ticker"], row["time_from"], row["time_to"]) for row in document["requests"]}
