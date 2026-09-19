import hashlib
import json
import os
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


NEWS_COLUMNS = [
    "article_id",
    "ticker",
    "published_at",
    "headline",
    "summary",
    "source",
    "url",
    "relevance_score",
    "provider_sentiment_score",
    "provider_sentiment_label",
]

DAILY_SENTIMENT_COLUMNS = [
    "date",
    "sentiment_score",
    "positive",
    "neutral",
    "negative",
    "headline_count",
]


class AlphaVantageNewsClient:
    def __init__(
        self,
        api_key=None,
        cache_dir="data/raw/news",
        timeout=30,
        limit=1000,
        session=None,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")

        load_dotenv()

        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.limit = limit
        self.session = session or requests.Session()
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_news(
        self,
        ticker,
        start_date,
        end_date,
        refresh=False,
    ):
        ticker = _normalize_ticker(ticker)
        start, end = _normalize_date_range(start_date, end_date)
        return self._fetch_range(ticker, start, end, refresh)

    def _fetch_range(self, ticker, start, end, refresh):
        cache_path = self.cache_path(ticker, start, end)

        if cache_path.exists() and not refresh:
            print(f"[cache] Loading {ticker} news from {cache_path}")
            return self._load_cache(cache_path)

        if not self.api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY is required when no cached news is available"
            )

        payload = self._request_range(ticker, start, end)

        if len(payload["feed"]) >= self.limit:
            duration = end - start

            if duration <= pd.Timedelta(days=1):
                raise RuntimeError(
                    "Alpha Vantage returned the maximum number of articles for "
                    "a single day; completeness cannot be guaranteed"
                )

            midpoint = start + pd.Timedelta(days=max(1, duration.days // 2))
            left = self._fetch_range(ticker, start, midpoint, refresh)
            right = self._fetch_range(ticker, midpoint, end, refresh)
            payload = self._merge_payloads(left, right)

        self._save_cache(cache_path, ticker, start, end, payload)
        print(f"[cache] Saved news to {cache_path}")
        return payload

    def _request_range(self, ticker, start, end):
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": start.strftime("%Y%m%dT%H%M"),
            "time_to": (end - pd.Timedelta(minutes=1)).strftime("%Y%m%dT%H%M"),
            "sort": "EARLIEST",
            "limit": self.limit,
            "apikey": self.api_key,
        }

        print(f"[fetch] Downloading {ticker} news from Alpha Vantage...")
        response = self.session.get(
            self.base_url,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise ValueError("Alpha Vantage returned invalid JSON") from exc

        self._validate_payload(payload)
        return payload

    @staticmethod
    def _merge_payloads(*payloads):
        merged = {}
        feed = []
        seen = set()

        for payload in payloads:
            for key, value in payload.items():
                if key not in {"feed", "items"} and key not in merged:
                    merged[key] = value

            for article in payload["feed"]:
                if not isinstance(article, dict):
                    continue

                identity = (
                    str(article.get("url", "")).strip(),
                    str(article.get("time_published", "")).strip(),
                    str(article.get("title", "")).strip(),
                )

                if identity in seen:
                    continue

                seen.add(identity)
                feed.append(article)

        feed.sort(
            key=lambda article: str(article.get("time_published", ""))
        )
        merged["items"] = str(len(feed))
        merged["feed"] = feed
        return merged

    def cache_path(self, ticker, start_date, end_date):
        ticker = _normalize_ticker(ticker)
        start, end = _normalize_date_range(start_date, end_date)
        safe_ticker = "".join(
            character if character.isalnum() else "_"
            for character in ticker
        )
        filename = (
            f"{safe_ticker}_{start.strftime('%Y%m%d')}"
            f"_{end.strftime('%Y%m%d')}.json"
        )
        return self.cache_dir / filename

    @staticmethod
    def _validate_payload(payload):
        if not isinstance(payload, dict):
            raise ValueError("Alpha Vantage response must be a JSON object")

        for key in ("Error Message", "Note", "Information"):
            if payload.get(key):
                raise RuntimeError(f"Alpha Vantage response: {payload[key]}")

        if "feed" not in payload:
            raise ValueError("Alpha Vantage response does not contain a news feed")

        if not isinstance(payload["feed"], list):
            raise ValueError("Alpha Vantage news feed must be a list")

    @staticmethod
    def _load_cache(cache_path):
        with cache_path.open("r", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)

        payload = cached.get("payload") if isinstance(cached, dict) else None

        if payload is None:
            payload = cached

        AlphaVantageNewsClient._validate_payload(payload)
        return payload

    def _save_cache(self, cache_path, ticker, start, end, payload):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".json.tmp")
        envelope = {
            "ticker": ticker,
            "start_date": start.isoformat(),
            "end_date_exclusive": end.isoformat(),
            "complete": True,
            "response_limit": self.limit,
            "payload": payload,
        }

        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(envelope, cache_file, ensure_ascii=False)

        temporary_path.replace(cache_path)


class NewsSentimentPipeline:
    def __init__(
        self,
        news_client=None,
        sentiment_model=None,
        processed_dir="data/processed/sentiment",
        market_timezone="America/New_York",
        market_close="16:00",
    ):
        self.news_client = news_client or AlphaVantageNewsClient()
        self.sentiment_model = sentiment_model
        self.processed_dir = Path(processed_dir)
        self.market_timezone = market_timezone
        self.market_close = market_close

    def get_daily_sentiment(
        self,
        ticker,
        start_date,
        end_date,
        trading_dates,
        refresh_news=False,
        rescore=False,
        batch_size=16,
    ):
        ticker = _normalize_ticker(ticker)
        start, end = _normalize_date_range(start_date, end_date)
        processed_path = self.processed_path(ticker, start, end)

        if processed_path.exists() and not refresh_news and not rescore:
            print(
                f"[cache] Loading {ticker} daily sentiment from {processed_path}"
            )
            daily = pd.read_parquet(processed_path)
            daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
            return daily[DAILY_SENTIMENT_COLUMNS]

        payload = self.news_client.fetch_news(
            ticker,
            start,
            end,
            refresh=refresh_news,
        )
        news = normalize_alpha_vantage_news(payload, ticker)
        sentiment_model = self._get_sentiment_model()
        daily = aggregate_daily_sentiment(
            news,
            sentiment_model,
            trading_dates,
            market_timezone=self.market_timezone,
            market_close=self.market_close,
            batch_size=batch_size,
        )

        processed_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(processed_path, index=False)
        print(f"[cache] Saved daily sentiment to {processed_path}")
        return daily

    def processed_path(self, ticker, start_date, end_date):
        ticker = _normalize_ticker(ticker)
        start, end = _normalize_date_range(start_date, end_date)
        safe_ticker = "".join(
            character if character.isalnum() else "_"
            for character in ticker
        )
        filename = (
            f"{safe_ticker}_{start.strftime('%Y%m%d')}"
            f"_{end.strftime('%Y%m%d')}.parquet"
        )
        return self.processed_dir / filename

    def _get_sentiment_model(self):
        if self.sentiment_model is None:
            from .sentiment import FinBERTSentiment

            self.sentiment_model = FinBERTSentiment()

        return self.sentiment_model


def normalize_alpha_vantage_news(payload, ticker):
    AlphaVantageNewsClient._validate_payload(payload)
    ticker = _normalize_ticker(ticker)
    rows = []

    for article in payload["feed"]:
        if not isinstance(article, dict):
            continue

        headline = str(article.get("title", "")).strip()
        published_at = _parse_published_at(article.get("time_published"))

        if not headline or pd.isna(published_at):
            continue

        ticker_sentiment = _ticker_sentiment(article, ticker)
        url = str(article.get("url", "")).strip()
        identity = "|".join((ticker, published_at.isoformat(), headline, url))
        article_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        rows.append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "published_at": published_at,
                "headline": headline,
                "summary": str(article.get("summary", "")).strip(),
                "source": str(article.get("source", "")).strip(),
                "url": url,
                "relevance_score": _optional_float(
                    ticker_sentiment.get("relevance_score")
                ),
                "provider_sentiment_score": _optional_float(
                    ticker_sentiment.get("ticker_sentiment_score")
                ),
                "provider_sentiment_label": str(
                    ticker_sentiment.get("ticker_sentiment_label", "")
                ).strip(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    return (
        pd.DataFrame(rows, columns=NEWS_COLUMNS)
        .drop_duplicates(subset="article_id", keep="first")
        .sort_values("published_at")
        .reset_index(drop=True)
    )


def align_news_to_trading_days(
    news,
    trading_dates,
    market_timezone="America/New_York",
    market_close="16:00",
):
    missing = {"published_at", "headline"}.difference(news.columns)

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    calendar = _normalize_trading_dates(trading_dates)
    result = news.copy()

    if result.empty:
        result["trading_date"] = pd.Series(dtype="datetime64[ns]")
        return result

    timezone = ZoneInfo(market_timezone)
    close_time = _parse_market_close(market_close)
    published = pd.to_datetime(result["published_at"], utc=True, errors="coerce")

    if published.isna().any():
        raise ValueError("News contains invalid publication timestamps")

    local_published = published.dt.tz_convert(timezone)
    trading_day_values = calendar.to_numpy(dtype="datetime64[ns]")
    aligned_dates = []

    for timestamp in local_published:
        local_date = pd.Timestamp(timestamp.date())
        known_by_close = timestamp.time().replace(tzinfo=None) <= close_time
        position = calendar.searchsorted(local_date, side="left")

        if (
            not known_by_close
            and position < len(calendar)
            and calendar[position] == local_date
        ):
            position += 1

        if position >= len(calendar):
            aligned_dates.append(pd.NaT)
        else:
            aligned_dates.append(pd.Timestamp(trading_day_values[position]))

    result["trading_date"] = pd.to_datetime(aligned_dates)
    return result


def aggregate_daily_sentiment(
    news,
    sentiment_model,
    trading_dates,
    market_timezone="America/New_York",
    market_close="16:00",
    batch_size=16,
):
    calendar = _normalize_trading_dates(trading_dates)
    aligned = align_news_to_trading_days(
        news,
        calendar,
        market_timezone=market_timezone,
        market_close=market_close,
    ).dropna(subset=["trading_date"])

    if aligned.empty:
        return _empty_daily_sentiment(calendar)

    scored = sentiment_model.score_texts(
        aligned["headline"].tolist(),
        batch_size=batch_size,
    ).reset_index(drop=True)

    if len(scored) != len(aligned):
        raise ValueError("Sentiment model returned an unexpected number of rows")

    aligned = aligned.reset_index(drop=True)

    for column in ("sentiment_score", "positive", "neutral", "negative"):
        aligned[column] = pd.to_numeric(scored[column], errors="raise")

    daily = (
        aligned.groupby("trading_date")
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            positive=("positive", "mean"),
            neutral=("neutral", "mean"),
            negative=("negative", "mean"),
            headline_count=("headline", "count"),
        )
        .reindex(calendar)
    )

    no_headlines = daily["headline_count"].isna()
    daily.loc[no_headlines, "sentiment_score"] = 0.0
    daily.loc[no_headlines, "positive"] = 0.0
    daily.loc[no_headlines, "neutral"] = 1.0
    daily.loc[no_headlines, "negative"] = 0.0
    daily["headline_count"] = daily["headline_count"].fillna(0).astype(int)
    daily.index.name = "date"
    return daily.reset_index()[DAILY_SENTIMENT_COLUMNS]


def _empty_daily_sentiment(calendar):
    return pd.DataFrame(
        {
            "date": calendar,
            "sentiment_score": np.zeros(len(calendar), dtype=float),
            "positive": np.zeros(len(calendar), dtype=float),
            "neutral": np.ones(len(calendar), dtype=float),
            "negative": np.zeros(len(calendar), dtype=float),
            "headline_count": np.zeros(len(calendar), dtype=int),
        },
        columns=DAILY_SENTIMENT_COLUMNS,
    )


def _normalize_ticker(ticker):
    normalized = str(ticker).strip().upper()

    if not normalized:
        raise ValueError("ticker cannot be empty")

    return normalized


def _normalize_date_range(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)

    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)

    start = start.normalize()
    end = end.normalize()

    if end <= start:
        raise ValueError("end_date must be later than start_date")

    return start, end


def _normalize_trading_dates(trading_dates):
    if isinstance(trading_dates, pd.DatetimeIndex):
        calendar = trading_dates
    else:
        calendar = pd.DatetimeIndex(trading_dates)

    if calendar.tz is not None:
        calendar = calendar.tz_convert("UTC").tz_localize(None)

    calendar = calendar.normalize().drop_duplicates().sort_values()

    if calendar.empty:
        raise ValueError("trading_dates cannot be empty")

    return calendar


def _parse_published_at(value):
    if value is None:
        return pd.NaT

    text = str(value).strip()

    for date_format in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return pd.to_datetime(text, format=date_format, utc=True)
        except ValueError:
            continue

    return pd.NaT


def _parse_market_close(value):
    if isinstance(value, time):
        return value.replace(tzinfo=None)

    try:
        parsed = pd.Timestamp(str(value)).time()
    except ValueError as exc:
        raise ValueError("market_close must be a valid time") from exc

    return parsed.replace(tzinfo=None)


def _ticker_sentiment(article, ticker):
    values = article.get("ticker_sentiment", [])

    if not isinstance(values, list):
        return {}

    for value in values:
        if not isinstance(value, dict):
            continue

        if str(value.get("ticker", "")).strip().upper() == ticker:
            return value

    return {}


def _optional_float(value):
    if value is None or str(value).strip() == "":
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
