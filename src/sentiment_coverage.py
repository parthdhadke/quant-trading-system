import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .news import DAILY_SENTIMENT_COLUMNS, _parse_published_at, normalize_alpha_vantage_news


def date_range(start_date, end_date):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if any(pd.isna(value) or value.tzinfo is not None or value != value.normalize()
           for value in (start, end)) or start >= end:
        raise ValueError("Use timezone-naive daily boundaries with start_date < end_date")
    return start, end


def trading_calendar(trading_dates, start, end):
    dates = pd.DatetimeIndex(trading_dates)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    if dates.isna().any() or not dates.equals(dates.normalize()):
        raise ValueError("Trading calendar requires non-missing daily dates")
    if dates.has_duplicates:
        raise ValueError("Trading calendar has duplicate dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("Trading calendar must be sorted")
    selected = dates[(dates >= start) & (dates < end)]
    if selected.empty:
        raise ValueError("No cached trading dates in requested range")
    return selected


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(date_range(*interval) for interval in ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def missing_ranges(start, end, coverage):
    start, end = date_range(start, end)
    cursor, missing = start, []
    for left, right in merge_ranges(coverage):
        if right <= cursor or left >= end:
            continue
        if cursor < left:
            missing.append((cursor, min(left, end)))
        cursor = max(cursor, min(right, end))
    if cursor < end:
        missing.append((cursor, end))
    return missing


def chunk_ranges(start, end, chunk_days):
    start, end = date_range(start, end)
    if isinstance(chunk_days, bool) or not isinstance(chunk_days, int) or chunk_days < 1:
        raise ValueError("chunk_days must be a positive integer")
    while start < end:
        following = min(start + pd.Timedelta(days=chunk_days), end)
        yield start, following
        start = following


def validate_daily_sentiment(data, expected_dates, require_complete=True):
    if not data.columns.is_unique or not set(DAILY_SENTIMENT_COLUMNS).issubset(data.columns):
        raise ValueError("Missing or duplicate required sentiment columns")
    dates = pd.DatetimeIndex(pd.to_datetime(data["date"], errors="raise"))
    expected = pd.DatetimeIndex(expected_dates)
    if expected.has_duplicates or not expected.is_monotonic_increasing:
        raise ValueError("Expected trading dates must be sorted and unique")
    if dates.tz is not None or dates.isna().any() or not dates.equals(dates.normalize()):
        raise ValueError("Sentiment requires non-missing timezone-naive daily dates")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Sentiment dates must be sorted with no duplicate dates")
    if len(dates.difference(expected)):
        raise ValueError("Sentiment contains unexpected trading dates")
    if require_complete and not dates.equals(expected):
        raise ValueError("Sentiment coverage is incomplete: missing required trading dates")
    values = data[DAILY_SENTIMENT_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Required sentiment values must be finite, without NaN")
    probabilities = data[["positive", "neutral", "negative"]].to_numpy(dtype=float)
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0
    ):
        raise ValueError("Invalid sentiment probabilities")
    if not np.allclose(values[:, 0], values[:, 1] - values[:, 3], atol=1e-6):
        raise ValueError("Sentiment score must equal positive minus negative")
    counts = data["headline_count"].to_numpy(dtype=float)
    if (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("headline_count must be a non-negative integer")
    empty = data.loc[counts == 0, ["sentiment_score", "positive", "neutral", "negative"]]
    if not np.allclose(empty.to_numpy(dtype=float), [0.0, 0.0, 1.0, 0.0]):
        raise ValueError("No-news trading days must be neutral")


@dataclass
class RawNewsCoverage:
    ranges: list
    news: pd.DataFrame
    paths: list
    warnings: list


def inspect_raw_coverage(client, ticker, start, end):
    ranges, frames, paths, warnings = [], [], [], []
    prefix = client.cache_path(ticker, start, end).stem.rsplit("_", 2)[0]
    for path in sorted(client.cache_dir.glob(f"{prefix}_????????_????????.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("ticker") != ticker:
                continue
            left, right = date_range(envelope["start_date"], envelope["end_date_exclusive"])
            if right <= start or left >= end:
                continue
            if path != client.cache_path(ticker, left, right):
                raise ValueError("Filename disagrees with cached query range")
            payload = envelope["payload"]
            client._validate_payload(payload)
            if envelope.get("complete") is False or (
                "complete" not in envelope and len(payload["feed"]) >= 1000
            ):
                raise ValueError("Cache lacks trustworthy uncapped/completed-range metadata")
            news = normalize_alpha_vantage_news(payload, ticker)
            identities = {
                (str(item.get("title", "")).strip(), _parse_published_at(item.get("time_published")),
                 str(item.get("url", "")).strip())
                for item in payload["feed"] if isinstance(item, dict)
            }
            if len(news) != len(identities) or any(not isinstance(item, dict) for item in payload["feed"]):
                raise ValueError("Invalid articles prevent reliable news coverage")
            left, right = max(start, left), min(end, right)
            if not news.empty:
                news = news.loc[(news["published_at"] >= left.tz_localize("UTC")) &
                                (news["published_at"] < right.tz_localize("UTC"))]
                frames.append(news)
            ranges.append((left, right))
            paths.append(path.resolve())
        except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError) as exc:
            warnings.append(f"{path.resolve()}: {exc}")
    news = normalize_alpha_vantage_news({"feed": []}, ticker)
    if frames:
        news = pd.concat(frames, ignore_index=True).drop_duplicates("article_id")
        news = news.sort_values(["published_at", "article_id"]).reset_index(drop=True)
    return RawNewsCoverage(merge_ranges(ranges), news, paths, warnings)


def information_windows(calendar, start, market_timezone, market_close):
    closes = pd.DatetimeIndex([
        pd.Timestamp(f"{day.date()} {market_close}").tz_localize(market_timezone).tz_convert("UTC")
        for day in calendar
    ])
    windows = {}
    for position, day in enumerate(calendar):
        left = start if position == 0 else closes[position - 1].tz_localize(None).normalize()
        right = closes[position].tz_localize(None).normalize() + pd.Timedelta(days=1)
        windows[day] = (max(start, left), right)
    return windows


def source_signature(day, window, article_ids, model_id, timezone, close):
    identity = [str(day), [str(value) for value in window], sorted(article_ids), model_id, timezone, close]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_parquet(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    data.to_parquet(temporary, index=False)
    temporary.replace(path)
