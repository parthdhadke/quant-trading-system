from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .news import DAILY_SENTIMENT_COLUMNS, NewsSentimentPipeline, aggregate_daily_sentiment, align_news_to_trading_days
from .news_budget import BudgetedNewsSession
from .sentiment_coverage import (
    atomic_parquet, chunk_ranges, date_range, information_windows, inspect_raw_coverage,
    missing_ranges, source_signature, trading_calendar, validate_daily_sentiment,
)


@dataclass
class SentimentBackfillResult:
    start: pd.Timestamp
    end: pd.Timestamp
    calendar: pd.DatetimeIndex
    daily: pd.DataFrame
    raw_before: list
    raw_after: list
    processed_before: pd.DatetimeIndex
    missing_raw: list
    missing_dates: pd.DatetimeIndex
    requests: list
    raw_paths: list
    reused_paths: list
    processed_paths: list
    article_count: int
    stop_reason: str | None
    warnings: list

    @property
    def complete(self):
        return not self.missing_raw and self.missing_dates.empty


class HistoricalSentimentBackfill:
    def __init__(
        self, pipeline=None, chunk_days=30, max_requests=20, daily_budget=20,
        min_request_interval=12.5, ledger_path=None, model_id="ProsusAI/finbert", batch_size=16,
    ):
        list(chunk_ranges("2024-01-01", "2024-01-02", chunk_days))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.pipeline = pipeline or NewsSentimentPipeline()
        self.chunk_days = chunk_days
        self.max_requests = max_requests
        self.daily_budget = daily_budget
        self.min_request_interval = min_request_interval
        self.ledger_path = Path(ledger_path) if ledger_path else self.pipeline.news_client.cache_dir / ".backfill" / "request_budget.json"
        self.model_id = model_id
        self.batch_size = batch_size

    def run(self, ticker, start_date, end_date, trading_dates):
        self.ticker = str(ticker).strip().upper()
        if not self.ticker:
            raise ValueError("ticker cannot be empty")
        self.start, self.end = date_range(start_date, end_date)
        self.calendar = trading_calendar(trading_dates, self.start, self.end)
        self.full_calendar = pd.DatetimeIndex(trading_dates)
        if self.full_calendar.tz is not None:
            self.full_calendar = self.full_calendar.tz_localize(None)
        self.windows = information_windows(
            self.calendar, self.start, self.pipeline.market_timezone, self.pipeline.market_close,
        )
        self.final_path = self.pipeline.processed_path(self.ticker, self.start, self.end)
        self.progress_path = self.pipeline.processed_dir / "backfill" / f"{self.final_path.stem}_progress.parquet"
        self.rows = pd.DataFrame(columns=DAILY_SENTIMENT_COLUMNS + ["_source_signature"])
        self.rows["date"] = pd.Series(dtype="datetime64[ns]")
        self.reused_paths, self.warnings = set(), set()
        client = self.pipeline.news_client
        raw = inspect_raw_coverage(client, self.ticker, self.start, self.end)
        raw_before = raw.ranges.copy()
        aligned, signatures = self._sources(raw)
        self._reuse_processed(raw, aligned, signatures)
        processed_before = pd.DatetimeIndex(self.rows["date"])
        original_session = client.session
        budget = BudgetedNewsSession(
            original_session, self.ledger_path, self.max_requests, self.daily_budget, self.min_request_interval,
        )
        client.session = budget
        stop_reason = None
        fetching = False
        try:
            self._process_available(raw)
            for left, right in chunk_ranges(self.start, self.end, self.chunk_days):
                for missing_start, missing_end in missing_ranges(left, right, raw.ranges):
                    fetching = True
                    client.fetch_news(self.ticker, missing_start, missing_end)
                    fetching = False
                    raw = inspect_raw_coverage(client, self.ticker, self.start, self.end)
                    if missing_ranges(missing_start, missing_end, raw.ranges):
                        raise ValueError("Downloaded cache failed completeness validation")
                    self._process_available(raw)
        except (Exception, KeyboardInterrupt) as exc:
            stop_reason = self._safe_error(exc)
            raw = inspect_raw_coverage(client, self.ticker, self.start, self.end)
            if fetching and not isinstance(exc, KeyboardInterrupt):
                try:
                    self._process_available(raw)
                except Exception as processing_error:
                    stop_reason += f"; checkpoint processing stopped: {self._safe_error(processing_error)}"
        finally:
            client.session = original_session
        raw = inspect_raw_coverage(client, self.ticker, self.start, self.end)
        self.warnings.update(raw.warnings)
        daily = self.rows[DAILY_SENTIMENT_COLUMNS].copy().reset_index(drop=True)
        validate_daily_sentiment(daily, self.calendar, require_complete=False)
        missing_dates = self.calendar.difference(pd.DatetimeIndex(daily["date"]))
        raw_missing = missing_ranges(self.start, self.end, raw.ranges)
        processed_paths = [self.progress_path.resolve()] if self.progress_path.exists() else []
        if not raw_missing and missing_dates.empty:
            validate_daily_sentiment(daily, self.calendar)
            daily.attrs["backfill_source_signatures"] = {
                str(row["date"].date()): row["_source_signature"] for row in self.rows.to_dict("records")
            }
            if self.final_path.exists():
                existing = pd.read_parquet(self.final_path)
                validate_daily_sentiment(existing, self.calendar)
                if not np.allclose(existing[DAILY_SENTIMENT_COLUMNS[1:]].to_numpy(dtype=float),
                                   daily[DAILY_SENTIMENT_COLUMNS[1:]].to_numpy(dtype=float), atol=1e-6):
                    raise ValueError(f"Existing final cache disagrees with backfill; preserved: {self.final_path}")
                if existing.attrs.get("backfill_source_signatures") != daily.attrs["backfill_source_signatures"]:
                    atomic_parquet(daily, self.final_path)
            else:
                atomic_parquet(daily, self.final_path)
            processed_paths.append(self.final_path.resolve())
        return SentimentBackfillResult(
            self.start, self.end, self.calendar, daily, raw_before, raw.ranges, processed_before,
            raw_missing, missing_dates, budget.calls, raw.paths, sorted(self.reused_paths),
            processed_paths, len(raw.news), stop_reason, sorted(self.warnings),
        )

    def _sources(self, raw):
        self.warnings.update(raw.warnings)
        aligned = align_news_to_trading_days(
            raw.news, self.calendar, self.pipeline.market_timezone, self.pipeline.market_close,
        )
        groups = aligned.dropna(subset=["trading_date"]).groupby("trading_date")["article_id"].agg(list).to_dict()
        signatures = {
            day: source_signature(
                day, window, groups.get(day, []), self.model_id,
                self.pipeline.market_timezone, self.pipeline.market_close,
            )
            for day, window in self.windows.items() if not missing_ranges(*window, raw.ranges)
        }
        return aligned, signatures

    def _reuse_processed(self, raw, aligned, signatures):
        prefix = self.final_path.stem.rsplit("_", 2)[0]
        paths = sorted(self.pipeline.processed_dir.glob(f"{prefix}_????????_????????.parquet"))
        paths += sorted((self.pipeline.processed_dir / "backfill").glob(f"{prefix}_????????_????????_progress.parquet"))
        counts = aligned.groupby("trading_date").size().to_dict()
        rows, conflicting_dates = {}, set()
        for path in paths:
            try:
                cached = pd.read_parquet(path)
                cached_signatures = cached.attrs.get("backfill_source_signatures")
                progress = path.stem.endswith("_progress")
                stem = path.stem.removesuffix("_progress")
                _, first, last = stem.rsplit("_", 2)
                left, right = date_range(pd.to_datetime(first, format="%Y%m%d"), pd.to_datetime(last, format="%Y%m%d"))
                if right <= self.start or left >= self.end:
                    continue
                expected = self.full_calendar[(self.full_calendar >= left) & (self.full_calendar < right)]
                validate_daily_sentiment(cached, expected, require_complete=not progress)
                for row in cached.to_dict("records"):
                    day = pd.Timestamp(row["date"])
                    if day in conflicting_dates or day not in signatures or row["headline_count"] != counts.get(day, 0):
                        continue
                    if progress or cached_signatures is not None:
                        cached_signature = row.get("_source_signature") if progress else cached_signatures.get(str(day.date()))
                        if cached_signature != signatures[day]:
                            continue
                    elif self.model_id != "ProsusAI/finbert" or not (
                        left <= self.windows[day][0] and self.windows[day][1] <= right
                    ) or self.pipeline.market_timezone != "America/New_York" or self.pipeline.market_close != "16:00":
                        continue
                    values = {column: row[column] for column in DAILY_SENTIMENT_COLUMNS}
                    values["_source_signature"] = signatures[day]
                    if day in rows and not np.allclose(
                        [rows[day][column] for column in DAILY_SENTIMENT_COLUMNS[1:]],
                        [values[column] for column in DAILY_SENTIMENT_COLUMNS[1:]], atol=1e-6,
                    ):
                        rows.pop(day)
                        conflicting_dates.add(day)
                        self.warnings.add(f"Conflicting processed caches for {day.date()}; rescoring required")
                        continue
                    rows[day] = values
                    self.reused_paths.add(path.resolve())
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.warnings.add(f"Processed cache not trusted: {path.resolve()}: {exc}")
        if rows:
            self.rows = pd.DataFrame([rows[day] for day in sorted(rows)])

    def _checkpoint(self):
        self.rows = self.rows.sort_values("date").reset_index(drop=True)
        validate_daily_sentiment(self.rows, self.calendar, require_complete=False)
        atomic_parquet(self.rows, self.progress_path)

    def _process_available(self, raw):
        aligned, signatures = self._sources(raw)
        if not self.rows.empty:
            valid = self.rows.apply(lambda row: signatures.get(row["date"]) == row["_source_signature"], axis=1)
            self.rows = self.rows.loc[valid].copy()
            self._checkpoint()
        completed = set(self.rows["date"])
        for left, right in chunk_ranges(self.start, self.end, self.chunk_days):
            pending = [day for day in self.calendar if left <= day < right and day in signatures and day not in completed]
            if not pending:
                continue
            news = aligned.loc[aligned["trading_date"].isin(pending)]
            model = self.pipeline._get_sentiment_model() if not news.empty else None
            print(f"[sentiment] Processing {len(pending)} trading days, {len(news)} unique headlines: {pending[0].date()} to {pending[-1].date()}", flush=True)
            daily = aggregate_daily_sentiment(
                news, model, self.calendar, self.pipeline.market_timezone, self.pipeline.market_close, self.batch_size,
            )
            daily = daily.loc[daily["date"].isin(pending)].copy()
            validate_daily_sentiment(daily, pd.DatetimeIndex(pending))
            daily["_source_signature"] = daily["date"].map(signatures)
            self.rows = daily if self.rows.empty else pd.concat([self.rows, daily], ignore_index=True)
            completed.update(pending)
            self._checkpoint()
            print(f"[checkpoint] {len(self.rows)}/{len(self.calendar)} trading days saved to {self.progress_path.resolve()}", flush=True)

    def _safe_error(self, error):
        message = str(error) or type(error).__name__
        key = self.pipeline.news_client.api_key
        if key:
            message = message.replace(key, "[redacted]")
        return f"{type(error).__name__}: {message}"
