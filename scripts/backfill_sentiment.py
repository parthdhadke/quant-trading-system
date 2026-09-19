import argparse
from pathlib import Path

import pandas as pd

from config import (
    END_DATE, HMM_N_ITER, HMM_N_STATES, HMM_RANDOM_STATE, HMM_VOLATILITY_WINDOW,
    START_DATE, TICKER, TRAIN_RATIO, VALIDATION_RATIO,
)
from src.integration import ModelDatasetPipeline, validate_model_dataset
from src.news_backfill import HistoricalSentimentBackfill
from src.regime_pipeline import CausalHMMRegimePipeline
from src.sentiment_coverage import missing_ranges


def format_ranges(ranges):
    return ", ".join(f"[{start.date()}, {end.date()})" for start, end in ranges) or "none"


def trading_date_spans(dates, calendar):
    positions = calendar.get_indexer(dates)
    spans = []
    for position in positions:
        day = calendar[position]
        if spans and position == spans[-1][2] + 1:
            spans[-1] = (spans[-1][0], day, position)
        else:
            spans.append((day, day, position))
    return ", ".join(f"{first.date()} to {last.date()}" for first, last, _ in spans) or "none"


def print_report(result):
    print(f"Requested UTC publication range (end exclusive): {format_ranges([(result.start, result.end)])}")
    print(f"Expected cached-price trading dates: {len(result.calendar)}")
    print(f"Raw coverage before: {format_ranges(result.raw_before)}")
    print(f"Missing raw ranges before: {format_ranges(missing_ranges(result.start, result.end, result.raw_before))}")
    print(f"Valid processed coverage before: {len(result.processed_before)} trading days; {trading_date_spans(result.processed_before, result.calendar)}")
    print(f"API requests attempted this run: {len(result.requests)}")
    for request in result.requests:
        print(f"  {request['time_from']} through {request['time_to']}: {request['outcome']}")
    print("Raw caches available/reused:")
    for path in result.raw_paths:
        print(f"  {path}")
    print("Processed caches reused:")
    for path in result.reused_paths:
        print(f"  {path}")
    print(f"Raw coverage after: {format_ranges(result.raw_after)}")
    print(f"Remaining missing raw ranges: {format_ranges(result.missing_raw)}")
    print(f"Processed coverage after: {len(result.daily)}/{len(result.calendar)} trading days; {trading_date_spans(pd.DatetimeIndex(result.daily['date']), result.calendar)}")
    print(f"Missing trading dates ({len(result.missing_dates)}): {trading_date_spans(result.missing_dates, result.calendar)}")
    print(f"Unique normalized articles cached in requested range: {result.article_count}")
    print(f"Articles represented in completed daily rows: {int(result.daily['headline_count'].sum())}")
    print(f"Trading days with news: {int((result.daily['headline_count'] > 0).sum())}")
    print(f"Neutral/no-news trading days: {int((result.daily['headline_count'] == 0).sum())}")
    print("Processed sentiment paths:")
    for path in result.processed_paths:
        print(f"  {path}")
    print(f"Complete requested interval: {result.complete}")
    print(f"Stop reason: {result.stop_reason or 'none'}")
    for warning in result.warnings:
        print(f"[warning] {warning}")
    print("First trading day's information window is clipped at the requested UTC start; no pre-start news is assumed.")
    print("Request accounting covers this backfill workflow only, not calls made elsewhere with the API key.")


def rebuild_model(result, ticker, prices):
    if not result.complete:
        print("Integrated model dataset NOT rebuilt: historical sentiment is incomplete.")
        return
    regimes = CausalHMMRegimePipeline(
        n_states=HMM_N_STATES, volatility_window=HMM_VOLATILITY_WINDOW,
        random_state=HMM_RANDOM_STATE, n_iter=HMM_N_ITER,
        train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO,
    )
    path = regimes.cache_path(ticker, START_DATE, END_DATE)
    if not path.is_file():
        print(f"Integrated model dataset NOT rebuilt: required regime cache missing: {path.resolve()}")
        return
    regime_data = regimes.get_regimes(ticker, START_DATE, END_DATE, prices)
    pipeline = ModelDatasetPipeline(regimes)
    data = pipeline.get_model_dataset(ticker, result.start, result.end, prices, result.daily, regime_data)
    validate_model_dataset(data)
    print(f"Aligned model dataset: {len(data)} rows; {data['date'].min().date()} to {data['date'].max().date()}")
    print(data["dataset_split"].value_counts().to_string())
    print(f"Aligned model cache: {pipeline.cache_path_.resolve()}")
    print("Cached HMM reused without fitting, tuning, signal shifting, or backtesting.")


def main():
    parser = argparse.ArgumentParser(description="Resumable cache-first historical FinBERT sentiment backfill")
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--daily-budget", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    if not ticker or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.^_-" for character in ticker):
        parser.error("Invalid ticker")
    price_path = Path(f"data/raw/prices/{ticker}.parquet")
    if not price_path.is_file():
        raise FileNotFoundError(f"Cached prices required: {price_path.resolve()}; no price download attempted")
    prices = pd.read_parquet(price_path)
    print(f"Cached price calendar: {price_path.resolve()}", flush=True)
    backfill = HistoricalSentimentBackfill(
        chunk_days=args.chunk_days, max_requests=args.max_requests, daily_budget=args.daily_budget,
        batch_size=args.batch_size,
    )
    result = backfill.run(ticker, args.start, args.end, prices.index)
    print_report(result)
    resume_budget = args.max_requests or args.daily_budget
    print(f"Resume from {Path.cwd()}: .\\venv\\Scripts\\python.exe -m scripts.backfill_sentiment --ticker {ticker} --start {args.start} --end {args.end} --chunk-days {args.chunk_days} --max-requests {resume_budget} --daily-budget {args.daily_budget} --batch-size {args.batch_size}")
    rebuild_model(result, ticker, prices)
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
