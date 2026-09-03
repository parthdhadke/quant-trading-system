import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    END_DATE,
    HMM_N_ITER,
    HMM_N_STATES,
    HMM_RANDOM_STATE,
    HMM_VOLATILITY_WINDOW,
    START_DATE,
    TICKER,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from src import (
    CausalHMMRegimePipeline,
    ModelDatasetPipeline,
    NewsSentimentPipeline,
    validate_model_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Cache-only alignment and target-signal smoke test")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    arguments = parser.parse_args()
    price_path = Path(f"data/raw/prices/{TICKER}.parquet")
    regime_pipeline = CausalHMMRegimePipeline(
        n_states=HMM_N_STATES,
        volatility_window=HMM_VOLATILITY_WINDOW,
        random_state=HMM_RANDOM_STATE,
        n_iter=HMM_N_ITER,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )
    regime_path = regime_pipeline.cache_path(TICKER, START_DATE, END_DATE)
    sentiment_path = NewsSentimentPipeline().processed_path(TICKER, arguments.start, arguments.end)

    for path in (price_path, regime_path, sentiment_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required local cache not found: {path}. No downloads attempted.")

    prices = pd.read_parquet(price_path)
    regimes = regime_pipeline.get_regimes(TICKER, START_DATE, END_DATE, prices)
    sentiment = pd.read_parquet(sentiment_path)
    pipeline = ModelDatasetPipeline(regime_pipeline)
    result = pipeline.get_model_dataset(
        TICKER, arguments.start, arguments.end, prices, sentiment, regimes,
    )
    validate_model_dataset(result)
    print(f"Sentiment source: {sentiment_path.resolve()}")
    print(f"Regime source: {regime_path.resolve()}")
    print(f"Signal settings: {pipeline.signal_settings}")
    print(f"Final rows: {len(result)}")
    print(f"Date range: {result['date'].min().date()} to {result['date'].max().date()}")
    print("Split counts (original HMM labels):")
    print(result["dataset_split"].value_counts().reindex(["train", "validation", "test"], fill_value=0).to_string())

    for column, labels in (
        ("sentiment_signal", ["positive", "neutral", "negative"]),
        ("regime_signal", ["bullish", "neutral", "bearish"]),
        ("combined_signal", ["long", "neutral", "short"]),
    ):
        counts = result[column].value_counts().reindex([1, 0, -1], fill_value=0)
        counts.index = labels
        print(f"{column} counts:")
        print(counts.to_string())

    columns = [
        "date", "return", "volatility", "sentiment_score", "headline_count",
        "regime", "state_probability", "sentiment_signal", "regime_signal",
        "combined_signal", "dataset_split",
    ]
    print(f"Last {min(10, len(result))} rows:")
    print(result[columns].tail(10).to_string(index=False))
    print(f"Duplicate dates: {result['date'].duplicated().any()}")
    print(f"Sorted dates: {result['date'].is_monotonic_increasing}")
    print(f"Actual price-calendar dates: {result['date'].isin(prices.index).all()}")
    print(f"Numeric fields finite: {np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all()}")
    print(f"Unexpected NaNs: {result.isna().any().any()}")
    print("Signal domain validation: PASS (-1, 0, 1 only)")
    print(f"Processed cache: {pipeline.cache_path_.resolve()}")
    print(f"Processed cache reused: {pipeline.loaded_from_cache_}")
    print("No API calls, model fitting, signal shifting, or backtesting performed.")


if __name__ == "__main__":
    main()
