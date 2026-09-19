import os
import pandas as pd
import yfinance as yf
from datetime import date
from config import (
    TICKER,
    START_DATE,
    END_DATE,
    SENTIMENT_BUY_THRESHOLD,
    SENTIMENT_SELL_THRESHOLD,
)

def get_price_data(ticker: str, start: str, end: str, cache_dir="data/raw/prices",
                   ensure_range=False, known_cache_range=None) -> pd.DataFrame:
    """Returns OHLCV price data for a ticker between start and end dates, caching to disk."""
    cache_path = os.path.join(cache_dir, f"{ticker}.parquet")

    if os.path.exists(cache_path):
        print(f"[cache] Loading {ticker} from {cache_path}")
        cached = pd.read_parquet(cache_path)
        if cached.attrs.get("ticker", ticker) != ticker:
            raise ValueError("Price cache ticker metadata mismatch")
        if not ensure_range:
            return cached
        from src.sentiment_coverage import missing_ranges

        coverage = cached.attrs.get("requested_ranges")
        if coverage is None:
            if known_cache_range is None:
                raise ValueError("Legacy price cache lacks requested-range provenance; preserved without assuming coverage")
            coverage = [list(known_cache_range)]
        gaps = missing_ranges(start, end, coverage)
        if not gaps:
            return cached
        for left, right in gaps:
            fragment_dir = os.path.join(cache_dir, "ranges", f"{left:%Y%m%d}_{right:%Y%m%d}")
            fragment = get_price_data(ticker, str(left.date()), str(right.date()), cache_dir=fragment_dir)
            if not fragment.columns.equals(cached.columns):
                raise ValueError("Price extension schema differs; original cache preserved")
            fragment = fragment.loc[~fragment.index.isin(cached.index)]
            combined = pd.concat([cached, fragment]).sort_index()
            coverage = [*coverage, [str(left.date()), str(right.date())]]
            combined.attrs = {**cached.attrs, "ticker": ticker, "requested_ranges": coverage}
            temporary = cache_path + ".tmp"
            combined.to_parquet(temporary)
            os.replace(temporary, cache_path)
            cached = combined
        return cached

    print(f"[fetch] Downloading {ticker} from yfinance...")
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")

    df.attrs.update({"ticker": ticker, "requested_ranges": [[str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())]]})

    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"[cache] Saved to {cache_path}")

    return df

def get_sentiment_score(ticker: str, day: str) -> float:
    """Returns a sentiment score in [-1, 1] for a ticker on a given day."""
    print(f"[stub] get_sentiment_score({ticker}, {day})")
    return 0.0

def get_regime_label(ticker: str, day: str) -> int:
    """Returns an integer market regime label (e.g. 0=bear, 1=neutral, 2=bull)."""
    print(f"[stub] get_regime_label({ticker}, {day})")
    return 1

def combine_signal(sentiment: float, regime: int) -> str:
    """Combines sentiment and regime into a trade action: BUY, SELL, or HOLD."""
    print(f"[stub] combine_signal(sentiment={sentiment}, regime={regime})")
    return "HOLD"

if __name__ == "__main__":
    price_data = get_price_data(
        TICKER,
        START_DATE,
        END_DATE
    )

    sentiment = get_sentiment_score(TICKER)
    regime = get_regime_label(price_data)

    signal = combine_signal(sentiment, regime)

    print("Ticker:", TICKER)
    print("Sentiment:", sentiment)
    print("Regime:", regime)
    print("Final Signal:", signal)
