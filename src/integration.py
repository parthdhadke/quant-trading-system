import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COMBINED_DECISION_THRESHOLD,
    REGIME_WEIGHT,
    SENTIMENT_BUY_THRESHOLD,
    SENTIMENT_SELL_THRESHOLD,
    SENTIMENT_WEIGHT,
)
from .news import DAILY_SENTIMENT_COLUMNS
from .regime_pipeline import CausalHMMRegimePipeline
from .signals import add_strategy_signals, combined_signal


MODEL_CACHE_VERSION = 1
SIGNAL_COLUMNS = ["sentiment_signal", "regime_signal", "combined_signal"]
REGIME_COLUMNS = [
    "date", "return", "volatility", "state", "state_probability",
    "regime", "volatility_regime", "regime_label", "dataset_split",
]


def build_model_dataset(
    price_data,
    sentiment_data,
    regime_data,
    *,
    ticker,
    start_date,
    end_date,
    buy_threshold=SENTIMENT_BUY_THRESHOLD,
    sell_threshold=SENTIMENT_SELL_THRESHOLD,
    sentiment_weight=SENTIMENT_WEIGHT,
    regime_weight=REGIME_WEIGHT,
    decision_threshold=COMBINED_DECISION_THRESHOLD,
):
    settings = _signal_settings(
        buy_threshold, sell_threshold, sentiment_weight,
        regime_weight, decision_threshold,
    )
    prices, sentiment, regimes = _prepare_sources(
        price_data, sentiment_data, regime_data, ticker, start_date, end_date,
    )
    eligible = regimes.dropna(subset=["return", "volatility"])
    aligned = prices.merge(eligible, on="date", how="inner", validate="one_to_one")

    if aligned.empty:
        raise ValueError("No eligible price dates with valid historical HMM features")

    aligned = aligned.merge(
        sentiment, on="date", how="left", validate="one_to_one", indicator=True,
    )
    missing = aligned.loc[aligned["_merge"] != "both", "date"]

    if not missing.empty:
        dates = ", ".join(missing.dt.strftime("%Y-%m-%d").head(5))
        raise ValueError(
            f"Missing daily sentiment for {len(missing)} eligible trading dates: "
            f"{dates}. Missing coverage is not evidence of no news."
        )

    aligned = aligned.drop(columns="_merge").sort_values("date").reset_index(drop=True)
    _validate_aligned(aligned)
    result = add_strategy_signals(aligned, **settings)
    validate_model_dataset(result)
    return result


class ModelDatasetPipeline:
    def __init__(
        self,
        regime_pipeline,
        buy_threshold=SENTIMENT_BUY_THRESHOLD,
        sell_threshold=SENTIMENT_SELL_THRESHOLD,
        sentiment_weight=SENTIMENT_WEIGHT,
        regime_weight=REGIME_WEIGHT,
        decision_threshold=COMBINED_DECISION_THRESHOLD,
        processed_dir="data/processed/model",
    ):
        if not isinstance(regime_pipeline, CausalHMMRegimePipeline):
            raise TypeError("regime_pipeline must be a CausalHMMRegimePipeline")

        self.regime_pipeline = regime_pipeline
        self.signal_settings = _signal_settings(
            buy_threshold, sell_threshold, sentiment_weight,
            regime_weight, decision_threshold,
        )
        self.processed_dir = Path(processed_dir)
        self.loaded_from_cache_ = False
        self.cache_path_ = None

    def get_model_dataset(
        self, ticker, start_date, end_date,
        price_data, sentiment_data, regime_data, refresh=False,
    ):
        path = self.cache_path(
            ticker, start_date, end_date, price_data, sentiment_data, regime_data,
        )
        self.cache_path_ = path

        if path.exists() and not refresh:
            result = pd.read_parquet(path)
            validate_model_dataset(result)
            self.loaded_from_cache_ = True
            print(f"[cache] Loading {ticker} model dataset from {path}")
            return result

        result = build_model_dataset(
            price_data, sentiment_data, regime_data,
            ticker=ticker, start_date=start_date, end_date=end_date,
            **self.signal_settings,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".parquet.tmp")
        result.to_parquet(temporary_path, index=False)
        temporary_path.replace(path)
        self.loaded_from_cache_ = False
        print(f"[cache] Saved model dataset to {path}")
        return result

    def cache_path(
        self, ticker, start_date, end_date,
        price_data, sentiment_data, regime_data,
    ):
        sources = _prepare_sources(
            price_data, sentiment_data, regime_data, ticker, start_date, end_date,
        )
        probability_columns = _probability_columns(sources[2])

        if len(probability_columns) != self.regime_pipeline.n_states:
            raise ValueError("Supplied regime state count does not match HMM configuration")

        regime_key = self.regime_pipeline.cache_path(ticker, start_date, end_date).stem
        identity = {
            "version": MODEL_CACHE_VERSION,
            "ticker": str(ticker).strip().upper(),
            "signals": self.signal_settings,
            "regime_key": regime_key,
            "hmm": {
                name: getattr(self.regime_pipeline, name)
                for name in (
                    "n_states", "volatility_window", "random_state", "n_iter",
                    "train_ratio", "validation_ratio",
                )
            },
            "sources": [_frame_digest(source) for source in sources],
            "regime_history": _frame_digest(_regime_source(regime_data)),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()[:20]
        return self.processed_dir / f"{regime_key}_model{digest}.parquet"


def validate_model_dataset(data):
    _validate_aligned(data)
    _require_columns(data, SIGNAL_COLUMNS, "Model dataset")

    for column in SIGNAL_COLUMNS:
        if not data[column].isin([-1, 0, 1]).all():
            raise ValueError(f"{column} must contain only -1, 0, 1")


def _signal_settings(buy, sell, sentiment_weight, regime_weight, decision_threshold):
    settings = {
        "buy_threshold": float(buy),
        "sell_threshold": float(sell),
        "sentiment_weight": float(sentiment_weight),
        "regime_weight": float(regime_weight),
        "decision_threshold": float(decision_threshold),
    }

    if not np.isfinite(list(settings.values())).all():
        raise ValueError("Signal parameters must be finite")

    if not -1 <= sell < 0 < buy <= 1:
        raise ValueError("Thresholds must satisfy -1 <= sell < 0 < buy <= 1")

    combined_signal(0.0, "neutral", **settings)
    return settings


def _dated_frame(data, name):
    if not data.columns.is_unique:
        raise ValueError(f"{name} has duplicate column names")

    result = data.copy()

    if "date" not in result.columns:
        if not isinstance(result.index, pd.DatetimeIndex):
            raise ValueError(f"{name} requires a date column or DatetimeIndex")
        result = result.rename_axis("date").reset_index()

    dates = pd.DatetimeIndex(pd.to_datetime(result["date"], errors="raise"))

    if dates.tz is not None:
        dates = dates.tz_localize(None)

    if dates.isna().any() or not dates.equals(dates.normalize()):
        raise ValueError(f"{name} requires non-missing daily date labels, not intraday timestamps")

    if dates.duplicated().any():
        raise ValueError(f"{name} has duplicate trading dates")

    result["date"] = dates
    return result.sort_values("date").reset_index(drop=True)


def _price_source(price_data, ticker):
    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise ValueError("Price data requires a trading-date DatetimeIndex")

    if isinstance(price_data.columns, pd.MultiIndex):
        columns = [
            column for column in price_data.columns
            if "Close" in column and str(ticker).strip().upper() in column
        ]
        if len(columns) != 1:
            raise ValueError("Price data must contain exactly one Close column for the ticker")
        close = price_data[columns[0]]
    else:
        _require_columns(price_data, ["Close"], "Price data")
        close = price_data["Close"]

    return _dated_frame(close.rename("close").to_frame(), "Price data")


def _regime_source(data):
    result = _dated_frame(data, "Regime data")
    _require_columns(result, REGIME_COLUMNS, "Regime data")
    return result[REGIME_COLUMNS + _probability_columns(result)]


def _prepare_sources(prices, sentiment, regimes, ticker, start_date, end_date):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)

    if (
        pd.isna(start) or pd.isna(end)
        or start.tzinfo is not None or end.tzinfo is not None
        or start != start.normalize() or end != end.normalize() or start >= end
    ):
        raise ValueError("Use timezone-naive daily boundaries with start_date < end_date")

    price_rows = _price_source(prices, ticker)
    sentiment_rows = _dated_frame(sentiment, "Sentiment data")
    _require_columns(sentiment_rows, DAILY_SENTIMENT_COLUMNS, "Sentiment data")
    sentiment_rows = sentiment_rows[DAILY_SENTIMENT_COLUMNS]
    regime_rows = _regime_source(regimes)
    return tuple(
        frame.loc[(frame["date"] >= start) & (frame["date"] < end)].reset_index(drop=True)
        for frame in (price_rows, sentiment_rows, regime_rows)
    )


def _probability_columns(data):
    columns = sorted(
        (column for column in data.columns if re.fullmatch(r"state_\d+_probability", str(column))),
        key=lambda column: int(column.split("_")[1]),
    )
    expected = [f"state_{state}_probability" for state in range(len(columns))]

    if len(columns) < 2 or columns != expected:
        raise ValueError("Regime data requires contiguous state-probability columns")

    return columns


def _require_columns(data, required, name):
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _validate_aligned(data):
    _require_columns(data, REGIME_COLUMNS + DAILY_SENTIMENT_COLUMNS + ["close"], "Model dataset")
    probability_columns = _probability_columns(data)

    if data.empty or data.isna().any().any():
        raise ValueError("Model dataset is empty or contains missing values")

    dates = pd.DatetimeIndex(data["date"])
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Model dates must be sorted and unique")

    numeric_columns = [
        "close", "return", "volatility", "state", "state_probability",
        "sentiment_score", "positive", "neutral", "negative", "headline_count",
        *probability_columns,
    ]
    values = data[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Required model fields must be finite")

    if (data["close"] <= 0).any() or (data["volatility"] < 0).any():
        raise ValueError("Close must be positive and volatility non-negative")

    probabilities = data[probability_columns].to_numpy(dtype=float)
    sentiment_probabilities = data[["positive", "neutral", "negative"]].to_numpy(dtype=float)
    for matrix in (probabilities, sentiment_probabilities):
        if ((matrix < 0) | (matrix > 1)).any() or not np.allclose(matrix.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Probabilities must lie in [0, 1] and sum to 1")

    if not np.allclose(data["state_probability"], probabilities.max(axis=1)):
        raise ValueError("State confidence must match the maximum state probability")
    if not np.array_equal(data["state"].to_numpy(), probabilities.argmax(axis=1)):
        raise ValueError("State IDs must match the highest-probability states")
    if not np.allclose(data["sentiment_score"], data["positive"] - data["negative"], atol=1e-6):
        raise ValueError("Sentiment score must equal positive minus negative probability")

    counts = data["headline_count"].to_numpy(dtype=float)
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Headline counts must be non-negative integers")
    no_news = data.loc[data["headline_count"] == 0, ["sentiment_score", "positive", "neutral", "negative"]]
    if not np.allclose(no_news.to_numpy(dtype=float), [0, 0, 1, 0]):
        raise ValueError("Explicit no-news rows must remain neutral")

    split_order = data["dataset_split"].map({"train": 0, "validation": 1, "test": 2})
    if split_order.isna().any() or not split_order.is_monotonic_increasing:
        raise ValueError("Existing HMM split labels must remain chronological")
    if not data["regime"].isin(["bull", "bear", "neutral"]).all():
        raise ValueError("Unexpected semantic regime label")
    if not data["volatility_regime"].isin(["high_volatility", "low_volatility", "unknown"]).all():
        raise ValueError("Unexpected volatility regime label")
    if not data["regime_label"].eq(data["regime"] + "_" + data["volatility_regime"]).all():
        raise ValueError("Combined regime labels are inconsistent")


def _frame_digest(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()
