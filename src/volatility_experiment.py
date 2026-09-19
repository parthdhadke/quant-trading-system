from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    BACKTEST_INITIAL_CAPITAL, COMMISSION_BPS, MAX_POSITION_SIZE,
    MIN_POSITION_SIZE, SLIPPAGE_BPS, TARGET_VOLATILITY,
)
from .ablation import run_ablation
from .backtest import run_backtest
from .evaluation import compare_strategies
from .signals import apply_volatility_sizing


@dataclass
class VolatilityComparisonResult:
    test_data: pd.DataFrame
    backtests: dict
    comparison: pd.DataFrame
    equity_curves: pd.DataFrame
    settings: dict


def _indexed_data(data):
    if not data.columns.is_unique:
        raise ValueError("Sizing experiment requires unique columns")
    if "date" in data.columns:
        result = data.set_index(pd.DatetimeIndex(pd.to_datetime(data["date"], errors="raise")))
        result = result.drop(columns="date")
    elif isinstance(data.index, pd.DatetimeIndex):
        result = data.copy()
    else:
        raise ValueError("Sizing experiment requires a date column or DatetimeIndex")
    dates = result.index
    if (dates.tz is not None or dates.isna().any() or dates.has_duplicates
            or not dates.equals(dates.normalize()) or not dates.is_monotonic_increasing):
        raise ValueError("Sizing experiment requires sorted, unique, timezone-naive daily dates")
    result.index = dates.rename("date")
    return result


def _checked_volatility(data):
    if "volatility" not in data:
        raise ValueError("Existing trailing annualized volatility is required")
    volatility = pd.to_numeric(data["volatility"], errors="raise").astype(float)
    if not np.isfinite(volatility.to_numpy()).all() or volatility.lt(0).any():
        raise ValueError("Cached volatility must be finite and nonnegative; no rows will be dropped")
    return volatility


def _sized_backtest(data, signal_column, sizing_settings, backtest_settings):
    source = data[["return", "dataset_split", "volatility", signal_column]].copy()
    source["volatility"] = _checked_volatility(source)
    sized = apply_volatility_sizing(source, signal_column, "volatility", **sizing_settings)
    return run_backtest(sized, "position", **backtest_settings)


def validation_preflight(
    validation_data, target_volatility=TARGET_VOLATILITY,
    max_position_size=MAX_POSITION_SIZE, min_position_size=MIN_POSITION_SIZE,
    commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
    initial_capital=BACKTEST_INITIAL_CAPITAL,
):
    data = _indexed_data(validation_data)
    if data.empty or "dataset_split" not in data or not data["dataset_split"].eq("validation").all():
        raise ValueError("Preflight accepts validation rows only; no training or test rows")
    return _sized_backtest(
        data, "combined_signal",
        {"target_volatility": target_volatility, "max_position_size": max_position_size,
         "min_position_size": min_position_size},
        {"commission_bps": commission_bps, "slippage_bps": slippage_bps,
         "initial_capital": initial_capital},
    )


def run_volatility_comparison(
    model_data, target_volatility=TARGET_VOLATILITY,
    max_position_size=MAX_POSITION_SIZE, min_position_size=MIN_POSITION_SIZE,
    commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
    initial_capital=BACKTEST_INITIAL_CAPITAL,
):
    indexed = _indexed_data(model_data)
    if "dataset_split" not in indexed:
        raise ValueError("Original dataset_split labels are required")
    test_volatility = _checked_volatility(indexed.loc[indexed["dataset_split"].eq("test")])
    baseline = run_ablation(model_data, commission_bps, slippage_bps, initial_capital)
    test = baseline.test_data.join(test_volatility)
    sizing_settings = {
        "target_volatility": target_volatility, "max_position_size": max_position_size,
        "min_position_size": min_position_size,
    }
    backtests = {"Buy & Hold": baseline.backtests["Buy & Hold"]}
    for prefix, original_name, signal in (
        ("Sentiment", "Sentiment Only", "sentiment_signal"),
        ("Regime", "Regime Only", "regime_signal"),
        ("Combined", "Combined", "combined_signal"),
    ):
        backtests[f"{prefix} Fixed"] = baseline.backtests[original_name]
        backtests[f"{prefix} Vol Sized"] = _sized_backtest(test, signal, sizing_settings, baseline.settings)
    for frame in backtests.values():
        if not frame.index.equals(test.index) or not frame["return"].equals(test["return"]):
            raise ValueError("All fixed and sized strategies must use identical dates and returns")
        expected = frame["target_position"].shift(1).fillna(0)
        if not frame["executed_position"].equals(expected):
            raise ValueError("Execution must have exactly one trading-period delay")
        if frame["executed_position"].abs().gt(1).any():
            raise ValueError("Leverage is disabled")
        if frame.isna().any().any() or not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("Backtest output must be finite without missing values")
    comparison = compare_strategies(backtests)
    comparison["average_absolute_exposure"] = comparison["strategy"].map(
        {name: frame["executed_position"].abs().mean() for name, frame in backtests.items()}
    )
    comparison["maximum_absolute_exposure"] = comparison["strategy"].map(
        {name: frame["executed_position"].abs().max() for name, frame in backtests.items()}
    )
    settings = {
        **baseline.settings, **{key: float(value) for key, value in sizing_settings.items()},
        "target_selection": "prespecified_no_search",
        "directional_source": "unchanged_cached_baseline_signals",
    }
    equity = pd.DataFrame({name: frame["equity"] for name, frame in backtests.items()})
    return VolatilityComparisonResult(test, backtests, comparison, equity, settings)
