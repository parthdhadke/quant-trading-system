from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BACKTEST_INITIAL_CAPITAL, COMMISSION_BPS, SLIPPAGE_BPS
from .backtest import run_backtest
from .evaluation import compare_strategies


STRATEGY_COLUMNS = {
    "Buy & Hold": None,
    "Sentiment Only": "sentiment_signal",
    "Regime Only": "regime_signal",
    "Combined": "combined_signal",
}


@dataclass
class AblationResult:
    test_data: pd.DataFrame
    backtests: dict
    comparison: pd.DataFrame
    equity_curves: pd.DataFrame
    position_counts: pd.DataFrame
    settings: dict


def run_ablation(
    model_data, commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
    initial_capital=BACKTEST_INITIAL_CAPITAL,
):
    columns = ["return", "dataset_split", *[column for column in STRATEGY_COLUMNS.values() if column]]
    if not model_data.columns.is_unique or not set(columns).issubset(model_data.columns):
        raise ValueError("Ablation requires return, dataset_split and all three existing signal columns")
    if "date" in model_data.columns:
        dates = pd.DatetimeIndex(pd.to_datetime(model_data["date"], errors="raise"))
    elif isinstance(model_data.index, pd.DatetimeIndex):
        dates = model_data.index
    else:
        raise ValueError("Ablation requires a date column or DatetimeIndex")
    if (dates.tz is not None or dates.isna().any() or dates.has_duplicates
            or not dates.equals(dates.normalize()) or not dates.is_monotonic_increasing):
        raise ValueError("Ablation requires sorted, unique, timezone-naive daily dates")
    labels = model_data["dataset_split"].map({"train": 0, "validation": 1, "test": 2})
    if labels.isna().any() or not labels.is_monotonic_increasing:
        raise ValueError("Original dataset_split labels must remain chronological")
    working = model_data[columns].copy()
    working.index = dates.rename("date")
    test = working.loc[working["dataset_split"] == "test"].copy()
    if test.empty:
        raise ValueError("No test rows available for ablation")
    for column in STRATEGY_COLUMNS.values():
        if column and not test[column].isin([-1, 0, 1]).all():
            raise ValueError(f"{column} must contain only -1, 0, +1")
    test["return"] = pd.to_numeric(test["return"], errors="raise").astype(float)
    if not np.isfinite(test["return"].to_numpy()).all():
        raise ValueError("All test returns must be finite")
    settings = {
        "commission_bps": float(commission_bps), "slippage_bps": float(slippage_bps),
        "initial_capital": float(initial_capital),
    }
    backtests, counts = {}, []
    for name, column in STRATEGY_COLUMNS.items():
        inputs = test[["return", "dataset_split"]].copy()
        inputs["target_position"] = 1.0 if column is None else test[column].astype(float)
        backtest = run_backtest(inputs, "target_position", **settings)
        if not backtest.index.equals(test.index) or not backtest["return"].equals(test["return"]):
            raise ValueError("Every strategy must preserve the exact same test dates and returns")
        if not backtest["dataset_split"].eq("test").all() or backtest["executed_position"].iloc[0] != 0:
            raise ValueError("All strategies must begin flat within the test split")
        if backtest.isna().any().any() or not np.isfinite(backtest.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("Unexpected missing or non-finite backtest output")
        backtests[name] = backtest
        row = {"strategy": name}
        for prefix, position_column in (("target", "target_position"), ("executed", "executed_position")):
            for direction, value in (("long", 1), ("neutral", 0), ("short", -1)):
                row[f"{prefix}_{direction}_days"] = int(backtest[position_column].eq(value).sum())
        counts.append(row)
    comparison = compare_strategies(backtests)
    if not np.isfinite(comparison.drop(columns="strategy").to_numpy(dtype=float)).all():
        raise ValueError("Comparison contains non-finite metrics")
    equity_curves = pd.DataFrame({name: backtest["equity"] for name, backtest in backtests.items()})
    return AblationResult(test, backtests, comparison, equity_curves, pd.DataFrame(counts), settings)
