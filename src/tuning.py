import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BACKTEST_INITIAL_CAPITAL, COMMISSION_BPS, HMM_N_ITER, HMM_RANDOM_STATE,
    HMM_VOLATILITY_WINDOW, SLIPPAGE_BPS, TRAIN_RATIO, VALIDATION_RATIO,
)
from .ablation import run_ablation
from .backtest import run_backtest
from .evaluation import chronological_split, performance_metrics
from .regime import CausalHMMRegimeModel
from .signals import add_strategy_signals


BUY_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40)
SELL_THRESHOLDS = (-0.20, -0.25, -0.30, -0.35, -0.40)
WEIGHT_PAIRS = ((0.25, 0.75), (0.50, 0.50), (0.75, 0.25))
DECISION_THRESHOLDS = (0.25, 0.50, 0.75)
RESULT_COLUMNS = [
    "config_id", "hmm_state_count", "buy_threshold", "sell_threshold",
    "sentiment_weight", "regime_weight", "decision_threshold", "eligible",
    "rejection_reason", "total_return", "gross_total_return", "annualized_return",
    "annualized_volatility", "sharpe_ratio", "max_drawdown", "win_rate",
    "number_of_trades", "total_turnover", "total_transaction_cost",
]


def validate_signal_config(config):
    values = np.asarray([
        config["buy_threshold"], config["sell_threshold"], config["sentiment_weight"],
        config["regime_weight"], config["decision_threshold"],
    ], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Signal configuration must be finite")
    buy, sell, sentiment_weight, regime_weight, decision = values
    if not -1 <= sell < 0 < buy <= 1:
        raise ValueError("Thresholds must satisfy -1 <= sell < 0 < buy <= 1")
    if sentiment_weight < 0 or regime_weight < 0 or not np.isclose(sentiment_weight + regime_weight, 1):
        raise ValueError("Signal weights must be non-negative and sum to 1")
    if not 0 < decision <= 1:
        raise ValueError("Decision threshold must be in (0, 1]")
    return {
        "buy_threshold": float(buy), "sell_threshold": float(sell),
        "sentiment_weight": float(sentiment_weight), "regime_weight": float(regime_weight),
        "decision_threshold": float(decision),
    }


def generate_parameter_grid(
    buy_thresholds=BUY_THRESHOLDS, sell_thresholds=SELL_THRESHOLDS,
    weight_pairs=WEIGHT_PAIRS, decision_thresholds=DECISION_THRESHOLDS,
):
    grid = []
    for buy in buy_thresholds:
        for sell in sell_thresholds:
            for sentiment_weight, regime_weight in weight_pairs:
                for decision in decision_thresholds:
                    grid.append(validate_signal_config({
                        "buy_threshold": buy, "sell_threshold": sell,
                        "sentiment_weight": sentiment_weight, "regime_weight": regime_weight,
                        "decision_threshold": decision,
                    }))
    return grid


def validation_regime_cache_path(
    cache_dir, ticker, start_date, end_date, n_states, feature_digest,
    volatility_window=HMM_VOLATILITY_WINDOW, random_state=HMM_RANDOM_STATE,
    n_iter=HMM_N_ITER, train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO,
):
    tag = lambda value: format(float(value), ".12g").replace("-", "m").replace(".", "p")
    name = "_".join([
        str(ticker).upper(), f"{pd.Timestamp(start_date):%Y%m%d}", f"{pd.Timestamp(end_date):%Y%m%d}",
        f"s{int(n_states)}", f"w{int(volatility_window)}", f"rs{int(random_state)}", f"i{int(n_iter)}",
        f"tr{tag(train_ratio)}", f"va{tag(validation_ratio)}", f"src{feature_digest[:16]}", "validation",
    ])
    return Path(cache_dir) / f"{name}.parquet"


def get_validation_regimes(
    price_data, ticker, start_date, end_date, n_states, cache_dir="data/processed/tuning/regimes",
    volatility_window=HMM_VOLATILITY_WINDOW, random_state=HMM_RANDOM_STATE,
    n_iter=HMM_N_ITER, train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO,
):
    model = CausalHMMRegimeModel(n_states, volatility_window, random_state, n_iter)
    features = model.prepare_features(price_data)
    index = pd.DatetimeIndex(pd.to_datetime(features.index))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    features.index = index.normalize()
    features = features.sort_index().loc[
        (features.index >= pd.Timestamp(start_date)) & (features.index < pd.Timestamp(end_date))
    ]
    digest = hashlib.sha256(pd.util.hash_pandas_object(features, index=True).to_numpy().tobytes()).hexdigest()
    path = validation_regime_cache_path(
        cache_dir, ticker, start_date, end_date, n_states, digest,
        volatility_window, random_state, n_iter, train_ratio, validation_ratio,
    )
    if path.is_file():
        cached = pd.read_parquet(path)
        _validate_validation_regimes(cached, n_states)
        return path.resolve(), cached
    train, validation, test = chronological_split(features, train_ratio, validation_ratio)
    model.fit(train)
    _, train_posterior = model.infer(train)
    validation_result, _ = model.infer(validation, initial_probs=train_posterior)
    validation_result["dataset_split"] = "validation"
    validation_result.index.name = "date"
    result = validation_result.reset_index()
    result.attrs["methodology"] = {
        "fit_split": "train", "inference_split": "validation", "test_rows_inferred": 0,
        "train_rows": len(train), "validation_rows": len(validation), "excluded_test_rows": len(test),
        "train_start": str(train.index.min().date()), "train_end": str(train.index.max().date()),
        "validation_start": str(validation.index.min().date()), "validation_end": str(validation.index.max().date()),
        "state_meanings_source": "training_only", "causal_filtering": True,
    }
    result.attrs["state_summary"] = json.loads(model.get_state_summary().reset_index().to_json(orient="records"))
    _validate_validation_regimes(result, n_states)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path.resolve(), result


def search_validation_parameters(
    validation_data, hmm_state_count, grid=None, minimum_trades=2,
    commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
):
    if minimum_trades < 1:
        raise ValueError("minimum_trades must be at least 1")
    required = {"date", "return", "sentiment_score", "regime", "dataset_split"}
    if not required.issubset(validation_data.columns) or not validation_data["dataset_split"].eq("validation").all():
        raise ValueError("Parameter search accepts validation rows only; test/train rows are forbidden")
    dates = pd.DatetimeIndex(pd.to_datetime(validation_data["date"], errors="raise"))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Validation dates must be sorted and unique")
    source = validation_data.set_index("date")
    rows = []
    for order, supplied in enumerate(grid or generate_parameter_grid()):
        config = validate_signal_config(supplied)
        identity = {**config, "hmm_state_count": int(hmm_state_count)}
        config_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        row = {"config_id": config_id, "hmm_state_count": int(hmm_state_count), **config}
        try:
            signaled = add_strategy_signals(source, **config)
            backtest = run_backtest(
                signaled[["return", "combined_signal"]], "combined_signal",
                commission_bps=commission_bps, slippage_bps=slippage_bps,
                initial_capital=BACKTEST_INITIAL_CAPITAL,
            )
            metrics = performance_metrics(backtest)
            finite = np.asarray(list(metrics.values()), dtype=float)
            if not np.isfinite(finite).all():
                raise ValueError("non-finite validation metrics")
            eligible = metrics["number_of_trades"] >= minimum_trades
            row.update(eligible=eligible, rejection_reason="" if eligible else "insufficient_trading_activity", **metrics)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            row.update(eligible=False, rejection_reason=str(exc))
            row.update({column: np.nan for column in RESULT_COLUMNS[9:]})
        row["_grid_order"] = order
        rows.append(row)
    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["eligible", "sharpe_ratio", "total_return", "max_drawdown", "total_turnover", "_grid_order"],
        ascending=[False, False, False, False, True, True], na_position="last",
    ).drop(columns="_grid_order").reset_index(drop=True)
    if not results["eligible"].any():
        raise ValueError("No eligible validation configurations")
    return results[RESULT_COLUMNS]


def select_best_validation_config(results):
    if results.empty or "eligible" not in results or not results["eligible"].any():
        raise ValueError("No eligible validation candidates")
    best = results.loc[results["eligible"]].iloc[0]
    config = {name: float(best[name]) for name in (
        "buy_threshold", "sell_threshold", "sentiment_weight", "regime_weight", "decision_threshold",
    )}
    return {
        **validate_signal_config(config), "hmm_state_count": int(best["hmm_state_count"]),
        "config_id": best["config_id"], "selection_split": "validation",
        "validation_sharpe_ratio": float(best["sharpe_ratio"]),
    }


def robustness_summary(results, selected):
    eligible = results.loc[results["eligible"]].copy()
    parameter_columns = ["buy_threshold", "sell_threshold", "sentiment_weight", "decision_threshold"]
    masks = [eligible["config_id"].eq(selected["config_id"])]
    for parameter in parameter_columns:
        fixed = np.ones(len(eligible), dtype=bool)
        for other in parameter_columns:
            if other != parameter:
                fixed &= np.isclose(eligible[other], selected[other])
        fixed &= eligible["hmm_state_count"].eq(selected["hmm_state_count"])
        values = sorted(eligible.loc[fixed, parameter].unique())
        position = values.index(selected[parameter])
        neighbors = values[max(0, position - 1):position] + values[position + 1:position + 2]
        masks.extend(fixed & np.isclose(eligible[parameter], value) for value in neighbors)
    fixed_config = np.ones(len(eligible), dtype=bool)
    for parameter in parameter_columns:
        fixed_config &= np.isclose(eligible[parameter], selected[parameter])
    masks.append(fixed_config & ~eligible["hmm_state_count"].eq(selected["hmm_state_count"]))
    nearby = eligible.loc[np.logical_or.reduce(masks)].copy()
    nearby["relationship"] = nearby.apply(lambda row: _relationship(row, selected), axis=1)
    nearby = nearby.sort_values(["relationship", "hmm_state_count", "config_id"]).reset_index(drop=True)
    best_sharpe = selected["validation_sharpe_ratio"]
    worst_drop = max(0.0, best_sharpe - nearby["sharpe_ratio"].min())
    sensitivity = "stable" if worst_drop <= 0.5 else "moderately sensitive" if worst_drop <= 1.5 else "highly sensitive"
    return nearby, {"classification": sensitivity, "maximum_neighbor_sharpe_drop": float(worst_drop),
                    "heuristic": "stable <= 0.5; moderately sensitive <= 1.5; otherwise highly sensitive"}


def evaluate_selected_test(model_data, selected, **costs):
    if selected.get("selection_split") != "validation":
        raise ValueError("Final evaluation requires a validation-locked configuration")
    config = validate_signal_config(selected)
    signaled = add_strategy_signals(model_data, **config)
    result = run_ablation(signaled, **costs)
    result.backtests["Combined Tuned"] = result.backtests.pop("Combined")
    result.comparison.loc[result.comparison["strategy"] == "Combined", "strategy"] = "Combined Tuned"
    result.equity_curves = result.equity_curves.rename(columns={"Combined": "Combined Tuned"})
    result.position_counts.loc[result.position_counts["strategy"] == "Combined", "strategy"] = "Combined Tuned"
    return result


def _relationship(row, selected):
    if row["config_id"] == selected["config_id"]:
        return "selected"
    if row["hmm_state_count"] != selected["hmm_state_count"]:
        return "alternate_hmm_state_count"
    changed = [name for name in ("buy_threshold", "sell_threshold", "sentiment_weight", "decision_threshold")
               if not np.isclose(row[name], selected[name])]
    return f"nearby_{changed[0]}" if len(changed) == 1 else "other"


def _validate_validation_regimes(data, n_states):
    probability_columns = [f"state_{state}_probability" for state in range(n_states)]
    required = {"date", "return", "volatility", "state", "state_probability", "regime",
                "volatility_regime", "regime_label", "dataset_split", *probability_columns}
    if not required.issubset(data.columns) or data.empty or not data["dataset_split"].eq("validation").all():
        raise ValueError("Invalid validation-only regime cache")
    probabilities = data[probability_columns].to_numpy(dtype=float)
    numeric = data[["return", "volatility", "state", "state_probability", *probability_columns]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Invalid validation regime values")
