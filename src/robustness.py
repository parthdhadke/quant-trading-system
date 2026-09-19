import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BACKTEST_INITIAL_CAPITAL, COMMISSION_BPS, END_DATE, HMM_N_ITER,
    HMM_RANDOM_STATE, HMM_VOLATILITY_WINDOW, MAX_POSITION_SIZE, MIN_POSITION_SIZE,
    SLIPPAGE_BPS, START_DATE, TARGET_VOLATILITY, TRAIN_RATIO, VALIDATION_RATIO,
)
from .sentiment_coverage import atomic_parquet
from .volatility_experiment import run_volatility_comparison, validation_preflight


DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "JPM", "XOM")
STRATEGIES = {
    "Buy & Hold": "Buy & Hold", "Sentiment Fixed": "Sentiment Only",
    "Regime Fixed": "Regime Only", "Combined Fixed": "Combined",
    "Combined Vol Sized": "Combined Volatility Sized",
}
SELECTION_FILE = "reports/tuning/AAPL_validation_27627953ee6a/best_validation_config.json"


def atomic_json(document, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_tickers(tickers):
    result = [str(value).strip().upper() for value in tickers]
    if not result or len(set(result)) != len(result) or any(not re.fullmatch(r"[A-Z][A-Z0-9.-]*", value) for value in result):
        raise ValueError("Tickers must be nonempty, unique safe symbols")
    return result


def load_global_settings(root, selection_path=SELECTION_FILE):
    path = Path(root) / selection_path
    selected = json.loads(path.read_text(encoding="utf-8"))
    if (selected.get("selection_split") != "validation" or selected.get("validation_only") is not True
            or selected.get("test_metrics_used_for_selection") is not False):
        raise ValueError("A locked validation-only configuration is required")
    signals = {name: selected[name] for name in (
        "buy_threshold", "sell_threshold", "sentiment_weight", "regime_weight", "decision_threshold",
    )}
    return {
        "signals": signals,
        "hmm": {"n_states": selected["hmm_state_count"], "volatility_window": HMM_VOLATILITY_WINDOW,
                "random_state": HMM_RANDOM_STATE, "n_iter": HMM_N_ITER,
                "train_ratio": TRAIN_RATIO, "validation_ratio": VALIDATION_RATIO},
        "evaluation": {"commission_bps": COMMISSION_BPS, "slippage_bps": SLIPPAGE_BPS,
                       "initial_capital": BACKTEST_INITIAL_CAPITAL, "target_volatility": TARGET_VOLATILITY,
                       "max_position_size": MAX_POSITION_SIZE, "min_position_size": MIN_POSITION_SIZE},
        "selection_source": str(path.resolve()), "selection_sha256": file_hash(path),
        "selection_asset": "AAPL", "selection_basis": "Previously locked AAPL validation; shared across all assets",
    }


def summarize_results(results, requested_count):
    columns = ["strategy", "tickers_completed", "tickers_requested", "mean_net_return", "median_net_return",
               "mean_sharpe", "median_sharpe", "mean_max_drawdown", "positive_tickers", "beating_buy_hold",
               "classification"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    if results.isna().any().any() or not np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Aggregate requires finite completed metrics")
    if results.duplicated(["ticker", "strategy"]).any():
        raise ValueError("Duplicate ticker/strategy rows")
    benchmark = results.loc[results["strategy"].eq("Buy & Hold")].set_index("ticker")["total_return"]
    rows = []
    for strategy, group in results.groupby("strategy", sort=False):
        if set(group["ticker"]) != set(benchmark.index):
            raise ValueError("Aggregate requires the same completed ticker set for every strategy")
        n = len(group)
        positive = int(group["total_return"].gt(0).sum())
        beats = int((group["total_return"].to_numpy() > group["ticker"].map(benchmark).to_numpy()).sum())
        if n < 3:
            classification = "insufficient coverage"
        elif positive == 0:
            classification = "weak"
        elif positive == n and (strategy == "Buy & Hold" or beats == n):
            classification = "consistent"
        elif 0 < positive < n and group["total_return"].max() - group["total_return"].min() >= 0.20:
            classification = "highly ticker-dependent"
        else:
            classification = "mixed"
        rows.append({
            "strategy": strategy, "tickers_completed": n, "tickers_requested": requested_count,
            "mean_net_return": group["total_return"].mean(), "median_net_return": group["total_return"].median(),
            "mean_sharpe": group["sharpe_ratio"].mean(), "median_sharpe": group["sharpe_ratio"].median(),
            "mean_max_drawdown": group["max_drawdown"].mean(), "positive_tickers": positive,
            "beating_buy_hold": beats, "classification": classification,
        })
    return pd.DataFrame(rows, columns=columns)


class MultiTickerRobustness:
    def __init__(self, root, settings, tickers=DEFAULT_TICKERS, start=START_DATE, end=END_DATE):
        self.root = Path(root).resolve()
        self.tickers = normalize_tickers(tickers)
        self.settings = json.loads(json.dumps(settings, allow_nan=False))
        self.start, self.end = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("start must precede end")
        self.identity = {"version": 1, "tickers": self.tickers, "start": self.start, "end_exclusive": self.end,
                         "settings": self.settings}
        digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()[:12]
        self.directory = self.root / "reports/robustness" / f"multi_ticker_{digest}"
        self.states = {ticker: {"status": "pending"} for ticker in self.tickers}
        manifest = self.directory / "metadata.json"
        if manifest.exists():
            document = json.loads(manifest.read_text(encoding="utf-8"))
            if document["identity"] != self.identity or set(document["tickers"]) != set(self.tickers):
                raise ValueError("Run metadata identity mismatch")
            self.states = document["tickers"]

    def _load_completed(self, ticker):
        state = self.states[ticker]
        if state.get("status") == "invalidated":
            raise ValueError("Previously completed artifacts or sources changed; preserved without reevaluation")
        if state.get("status") != "completed":
            return None
        for name, expected in state["artifacts"].items():
            path = self.directory / ticker / name
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"Completed artifact changed: {path}; preserved, refusing silent reevaluation")
        for path, expected in state.get("source_hashes", {}).items():
            if not Path(path).is_file() or file_hash(path) != expected:
                raise ValueError(f"Completed source changed: {path}; use a separately identified experiment")
        data = pd.read_parquet(self.directory / ticker / "comparison.parquet")
        if not data["ticker"].eq(ticker).all() or set(data["strategy"]) != set(STRATEGIES.values()):
            raise ValueError("Ticker-specific completed report identity mismatch")
        if data.isna().any().any() or not np.isfinite(data.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("Completed metrics must be finite")
        return data

    def _evaluate(self, ticker, model, provenance):
        validation_preflight(model.loc[model["dataset_split"].eq("validation")], **self.settings["evaluation"])
        result = run_volatility_comparison(model, **self.settings["evaluation"])
        comparison = result.comparison.loc[result.comparison["strategy"].isin(STRATEGIES)].copy()
        comparison["strategy"] = comparison["strategy"].map(STRATEGIES)
        comparison.insert(0, "ticker", ticker)
        comparison["test_start"] = str(result.test_data.index.min().date())
        comparison["test_end"] = str(result.test_data.index.max().date())
        comparison["test_rows"] = len(result.test_data)
        directory = self.directory / ticker
        for name, frame in result.backtests.items():
            if name in STRATEGIES:
                slug = STRATEGIES[name].lower().replace(" & ", "_and_").replace(" ", "_")
                atomic_parquet(frame.reset_index(), directory / f"{slug}.parquet")
        atomic_parquet(comparison, directory / "comparison.parquet")
        selected_equity = result.equity_curves[list(STRATEGIES)].rename(columns=STRATEGIES)
        atomic_parquet(selected_equity.reset_index(), directory / "equity_curves.parquet")
        source_hashes = {str(Path(path).resolve()): file_hash(path) for path in provenance.get("source_paths", [])}
        self.states[ticker] = {
            "status": "completed", "test_start": comparison["test_start"].iloc[0],
            "test_end": comparison["test_end"].iloc[0], "test_rows": len(result.test_data),
            "provenance": provenance, "source_hashes": source_hashes,
            "artifacts": {path.name: file_hash(path) for path in directory.glob("*.parquet")},
        }
        return comparison

    def _save(self, frames, stop_reason):
        results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "strategy"])
        summary = summarize_results(results, len(self.tickers))
        atomic_parquet(results, self.directory / "multi_ticker_results.parquet")
        atomic_parquet(summary, self.directory / "multi_ticker_summary.parquet")
        metadata = {
            "identity": self.identity, "tickers": self.states, "stop_reason": stop_reason,
            "completed": [ticker for ticker in self.tickers if self.states[ticker]["status"] == "completed"],
            "incomplete": [ticker for ticker in self.tickers if self.states[ticker]["status"] != "completed"],
            "interpretation": "Descriptive equal-weight summaries of per-asset metrics, not pooled portfolio returns or significance tests",
            "classification_rules": "Fewer than 3 completed: insufficient coverage; none positive: weak; all positive and all beat benchmark (or benchmark itself): consistent; mixed signs and >=20pp return spread: highly ticker-dependent; otherwise mixed",
            "limitations": ["AAPL was the configuration-selection asset, not independent cross-asset evidence",
                            "Incomplete baskets are provisional and subject to coverage bias",
                            "First test row flat; t+1 close-to-close proxy; no terminal liquidation",
                            "Trades are turnover events; exposure includes flat rows; costs are summed return units",
                            "No portfolio pooling, per-ticker tuning or test-based parameter selection"],
        }
        atomic_json(metadata, self.directory / "metadata.json")
        return results, summary

    def run(self, provider):
        frames, stop_reason = [], None
        for ticker in self.tickers:
            print(f"[basket] {ticker}", flush=True)
            try:
                completed = self._load_completed(ticker)
                if completed is not None:
                    frames.append(completed)
                    print(f"[resume] Skipping completed {ticker}", flush=True)
                    continue
                if stop_reason:
                    self.states[ticker] = {"status": "incomplete", **provider.inspect(ticker), "reason": f"Not attempted after stop: {stop_reason}"}
                else:
                    outcome = provider.prepare(ticker)
                    if outcome["status"] != "ready":
                        self.states[ticker] = outcome
                        if outcome.get("stop_basket"):
                            stop_reason = outcome["reason"]
                    else:
                        frames.append(self._evaluate(ticker, outcome["model"], outcome["provenance"]))
            except KeyboardInterrupt:
                stop_reason = "Interrupted; resume from checkpoints"
                self.states[ticker] = {"status": "incomplete", **provider.inspect(ticker), "reason": stop_reason}
            except Exception as error:
                previous = self.states[ticker]
                if previous.get("status") in ("completed", "invalidated"):
                    self.states[ticker] = {**previous, "status": "invalidated", "reason": provider.safe_error(error)}
                else:
                    self.states[ticker] = {"status": "failed", **provider.inspect(ticker), "reason": provider.safe_error(error)}
            self._save(frames, stop_reason)
        return (*self._save(frames, stop_reason), self.states)
