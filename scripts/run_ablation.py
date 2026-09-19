import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from config import (
    END_DATE, HMM_N_ITER, HMM_N_STATES, HMM_RANDOM_STATE, HMM_VOLATILITY_WINDOW,
    START_DATE, TICKER, TRAIN_RATIO, VALIDATION_RATIO,
)
from src.ablation import run_ablation
from src.integration import ModelDatasetPipeline, validate_model_dataset
from src.regime_pipeline import CausalHMMRegimePipeline


def load_cached_model(project_root):
    root = Path(project_root)
    regimes = CausalHMMRegimePipeline(
        n_states=HMM_N_STATES, volatility_window=HMM_VOLATILITY_WINDOW,
        random_state=HMM_RANDOM_STATE, n_iter=HMM_N_ITER,
        train_ratio=TRAIN_RATIO, validation_ratio=VALIDATION_RATIO,
        processed_dir=root / "data/processed/regimes",
    )
    price_path = root / f"data/raw/prices/{TICKER}.parquet"
    sentiment_path = root / "data/processed/sentiment" / f"{TICKER}_{pd.Timestamp(START_DATE):%Y%m%d}_{pd.Timestamp(END_DATE):%Y%m%d}.parquet"
    regime_path = regimes.cache_path(TICKER, START_DATE, END_DATE)
    for path in (price_path, sentiment_path, regime_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source cache missing: {path.resolve()}. No downloading or refitting allowed.")
    sources = [pd.read_parquet(path) for path in (price_path, sentiment_path, regime_path)]
    integration = ModelDatasetPipeline(regimes, processed_dir=root / "data/processed/model")
    path = integration.cache_path(TICKER, START_DATE, END_DATE, *sources)
    if not path.is_file():
        raise FileNotFoundError(f"No aligned cache matching current sources/configuration: {path.resolve()}")
    model = pd.read_parquet(path)
    validate_model_dataset(model)
    return path.resolve(), model


def comparison_display(comparison):
    result = comparison.copy()
    percentage_columns = [
        "total_return", "gross_total_return", "annualized_return", "annualized_volatility",
        "max_drawdown", "win_rate", "total_transaction_cost",
    ]
    for column in percentage_columns:
        result[column] = result[column].map(lambda value: f"{value * 100:.4f}%")
    result["sharpe_ratio"] = result["sharpe_ratio"].map(lambda value: f"{value:.4f}")
    result["total_turnover"] = result["total_turnover"].map(lambda value: f"{value:.1f}")
    return result.rename(columns={
        "strategy": "Strategy", "total_return": "Net Return", "gross_total_return": "Gross Return",
        "annualized_return": "Ann. Return", "annualized_volatility": "Ann. Volatility",
        "sharpe_ratio": "Sharpe", "max_drawdown": "Max Drawdown", "win_rate": "Win Rate",
        "number_of_trades": "Trades", "total_turnover": "Turnover", "total_transaction_cost": "Summed Costs",
    })


def write_outputs(result, model_path, report_root, ticker=TICKER):
    model_path, report_root = Path(model_path), Path(report_root)
    source_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    identity = {"version": 1, "model_sha256": source_hash, "settings": result.settings, "start_flat": True, "liquidate_at_end": False}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    start, end = result.test_data.index.min(), result.test_data.index.max()
    run_name = f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}_baseline_{digest}"
    directory = report_root / "backtests" / run_name
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in result.backtests.items():
        slug = name.lower().replace(" & ", "_and_").replace(" ", "_")
        data.to_parquet(directory / f"{slug}.parquet", index=True)
    result.comparison.to_parquet(directory / "comparison.parquet", index=False)
    result.equity_curves.to_parquet(directory / "equity_curves.parquet", index=True)
    result.position_counts.to_parquet(directory / "position_counts.parquet", index=False)
    (directory / "comparison.txt").write_text(comparison_display(result.comparison).to_string(index=False) + "\n", encoding="utf-8")
    metadata = {
        **identity, "source_model_path": str(model_path.resolve()), "test_start": str(start.date()),
        "test_end": str(end.date()), "test_rows": len(result.test_data),
        "annualization_factor": 252, "risk_free_rate": 0,
        "execution": "Target on test row t is executed on test row t+1; first test row flat for every strategy",
        "return_convention": "Daily close-to-close proxy, not an explicit next-open fill simulation",
        "win_rate_definition": "Positive net-return days divided by days with nonzero executed exposure",
        "trade_definition": "A day with nonzero executed-position turnover; a reversal is one event, two turnover units",
        "cost_units": "Sum of per-period turnover times cost rate; return units, not currency or compounded cost drag",
        "excluded_costs": ["stock borrow fees", "financing", "market impact beyond fixed slippage", "terminal liquidation"],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def plot_equity(result, figure_path, ticker=TICKER, title=None):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quant-trading-system-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    for name in result.equity_curves:
        axis.plot(result.equity_curves.index, result.equity_curves[name], label=name, linewidth=1.6)
    axis.set_title(title or f"{ticker}: test-only baseline equity curves")
    axis.set_xlabel("Test trading date")
    axis.set_ylabel(f"Equity (starting capital {result.settings['initial_capital']:,.0f})")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_path, model = load_cached_model(project_root)
    result = run_ablation(model)
    print(f"Source model cache: {model_path}")
    print(f"Test period: {result.test_data.index.min().date()} to {result.test_data.index.max().date()}")
    print(f"Test rows: {len(result.test_data)}")
    print(f"Settings: {result.settings}")
    print("Every strategy starts flat; first test-day return is unearned. Targets execute one trading row later.")
    print(comparison_display(result.comparison).to_string(index=False))
    print("Target signal and executed position counts:")
    print(result.position_counts.to_string(index=False))
    combined = result.comparison.set_index("strategy").loc["Combined"]
    print(f"Combined gross total return: {combined['gross_total_return']:.8%}")
    print(f"Combined net total return: {combined['total_return']:.8%}")
    print(f"Combined compounded cost drag: {(combined['gross_total_return'] - combined['total_return']) * 100:.8f} percentage points")
    directory = write_outputs(result, model_path, project_root / "reports")
    figure_path = project_root / "reports/figures" / f"{directory.name}_equity.png"
    plot_equity(result, figure_path)
    print(f"Strategy backtests, comparison, counts and metadata: {directory.resolve()}")
    print(f"Equity curve dataset: {(directory / 'equity_curves.parquet').resolve()}")
    print(f"Equity curve plot: {figure_path.resolve()}")
    print("Summed costs are return units, not cash fees or compounded return drag. Trades are turnover events, not completed round trips.")
    print("Sharpe assumes zero risk-free rate. Annualized metrics extrapolate a short test period, not a full year of observations.")
    print("No downloads, FinBERT inference, HMM fitting, signal changes, tuning or volatility sizing performed.")


if __name__ == "__main__":
    main()
