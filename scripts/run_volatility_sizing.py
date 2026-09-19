import hashlib
import json
from pathlib import Path

from config import (
    COMBINED_DECISION_THRESHOLD, HMM_N_STATES, HMM_VOLATILITY_WINDOW,
    REGIME_WEIGHT, SENTIMENT_BUY_THRESHOLD, SENTIMENT_SELL_THRESHOLD,
    SENTIMENT_WEIGHT, TARGET_VOLATILITY, TICKER,
)
from scripts.run_ablation import comparison_display, load_cached_model, plot_equity
from src.evaluation import performance_metrics
from src.volatility_experiment import run_volatility_comparison, validation_preflight


DETAIL_COLUMNS = [
    "volatility", "combined_signal", "position_size", "target_position",
    "executed_position", "return", "gross_return", "transaction_cost", "net_return",
]


def sizing_comparison_display(comparison):
    result = comparison_display(comparison)
    result["Turnover"] = comparison["total_turnover"].map(lambda value: f"{value:.6f}")
    for column in ("average_absolute_exposure", "maximum_absolute_exposure"):
        result[column] = comparison[column].map(lambda value: f"{value:.4%}")
    return result.rename(columns={
        "average_absolute_exposure": "Average Exposure", "maximum_absolute_exposure": "Maximum Exposure",
    })


def write_outputs(result, validation, model_path, report_root, ticker=TICKER):
    model_path, report_root = Path(model_path), Path(report_root)
    identity = {
        "version": 1, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "settings": result.settings, "start_flat": True, "liquidate_at_end": False,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    start, end = result.test_data.index.min(), result.test_data.index.max()
    directory = report_root / "backtests" / f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}_volatility_{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in result.backtests.items():
        slug = name.lower().replace(" & ", "_and_").replace(" ", "_")
        data.to_parquet(directory / f"{slug}.parquet", index=True)
    result.comparison.to_parquet(directory / "comparison.parquet", index=False)
    result.equity_curves.to_parquet(directory / "equity_curves.parquet", index=True)
    validation.to_parquet(directory / "validation_preflight.parquet", index=True)
    detail = result.backtests["Combined Vol Sized"][DETAIL_COLUMNS].tail(10)
    detail.to_parquet(directory / "combined_last10.parquet", index=True)
    (directory / "combined_last10.txt").write_text(detail.to_string(float_format=lambda value: f"{value:.8f}") + "\n", encoding="utf-8")
    (directory / "comparison.txt").write_text(sizing_comparison_display(result.comparison).to_string(index=False) + "\n", encoding="utf-8")
    metadata = {
        **identity, "source_model_path": str(model_path.resolve()),
        "test_start": str(start.date()), "test_end": str(end.date()), "test_rows": len(result.test_data),
        "validation_start": str(validation.index.min().date()),
        "validation_end": str(validation.index.max().date()), "validation_rows": len(validation),
        "validation_metrics": performance_metrics(validation),
        "validation_use": "Preflight of a prespecified target; no target search, ranking or test-based selection",
        "baseline_configuration": {
            "hmm_states": HMM_N_STATES, "sentiment_buy_threshold": SENTIMENT_BUY_THRESHOLD,
            "sentiment_sell_threshold": SENTIMENT_SELL_THRESHOLD,
            "sentiment_weight": SENTIMENT_WEIGHT, "regime_weight": REGIME_WEIGHT,
            "decision_threshold": COMBINED_DECISION_THRESHOLD,
        },
        "volatility": f"Existing backward-looking {HMM_VOLATILITY_WINDOW}-day annualized volatility; no recomputation",
        "annualization_factor": 252, "risk_free_rate": 0,
        "execution": "Target on row t executes on row t+1; first evaluation row flat; no manual shift",
        "return_convention": "Daily close-to-close proxy, not explicit next-open fills",
        "exposure_definition": "Mean and maximum absolute executed exposure across all test rows, including flat rows",
        "trade_definition": "Nonzero turnover days, including fractional resizing; not completed round trips",
        "cost_units": "Sum of per-period turnover times cost rate; not cash fees or compounded return drag",
        "invalid_volatility_policy": "Experiment rejects nonfinite or negative cached volatility; zero uses capped maximum size",
        "excluded_costs": ["stock borrow fees", "financing", "market impact beyond fixed slippage", "terminal liquidation"],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def main():
    root = Path(__file__).resolve().parents[1]
    model_path, model = load_cached_model(root)
    print(f"Source model cache: {model_path}")
    print(f"Selected target volatility: {TARGET_VOLATILITY:.2%} (prespecified; no search)")
    print("Directional signals: unchanged cached baseline configuration, not the separately tuned strategy.")
    validation = validation_preflight(model.loc[model["dataset_split"].eq("validation")])
    print(f"Validation preflight passed: {validation.index.min().date()} to {validation.index.max().date()}, {len(validation)} rows")
    print(f"Validation Combined sized metrics: {performance_metrics(validation)}")
    result = run_volatility_comparison(model)
    print(f"TEST PERIOD: {result.test_data.index.min().date()} to {result.test_data.index.max().date()}; rows: {len(result.test_data)}")
    print(f"Settings: {result.settings}")
    print(sizing_comparison_display(result.comparison).to_string(index=False))
    comparison = result.comparison.set_index("strategy")
    fixed, sized = comparison.loc["Combined Fixed"], comparison.loc["Combined Vol Sized"]
    print("Combined risk/return differences (sized minus fixed):")
    for column in ("annualized_volatility", "max_drawdown", "total_return"):
        print(f"{column}: {fixed[column]:.6%} -> {sized[column]:.6%}; delta {(sized[column] - fixed[column]) * 100:+.6f} percentage points")
    print(f"Sharpe: {fixed['sharpe_ratio']:.6f} -> {sized['sharpe_ratio']:.6f}; delta {sized['sharpe_ratio'] - fixed['sharpe_ratio']:+.6f}")
    print("Last 10 Combined Vol Sized rows (return is market return; daily returns/costs are fractions):")
    print(result.backtests["Combined Vol Sized"][DETAIL_COLUMNS].tail(10).to_string(float_format=lambda value: f"{value:.8f}"))
    directory = write_outputs(result, validation, model_path, root / "reports")
    figure_path = root / "reports/figures" / f"{directory.name}_equity.png"
    plot_equity(result, figure_path, title=f"{TICKER}: fixed vs volatility-sized test equity")
    print(f"Reports: {directory.resolve()}")
    print(f"Equity plot: {figure_path.resolve()}")
    print("Exposure metrics include neutral/flat rows. Trades include daily fractional rebalancing.")
    print("Summed costs are return units. Annualized metrics extrapolate a short test period.")
    print("No downloads, news requests, FinBERT inference, HMM fitting, signal changes or parameter search.")


if __name__ == "__main__":
    main()
