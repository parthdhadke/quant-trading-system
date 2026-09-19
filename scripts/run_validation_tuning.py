import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from config import (
    COMMISSION_BPS, END_DATE, HMM_N_ITER, HMM_RANDOM_STATE, HMM_VOLATILITY_WINDOW,
    SLIPPAGE_BPS, START_DATE, TICKER, TRAIN_RATIO, VALIDATION_RATIO,
)
from scripts.run_ablation import comparison_display, load_cached_model
from src.integration import build_model_dataset
from src.regime_pipeline import CausalHMMRegimePipeline
from src.tuning import (
    evaluate_selected_test, generate_parameter_grid, get_validation_regimes,
    robustness_summary, search_validation_parameters, select_best_validation_config,
)


def load_sources(root):
    paths = {
        "prices": root / f"data/raw/prices/{TICKER}.parquet",
        "sentiment": root / f"data/processed/sentiment/{TICKER}_{pd.Timestamp(START_DATE):%Y%m%d}_{pd.Timestamp(END_DATE):%Y%m%d}.parquet",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required cache missing: {path.resolve()}; no download allowed")
    return paths, pd.read_parquet(paths["prices"]), pd.read_parquet(paths["sentiment"])


def load_baseline_combined(report_root, source_model_path):
    matches = []
    for metadata_path in Path(report_root).glob("backtests/*_baseline_*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (Path(metadata["source_model_path"]).resolve() == Path(source_model_path).resolve()
                and metadata["settings"]["commission_bps"] == COMMISSION_BPS
                and metadata["settings"]["slippage_bps"] == SLIPPAGE_BPS):
            comparison = pd.read_parquet(metadata_path.parent / "comparison.parquet")
            matches.append(comparison.loc[comparison["strategy"] == "Combined"].iloc[0])
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching locked baseline report")
    return matches[0]


def save_validation_selection(root, source_model_path, results, selected, nearby, sensitivity, cache_paths):
    identity = {
        "version": 1, "source_model_sha256": hashlib.sha256(Path(source_model_path).read_bytes()).hexdigest(),
        "grid_size": len(results), "minimum_trades": 2, "commission_bps": COMMISSION_BPS,
        "slippage_bps": SLIPPAGE_BPS, "hmm_state_counts": [2, 3],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    directory = root / "reports/tuning" / f"{TICKER}_validation_{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    results.to_parquet(directory / "all_validation_results.parquet", index=False)
    results.head(10).to_parquet(directory / "top_10_validation_results.parquet", index=False)
    nearby.to_parquet(directory / "robustness_neighbors.parquet", index=False)
    best_document = {**selected, "minimum_trades": 2, "ranking_metric": "validation_sharpe_ratio",
                     "tie_breakers": ["net_return", "max_drawdown", "lower_turnover", "configuration_id"],
                     "validation_only": True, "test_metrics_used_for_selection": False}
    (directory / "best_validation_config.json").write_text(json.dumps(best_document, indent=2), encoding="utf-8")
    (directory / "robustness_summary.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    manifest = {**identity, "validation_regime_caches": [str(path) for path in cache_paths],
                "selection_saved_before_test_evaluation": True, "tuned_combined_test_evaluations": 0}
    (directory / "experiment_state.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory, manifest


def save_final_outputs(directory, final_result, baseline_combined, manifest):
    final_result.comparison.to_parquet(directory / "final_test_comparison.parquet", index=False)
    final_result.equity_curves.to_parquet(directory / "final_test_equity_curves.parquet", index=True)
    for name, frame in final_result.backtests.items():
        slug = name.lower().replace(" & ", "_and_").replace(" ", "_")
        frame.to_parquet(directory / f"final_{slug}.parquet", index=True)
    tuned = final_result.comparison.set_index("strategy").loc["Combined Tuned"]
    baseline_comparison = {
        "baseline_total_return": float(baseline_combined["total_return"]),
        "baseline_sharpe_ratio": float(baseline_combined["sharpe_ratio"]),
        "tuned_total_return": float(tuned["total_return"]),
        "tuned_sharpe_ratio": float(tuned["sharpe_ratio"]),
        "net_return_change": float(tuned["total_return"] - baseline_combined["total_return"]),
        "sharpe_change": float(tuned["sharpe_ratio"] - baseline_combined["sharpe_ratio"]),
    }
    (directory / "baseline_vs_tuned.json").write_text(json.dumps(baseline_comparison, indent=2), encoding="utf-8")
    manifest["tuned_combined_test_evaluations"] = 1
    manifest["final_test_evaluation_complete"] = True
    (directory / "experiment_state.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return baseline_comparison


def main():
    root = Path(__file__).resolve().parents[1]
    model_path, baseline_model = load_cached_model(root)
    source_paths, prices, sentiment = load_sources(root)
    grid = generate_parameter_grid()
    result_frames, regime_cache_paths = [], []
    validation_ranges = []
    for state_count in (2, 3):
        cache_path, regimes = get_validation_regimes(
            prices, TICKER, START_DATE, END_DATE, state_count,
            cache_dir=root / "data/processed/tuning/regimes",
        )
        regime_cache_paths.append(cache_path)
        validation = build_model_dataset(
            prices, sentiment, regimes, ticker=TICKER, start_date=START_DATE, end_date=END_DATE,
        )
        validation_ranges.append((validation["date"].min(), validation["date"].max(), len(validation)))
        result_frames.append(search_validation_parameters(validation, state_count, grid=grid, minimum_trades=2))
    results = pd.concat(result_frames, ignore_index=True).sort_values(
        ["eligible", "sharpe_ratio", "total_return", "max_drawdown", "total_turnover", "config_id"],
        ascending=[False, False, False, False, True, True], na_position="last",
    ).reset_index(drop=True)
    selected = select_best_validation_config(results)
    nearby, sensitivity = robustness_summary(results, selected)
    directory, manifest = save_validation_selection(
        root, model_path, results, selected, nearby, sensitivity, regime_cache_paths,
    )
    print(f"Validation period: {validation_ranges[0][0].date()} to {validation_ranges[0][1].date()}")
    print(f"Validation rows: {validation_ranges[0][2]}")
    print(f"Parameter grid: {len(grid)} per HMM state count; {len(results)} total candidates")
    print("Top 10 validation configurations:")
    top_columns = ["buy_threshold", "sell_threshold", "sentiment_weight", "regime_weight",
                   "decision_threshold", "hmm_state_count", "total_return", "sharpe_ratio",
                   "max_drawdown", "number_of_trades", "total_turnover"]
    print(results.loc[results["eligible"], top_columns].head(10).to_string(index=False))
    print("Best eligible configuration per HMM state count:")
    print(results.loc[results["eligible"]].groupby("hmm_state_count", sort=True).head(1)[top_columns].to_string(index=False))
    print(f"Selected validation configuration: {json.dumps(selected, sort_keys=True)}")
    print("Nearby validation robustness configurations:")
    print(nearby[["relationship", *top_columns[:7], "sharpe_ratio", "max_drawdown"]].to_string(index=False))
    print(f"Robustness: {sensitivity['classification']} (maximum nearby Sharpe drop {sensitivity['maximum_neighbor_sharpe_drop']:.4f})")
    selected_pipeline = CausalHMMRegimePipeline(
        n_states=selected["hmm_state_count"], volatility_window=HMM_VOLATILITY_WINDOW,
        random_state=HMM_RANDOM_STATE, n_iter=HMM_N_ITER, train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO, processed_dir=root / "data/processed/regimes",
    )
    selected_regimes = selected_pipeline.get_regimes(TICKER, START_DATE, END_DATE, prices)
    selected_model = build_model_dataset(
        prices, sentiment, selected_regimes, ticker=TICKER, start_date=START_DATE, end_date=END_DATE,
    )
    final_result = evaluate_selected_test(
        selected_model, selected, commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
    )
    baseline = load_baseline_combined(root / "reports", model_path)
    comparison = save_final_outputs(directory, final_result, baseline, manifest)
    print("Final test comparison (one locked tuned-Combined evaluation):")
    print(comparison_display(final_result.comparison).to_string(index=False))
    print("Combined baseline versus tuned:")
    print(json.dumps(comparison, indent=2))
    print(f"Tuning outputs: {directory.resolve()}")
    print(f"Validation results: {(directory / 'all_validation_results.parquet').resolve()}")
    print(f"Selected config: {(directory / 'best_validation_config.json').resolve()}")
    print(f"Robustness report: {(directory / 'robustness_summary.json').resolve()}")
    print(f"Final test comparison: {(directory / 'final_test_comparison.parquet').resolve()}")
    print("No downloads, test-based selection, volatility sizing, parameter revision, or repeated tuned test evaluation performed.")


if __name__ == "__main__":
    main()
