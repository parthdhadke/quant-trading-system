import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from config import END_DATE, START_DATE
from scripts.run_volatility_sizing import sizing_comparison_display
from src.robustness import DEFAULT_TICKERS, SELECTION_FILE, MultiTickerRobustness, load_global_settings
from src.robustness_data import RobustnessDataProvider


def main():
    parser = argparse.ArgumentParser(description="Cache-first, resumable cross-ticker robustness without per-ticker tuning")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--selection", default=SELECTION_FILE)
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--daily-budget", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-days", type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = load_global_settings(root, args.selection)
    experiment = MultiTickerRobustness(root, settings, args.tickers, args.start, args.end)
    provider = RobustnessDataProvider(root, settings, args.start, args.end, args.max_requests,
                                      args.daily_budget, args.batch_size, args.chunk_days)
    print("Locked shared configuration (no per-ticker selection):", json.dumps(settings, sort_keys=True), flush=True)
    results, summary, states = experiment.run(provider)
    print("TICKERS COMPLETED:", [ticker for ticker, state in states.items() if state["status"] == "completed"])
    print("TICKERS INCOMPLETE:", [ticker for ticker, state in states.items() if state["status"] != "completed"])
    for ticker, state in states.items():
        if state["status"] != "completed":
            print(ticker, json.dumps(state, indent=2))
    if not results.empty:
        print("PER-TICKER RESULTS (costs are summed return fractions; exposures use executed positions):")
        print(sizing_comparison_display(results).to_string(index=False))
    print("CROSS-TICKER SUMMARY (unpooled; returns/drawdowns are fractions):")
    print(summary.to_string(index=False))
    print("AAPL was used for configuration selection. Incomplete baskets cannot establish generalization.")
    print(f"Reports: {experiment.directory}")
    resume = (f".\\venv\\Scripts\\python.exe -m scripts.run_multi_ticker --tickers {' '.join(experiment.tickers)}"
              f" --start {experiment.start} --end {experiment.end} --max-requests {args.max_requests or args.daily_budget}"
              f" --daily-budget {args.daily_budget} --chunk-days {args.chunk_days} --batch-size {args.batch_size}"
              f' --selection "{args.selection}"')
    print(f"RESUME from {root}: {resume}")
    return 0 if all(state["status"] == "completed" for state in states.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
