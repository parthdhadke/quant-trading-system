# Multi-ticker robustness

The fixed basket is AAPL, MSFT, NVDA, JPM and XOM, over 2024-01-01 inclusive
through 2025-01-01 exclusive. Each asset keeps its actual price calendar and
chronological 60/20/20 feature split; no dates or missing news are fabricated.

## Locked configuration

The runner reads the existing `best_validation_config.json` from
`reports/tuning/AAPL_validation_27627953ee6a`. It accepts only a selection marked
validation-only with no test metrics used for selection. The same configuration
is applied to every ticker: 3 HMM states, sentiment thresholds +0.20/-0.25,
sentiment/regime weights 0.75/0.25 and decision threshold 0.50. HMM/scaler fitting
is still asset-specific and training-only; no per-asset hyperparameters are
selected. Existing causal continuous inference is reused.

Costs, split ratios, trailing volatility window and sizing settings come from
`config.py`. The 15% volatility target is prespecified, not tuned. Validation
preflight precedes test evaluation. Fixed and sized variants share directions,
test dates, market returns, costs and the existing single t+1 shift. Each test
starts flat. The five reported strategies are Buy & Hold, Sentiment Only,
Regime Only, Combined and Combined Volatility Sized.

These results use the locked selected configuration, not the earlier 2-state
baseline. AAPL was the selection asset and is not independent evidence of
cross-asset generalization. Prior baseline and tuning reports are not replaced.

## Caches and resumption

The existing price loader now accepts a cache directory and optional range
checking. New downloads retain ticker and requested-range metadata. Extensions
download only uncovered ranges into reusable fragment caches and preserve
existing observations when merging. The legacy AAPL price cache uses the
2024 requested interval explicitly documented in the project handoff. Other
legacy caches without range provenance fail closed rather than assume coverage.
An empty price response is an error, not fabricated price rows.

The existing historical news backfill performs raw-cache completeness checking,
missing-range downloads, FinBERT processing and daily checkpoints. A single
rolling-24-hour news request ledger is shared across the basket and existing
backfill workflow. Per-run request accounting does not reset between tickers
and counts attempts even when processing fails. Provider quota refusal stops
the basket; subsequent tickers remain explicitly listed with missing coverage.
No paid plan or API-limit bypass is used. Calls made outside this workflow with
the same key are not visible in the local ledger; provider refusals still apply.

One ticker's ordinary failure does not remove others' completed files. Completed
reports and their source hashes are verified before skipping the ticker on
rerun. Report identities include the basket, period and global settings. Changed
completed sources or artifacts are reported as errors rather than silently
accepted. Do not run simultaneous processes against the same robustness report
directory. The existing news ledger also has a request-level lock.

## Outputs and interpretation

`reports/robustness/multi_ticker_<identity>/` contains:

- `multi_ticker_results.parquet`: all requested metrics per completed asset/strategy.
- `multi_ticker_summary.parquet`: means, medians, positive counts and benchmark wins.
- `metadata.json`: global selection, all requested tickers, statuses and missing ranges.
- A directory per completed ticker with comparison, five backtests and equity curves.

Exposure is absolute executed exposure including flat rows. Trades count
turnover events, including fractional resizing. Costs are summed daily return
fractions, not cash fees or compounded cost drag. Return calculations preserve
the existing close-to-close proxy, not explicit next-open fills. Borrow fees,
financing and terminal liquidation are not modeled.

Aggregates are equal-weight summaries of metrics, not pooled portfolio returns.
Only validated completed tickers enter them; incomplete assets are never filled
with zero performance. Classification rules are fixed descriptive heuristics:

- Fewer than three completed assets: insufficient coverage.
- No assets with positive net return: weak.
- All positive and all beating Buy & Hold: consistent; the benchmark itself
  needs all positive returns and cannot beat itself.
- Mixed return signs and at least a 20-percentage-point return spread: highly ticker-dependent.
- Otherwise: mixed.

These labels are not statistical significance claims. An incomplete basket is
provisional, subject to coverage bias, and cannot establish generalization.

## Commands

From `C:\Users\dhadk\quant-trading-system` in PowerShell:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_robustness -v
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m scripts.run_multi_ticker --max-requests 20 --daily-budget 20
```

Use the same experiment command to resume on another day. Completed tickers and
completed news ranges are skipped. Exit code 2 means the basket is incomplete;
the runner prints reasons, missing ranges, report path and exact resume command.
No dashboard, messaging bot, portfolio optimization or additional models are added.
