# Validation-only tuning

Run from `C:\Users\dhadk\quant-trading-system`:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -u -m scripts.run_validation_tuning
```

The search evaluates 225 fixed signal configurations for each of two HMM state counts, 2 and 3. The threshold values, paired weights, and decision thresholds are the explicit values in `src/tuning.py`. Candidates are ranked by validation Sharpe, then validation net return, maximum drawdown, lower turnover, and deterministic configuration identity. A candidate needs at least two validation turnover events. Insolvent, invalid, or non-finite candidates are retained in the full results as ineligible and cannot be selected.

State-count artifacts use a separate `data/processed/tuning/regimes/` cache keyed by state count, HMM settings, date range, and price-feature content. For each state count, the scaler and HMM are fitted only on the chronological training split. State meanings come only from causally filtered training states. Only validation observations are inferred afterward; test observations are neither inferred nor stored in these selection artifacts.

The tuning function accepts validation-labelled rows only. The selected configuration is written to `best_validation_config.json` before any final test evaluation. It is not written into `config.py`. Once locked, the selected HMM state count is loaded or fitted with the existing full causal regime pipeline, and the selected configuration is evaluated on the test split once. Buy & Hold, selected-threshold Sentiment Only, selected-state Regime Only, and Combined Tuned share the same test dates, returns, costs, and t+1 execution convention.

Robustness neighbors vary one grid family at a time by one adjacent value while holding the others fixed, plus the alternate HMM state count with the same signal configuration when eligible. The descriptive classification uses the largest Sharpe decrease from the selected validation result: at most 0.5 is `stable`, at most 1.5 is `moderately sensitive`, and more than 1.5 is `highly sensitive`. This is a transparent heuristic, not a statistical significance test.

All validation candidates, the top ten, HMM cache paths, selected JSON, robustness rows, final test backtests, final comparison, and baseline-versus-tuned results are saved under a content-keyed directory in `reports/tuning/`. Existing 2-state regime and baseline report caches are not overwritten.
