# Baseline ablation

Run these commands in PowerShell from the project root:

```powershell
Set-Location 'C:\Users\dhadk\quant-trading-system'
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -u -m scripts.run_ablation
```

The runner reads only the cached price, sentiment, regime and aligned-model parquet files matching the current `config.py`. Missing or mismatched caches cause an error. It does not download data, run FinBERT, fit the HMM, change signals, size for volatility, tune parameters or backtest training/validation rows.

## Test boundary and timing

All strategies use only rows already labelled `dataset_split == "test"`. Training and validation positions are not carried into the test. Each strategy starts with an executed position of zero on the first test row, including Buy & Hold. A target based on test row t becomes the executed position on test row t+1 and earns row t+1's close-to-close market return. Returns are not shifted backward.

This is an idealized daily close-to-close execution proxy. It is not an explicit next-open fill simulation and does not separately model the overnight gap between the signal close and the next open. The final target is not liquidated after the last test row, so no unobservable post-test return or terminal liquidation fee is added.

Buy & Hold has target +1 throughout the test split. Sentiment Only, Regime Only and Combined consume their existing cached target-signal columns without regenerating or shifting them in the ablation layer.

## Costs and metrics

Baseline commission and slippage are each 5 basis points. On each row, turnover is the absolute change in executed position and transaction cost is turnover times the combined rate. A long-to-short reversal is one trade event and two turnover units. `number_of_trades` counts rows with nonzero turnover, not completed round trips.

Total, gross and annualized returns are compounded daily. Annualized return and volatility use 252 trading periods. Sharpe is the annualized mean daily net return divided by its sample standard deviation, with a zero risk-free rate. A zero or undefined daily standard deviation is reported as Sharpe zero. Win rate is the proportion of positive daily net returns among rows with nonzero executed exposure; commission-only exit rows are excluded from its denominator. Drawdown includes initial capital as the initial high-water mark.

`total_transaction_cost` is the sum of daily turnover-cost rates. It is a return-unit diagnostic, not a cash total or the compounded performance drag. Gross minus net total return is printed separately as compounded cost drag.

The baseline excludes stock-borrow fees, financing, taxes, dynamic market impact and terminal liquidation. Profit factor is intentionally omitted at this stage because a daily-return definition can be confused with a trade-level profit factor.

## Outputs

Each reproducible run is stored below `reports/backtests/` in a directory keyed by test dates, source-model content and cost settings. It includes one full parquet per strategy, raw numeric comparison and position-count parquets, an equity-curve parquet, a readable text table, and metadata recording assumptions. The simple equity plot is saved under `reports/figures/`.
