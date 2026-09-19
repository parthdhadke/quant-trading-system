# Volatility-aware position sizing

The experiment compares the unchanged cached baseline directions at fixed size
against sized Sentiment, Regime and Combined variants, with Buy & Hold as a
reference. It does not use the separately tuned directional strategy.

## Configuration and causality

`config.py` supplies a prespecified 15% annualized volatility target, maximum
position size 1.0 and minimum 0.0. There is no target search or selection using
test performance. The runner checks validation first, then evaluates the seven
test strategies. No separate model-selection framework is introduced.

The existing `src/signals.py` helpers calculate
`position_size = clip(target_volatility / volatility, minimum, maximum)` and
`position = directional_signal * position_size`. Directions remain -1, 0 or +1.
The source is the integrated dataset's existing trailing annualized volatility,
computed from backward-looking returns with the configured 20-day window. There
is no centered window, full-sample volatility estimate or future return input.

Sizing does not shift positions. The existing `run_backtest()` applies exactly
one row of execution delay. All seven strategies use identical test dates,
returns, costs and starting capital, begin flat and omit terminal liquidation.
The existing close-to-close return proxy is retained; this is not a next-open
fill simulation.

## Invalid volatility

The scalar helper preserves its existing bounded fallback: None, NaN and
infinity use minimum size; zero and negative volatility use maximum size.
Finite positive values follow the ratio and clipping rule. Invalid sizing
settings and leverage above 1.0 are rejected.

The audited experiment is stricter about source data: nonfinite or negative
cached volatility fails before entering a backtest. It does not silently drop
dates, backfill estimates or pass invalid audit columns to the backtester.
Zero volatility is allowed and uses the capped maximum size.

## Metrics and interpretation

The existing evaluation functions supply return, volatility, Sharpe, drawdown,
win rate, turnover, trade-event and cost metrics. Additional exposure metrics
are the mean and maximum absolute **executed** positions across every test row,
including flat and neutral rows. The 15% target is a sizing input, not a
guarantee of 15% realized strategy volatility.

Trades count nonzero-turnover days, including resizing without a direction
change. Sized strategies can therefore have more trades while consuming less
turnover. Transaction costs are calculated only by the existing backtester.
Summed costs are return units, not cash fees or compounded return drag. Borrow
fees, financing and additional market impact are excluded. Annualized metrics
extrapolate a short test window and do not establish general superiority.

## Run from the project root

Folder: `C:\Users\dhadk\quant-trading-system`

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_volatility_sizing tests.test_volatility_experiment -v
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m scripts.run_volatility_sizing
```

The runner requires the existing source-matching aligned cache. Missing caches
raise an error; no downloading, news requests, FinBERT inference or HMM fitting
is performed. Reports are written to a distinct `reports/backtests/` directory
with a `volatility` name and a source/settings hash; baseline and tuning reports
are not overwritten. The corresponding equity plot is under `reports/figures/`.
The report includes all seven backtests, comparison, equity curves, validation
preflight, last ten Combined sized rows, and configuration/provenance metadata.

## Files

- `config.py`: sizing defaults after the backtest configuration.
- `src/signals.py`: existing sizing helpers, bounded finite validation and visible size column.
- `src/volatility_experiment.py`: validation preflight and seven-strategy orchestration.
- `scripts/run_volatility_sizing.py`: cached-only experiment and report output.
- `scripts/run_ablation.py`: optional plot title; baseline behavior unchanged.
- `tests/test_volatility_sizing.py`: scalar sizing, causality and fractional-cost tests.
- `tests/test_volatility_experiment.py`: comparison, split isolation and report tests.
