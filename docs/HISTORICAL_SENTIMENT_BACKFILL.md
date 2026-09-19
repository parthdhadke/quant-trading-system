# Historical sentiment backfill

Run all commands in PowerShell from the project root:

```powershell
Set-Location 'C:\Users\dhadk\quant-trading-system'
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Start or resume the AAPL research-period backfill:

```powershell
.\venv\Scripts\python.exe -u -m scripts.backfill_sentiment --ticker AAPL --start 2024-01-01 --end 2025-01-01 --chunk-days 30 --max-requests 20 --daily-budget 20 --batch-size 16
```

Process only already-cached news, without any Alpha Vantage requests:

```powershell
.\venv\Scripts\python.exe -m scripts.backfill_sentiment --ticker AAPL --start 2024-01-01 --end 2025-01-01 --max-requests 0
```

FinBERT uses the existing `ProsusAI/finbert` implementation. If its weights are already downloaded and Hugging Face access is unavailable, these optional settings force local model loading:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

## Coverage and checkpoints

All requested publication intervals are UTC, with an inclusive start and exclusive end. Deterministic 30-day chunks are anchored to the requested start. Previously cached subranges are subtracted before fetching. The existing client's response-cap splitting is retained; completed child ranges are reusable even when their parent request is interrupted. Raw responses are saved before FinBERT runs.

Coverage comes from validated query-range envelopes, not the earliest/latest article timestamp. Empty, successfully retrieved feeds prove no-news coverage; absent or invalid raw caches do not. The workflow cannot establish that the provider's historical archive includes every article originally published or reflects a point-in-time news archive.

News is normalized and deduplicated with the existing article identity. The full cached price trading calendar is used for alignment, so Friday after-close and weekend news can reach Monday across chunk boundaries. The existing 16:00 America/New_York cutoff is preserved, including its existing limitation that early-close sessions are not modeled separately. The first requested trading day's information window is clipped at the requested UTC start; earlier news is not assumed available.

A trading day is finalized only when its preceding information window is covered. Consequently, a legacy processed cache's first row may need rescoring when its original query omitted the preceding after-close/weekend interval. Other valid rows are reused. Original caches are not deleted or overwritten with partial results.

Progress checkpoints carry per-day source fingerprints, including article identities, the information window, model identity, timezone and cutoff. Changed source identities invalidate only affected rows. Missing dates are never filled with neutral sentiment merely to complete the dataset.

Paths for the initial AAPL research range:

- Raw ranges: `data/raw/news/AAPL_YYYYMMDD_YYYYMMDD.json`
- Request ledger: `data/raw/news/.backfill/request_budget.json`
- Incremental checkpoint: `data/processed/sentiment/backfill/AAPL_20240101_20250101_progress.parquet`
- Complete daily sentiment: `data/processed/sentiment/AAPL_20240101_20250101.parquet`
- Aligned dataset, only after completeness passes: the existing configuration/content-keyed cache under `data/processed/model/`

The progress parquet includes an internal `_source_signature` column. The final daily sentiment parquet contains the existing six public columns. Use the backfill result's `complete` status and coverage validator before modelling, not the presence of a progress filename.

Completeness requires every expected trading date, sorted unique dates, no unexpected dates, finite required fields, valid probabilities and counts, and exact neutral values for genuine no-news days. A completed range is also checked against the full raw publication interval.

## Request limits and failures

The default limit is 20 attempted calls per run and per rolling 24 hours, with at least 12.5 seconds between calls. This is deliberately below Alpha Vantage's documented standard allowance of 25 requests/day. Calls made elsewhere with the same API key are unknown to this ledger; choose a smaller `--max-requests` if you have used the key elsewhere. Do not delete the ledger to bypass limits.

Every attempted HTTP call is counted, including recursive split requests and failures. A transport failure is not proof the provider consumed quota, so reported attempts are conservative. Provider quota refusals stop immediately and persist a 24-hour cooldown. There are no automatic request retries. Successful raw chunks and earlier sentiment checkpoints remain available on restart.

The ledger lock prevents concurrent requests from this workflow. A forcibly killed process can leave `request_budget.lock`; verify no backfill process is running before manually removing that specific lock. Keep the JSON ledger. Normal failures release the lock automatically.

Exit status is 0 for complete sentiment and 2 for partial coverage. The report gives remaining raw intervals, missing trading-date spans, request outcomes, article counts and exact cache paths. Resume with the same command after quota recovery or after resolving the reported failure.

Full sentiment coverage triggers the existing integration pipeline using the already-cached HMM history and original chronological split labels. Missing HMM caches are reported without model fitting. No backtesting, threshold selection, HMM tuning or signal execution occurs.

## Implementation locations

- `src/sentiment_coverage.py`: interval operations, calendar validation, raw coverage inspection and sentiment validation.
- `src/news_budget.py`: persistent request accounting, pacing and quota cooldowns.
- `src/news_backfill.py`: cache reuse, incremental processing and resumable orchestration.
- `scripts/backfill_sentiment.py`: command-line runner, coverage report and completeness-gated model integration.
- `tests/test_news_backfill.py`: local deterministic tests; no real API or model downloads.

Provider references: [NEWS_SENTIMENT documentation](https://www.alphavantage.co/documentation/#news-sentiment) and [standard API request allowance](https://www.alphavantage.co/premium/).
