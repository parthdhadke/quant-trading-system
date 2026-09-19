import json
import time
from pathlib import Path


class NewsRequestLimit(RuntimeError):
    pass


class BudgetedNewsSession:
    def __init__(self, session, ledger_path, max_requests=20, daily_budget=20, min_interval=12.5, clock=None):
        if not 0 <= max_requests <= 25 or not 1 <= daily_budget <= 25:
            raise ValueError("Request budgets must be between 0 and 25 (daily at least 1)")
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        self.session = session
        self.ledger_path = Path(ledger_path)
        self.max_requests = max_requests
        self.daily_budget = daily_budget
        self.min_interval = min_interval
        self.clock = clock or time.time
        self.calls = []

    def _read(self):
        if not self.ledger_path.exists():
            return {"requests": [], "blocked_until": 0}
        ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if not isinstance(ledger.get("requests"), list) or "blocked_until" not in ledger:
            raise ValueError("Invalid request ledger; refusing to reset API accounting")
        return ledger

    def _save(self, ledger):
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.ledger_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        temporary.replace(self.ledger_path)

    def get(self, url, params, timeout):
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.ledger_path.with_suffix(".lock")
        try:
            lock = lock_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise NewsRequestLimit(f"Concurrent or interrupted API request lock exists: {lock_path.resolve()}") from exc
        try:
            return self._get(url, params, timeout)
        finally:
            lock.close()
            lock_path.unlink()

    def _get(self, url, params, timeout):
        ledger = self._read()
        now = self.clock()
        recent = [item for item in ledger["requests"] if item["at"] > now - 86400]
        if len(self.calls) >= self.max_requests:
            raise NewsRequestLimit(f"Per-run request budget ({self.max_requests}) reached")
        if ledger["blocked_until"] > now:
            raise NewsRequestLimit("Previous provider quota refusal: 24-hour cooldown is still active")
        if len(recent) >= self.daily_budget:
            raise NewsRequestLimit(f"Local rolling 24-hour request budget ({self.daily_budget}) reached")
        if recent:
            delay = max(0, self.min_interval - (now - recent[-1]["at"]))
            if delay:
                time.sleep(delay)
        record = {
            "at": self.clock(), "ticker": params["tickers"],
            "time_from": params["time_from"], "time_to": params["time_to"], "outcome": "attempted",
        }
        ledger["requests"] = recent + [record]
        self._save(ledger)
        self.calls.append(record)
        try:
            response = self.session.get(url, params=params, timeout=timeout)
            status = getattr(response, "status_code", 200)
            if status == 429:
                raise NewsRequestLimit("Alpha Vantage returned HTTP 429")
            response.raise_for_status()
            payload = response.json()
            error = " ".join(str(payload.get(key, "")) for key in ("Note", "Information", "Error Message"))
            lowered = error.lower()
            if any(term in lowered for term in ("rate limit", "call frequency", "requests per day", "requests/day", "api call volume")):
                raise NewsRequestLimit("Alpha Vantage refused the request because of its API quota")
            record["outcome"] = "provider_error" if error.strip() else "received"
            return response
        except NewsRequestLimit:
            record["outcome"] = "quota_refused"
            ledger["blocked_until"] = self.clock() + 86400
            raise
        except Exception:
            record["outcome"] = "failed"
            raise
        finally:
            self._save(ledger)
