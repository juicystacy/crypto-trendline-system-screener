"""
Persistent CSV log of setups that PASSED the LLM filter.

One row per setup lifecycle: a row is appended when the LLM first validates the
setup (breakout_date blank = still open), and completed with a breakout date +
outcome when the setup later breaks out and leaves the watchlist. Rows older
than HISTORY_KEEP_DAYS are dropped on every write.

Only LLM-passed setups are ever written here (log_closed no-ops on setups that
were never logged), so the file stays small. CSV, not xlsx: append-cheap, no
dependency, and it opens directly in Excel.

Deduped on symbol|timeframe (one open row per pair, matching the watchlist's own
keying) — NOT on the trendline coefficients, whose anchor_ts shifts every cycle.
"""

import csv
import logging
import os
import tempfile
import time
from datetime import datetime, timezone

import config

log = logging.getLogger("screener.history")

FIELDS = [
    "coin", "timeframe", "trend", "valid_date", "entry_price",
    "breakout_date", "exit_price", "distance_pct", "touches",
    "corr_30d", "corr_90d", "close", "llm_reason", "outcome", "logged_ts",
]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _fmt_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M")


def _load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError as exc:
        log.warning("Could not read history (%s) — treating as empty.", exc)
        return []


def _save(rows: list[dict], path: str) -> None:
    """Prune rows older than HISTORY_KEEP_DAYS, then atomic-write the CSV."""
    cutoff = _now_ms() - config.HISTORY_KEEP_DAYS * 86_400_000
    rows = [r for r in rows if int(r.get("logged_ts") or 0) >= cutoff]

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        # OneDrive/antivirus briefly lock the target during sync, so os.replace
        # can fail with a transient PermissionError — retry before giving up
        # (same pattern as watchlist.save_watchlist).
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.5)
    except OSError as exc:
        log.error("Failed to save history: %s", exc)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _row_pair(r: dict) -> str:
    # Reconstruct the dedup key from stored columns (coin already has USDT).
    return f'{r["coin"]}|{r["timeframe"]}'


def log_validated(entry: dict, path: str = None) -> None:
    """
    Record a setup the LLM just judged VALID. No-op if this symbol|timeframe
    already has an open row (avoids a duplicate every cycle it re-qualifies).
    """
    path = path or config.HISTORY_FILE
    coin = entry["symbol"].split("/")[0] + "USDT"
    tf_label = config.TIMEFRAMES.get(entry["timeframe"], entry["timeframe"])
    pair = f"{coin}|{tf_label}"

    rows = _load(path)
    if any(_row_pair(r) == pair and not r["breakout_date"] for r in rows):
        return

    corr30, corr90 = entry.get("corr_30d"), entry.get("corr_90d")
    rows.append({
        "coin": coin,
        "timeframe": tf_label,
        "trend": entry["direction"],
        "valid_date": _fmt_date(_now_ms()),
        "entry_price": "",              # filled manually
        "breakout_date": "",
        "exit_price": "",               # filled manually
        "distance_pct": f'{entry.get("distance_pct", 0.0):+.3f}',
        "touches": entry["trendline"].get("touches", ""),
        "corr_30d": f"{corr30:.2f}" if corr30 is not None else "",
        "corr_90d": f"{corr90:.2f}" if corr90 is not None else "",
        "close": f'{entry.get("close", 0.0):.6g}',
        "llm_reason": entry.get("llm", {}).get("reason", ""),
        "outcome": "",
        "logged_ts": _now_ms(),
    })
    _save(rows, path)


def log_closed(entry: dict, reason: str, path: str = None) -> None:
    """
    Complete the open row for this setup with its breakout date + outcome.
    No-op when the setup was never logged (i.e. never passed the LLM filter) —
    this is what keeps non-LLM setups out of the history file entirely.
    """
    path = path or config.HISTORY_FILE
    coin = entry["symbol"].split("/")[0] + "USDT"
    tf_label = config.TIMEFRAMES.get(entry["timeframe"], entry["timeframe"])
    pair = f"{coin}|{tf_label}"

    rows = _load(path)
    for r in rows:
        if _row_pair(r) == pair and not r["breakout_date"]:
            r["breakout_date"] = _fmt_date(_now_ms())
            r["outcome"] = reason
            _save(rows, path)
            return


def _demo() -> None:
    """Offline self-check: validate → dedup → close."""
    import shutil

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "history.csv")
    try:
        entry = {
            "symbol": "FOO/USDT:USDT", "timeframe": "4h", "direction": "up",
            "distance_pct": 1.234, "close": 0.5, "corr_30d": 0.1,
            "corr_90d": 0.2, "trendline": {"touches": 4},
            "llm": {"valid": True, "reason": "clean stepping pullbacks"},
        }
        log_validated(entry, path)
        log_validated(entry, path)                 # same pair, still open
        rows = _load(path)
        assert len(rows) == 1, f"dedup failed: {len(rows)} rows"
        assert rows[0]["coin"] == "FOOUSDT"
        assert rows[0]["timeframe"] == "h4"        # config label, not "4h"
        assert rows[0]["breakout_date"] == ""

        log_closed(entry, "3-candle breakout below support", path)
        rows = _load(path)
        assert len(rows) == 1
        assert rows[0]["breakout_date"], "breakout_date not filled"
        assert rows[0]["outcome"].startswith("3-candle")

        # A now-closed pair logging valid again opens a fresh row.
        log_validated(entry, path)
        assert len(_load(path)) == 2

        # Closing a setup that was never logged is a no-op.
        other = {**entry, "symbol": "BAR/USDT:USDT"}
        log_closed(other, "whatever", path)
        assert len(_load(path)) == 2
        print("history self-check OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _demo()
