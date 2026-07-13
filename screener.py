"""
Binance USDT-M Futures market screener — main entry point.

Workflow per cycle (default: every hour):
  1. Fetch all active USDT perpetual symbols.
  2. BTC-correlation filter (Pearson <= 0.5 on both 30d and 90d windows);
     TradFi underlyings (stocks/gold/commodities) skip it.
  3. For each surviving coin and each timeframe (1h/4h/1d):
     structure check (3 HH+HL or 3 LH+LL), trendline with >= 3 touches
     ->  add to watchlist (table sorts by distance to the line).
  4. Re-check every existing watchlist entry against the 3-candle
     breakout/invalidation sequence  ->  remove broken entries.
  5. Persist state to watchlist.json and render the table.

Run:  python screener.py            (loop forever, hourly)
      python screener.py --once     (single scan, then exit)
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

import analysis
import config
import exchange as ex
import history
import llm_filter
import watchlist as wl

# ponytail: Windows consoles default to cp1252, which can't encode the arrows,
# dashes and box-drawing rich emits (→, —, ▲ …). Force UTF-8 on
# both streams so the table and every log line render instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# ponytail: when output is piped (background run, file redirect), rich sees no
# tty and falls back to 80 cols, mangling the table into "…". Pin a full width
# in that case; a real terminal still auto-detects its own window.
console = Console(width=None if sys.stdout.isatty() else 140)
log = logging.getLogger("screener")


# ============================================================================
# Scan-cycle building blocks
# ============================================================================

def build_correlation_universe(exchange, symbols: list[str],
                               tradfi: set[str]) -> dict:
    """
    Stage-1 filter: return {symbol: corr-dict} for every coin whose Pearson
    correlation with BTC is <= threshold on BOTH lookbacks.

    TradFi underlyings (stocks, gold, commodities) skip the filter entirely —
    BTC correlation is meaningless for them (and new listings lack 90d data).

    BTC daily closes are fetched once and reused for every pair.
    """
    btc_df = ex.fetch_ohlcv_df(exchange, config.BTC_SYMBOL, "1d",
                               config.DAILY_CANDLE_LIMIT)
    if btc_df is None:
        log.error("Could not fetch BTC daily data — skipping this cycle.")
        return {}
    btc_closes = btc_df["close"]

    passed: dict[str, dict] = {}
    for n, symbol in enumerate(symbols, 1):
        if symbol == config.BTC_SYMBOL:
            continue                      # BTC vs itself is corr 1.0 anyway
        if symbol in tradfi:
            passed[symbol] = {30: None, 90: None}   # renders "-" in the table
            continue
        df = ex.fetch_ohlcv_df(exchange, symbol, "1d",
                               config.DAILY_CANDLE_LIMIT)
        if df is None:
            continue
        ok, corrs = analysis.passes_correlation_filter(df["close"], btc_closes)
        if ok:
            passed[symbol] = corrs
        if n % 50 == 0:
            log.info("Correlation filter: %d/%d symbols checked, %d passed",
                     n, len(symbols), len(passed))
    return passed


def fetch_cached(exchange, cache: dict, symbol: str, tf: str):
    """
    Per-cycle OHLCV cache: a symbol on the watchlist that also re-qualified
    this cycle would otherwise be fetched twice per timeframe.
    """
    key = (symbol, tf)
    if key not in cache:
        cache[key] = ex.fetch_ohlcv_df(exchange, symbol, tf,
                                       config.CANDLE_LIMIT)
    return cache[key]


def scan_structures(exchange, candidates: dict, watchlist: dict,
                    cache: dict) -> None:
    """
    Stage-2: run the trend/trendline/distance pipeline on every candidate
    for every timeframe and upsert qualifiers into the watchlist.
    """
    for symbol, corrs in candidates.items():
        for tf in config.TIMEFRAMES:
            df = fetch_cached(exchange, cache, symbol, tf)
            evaluation = analysis.evaluate_symbol_timeframe(
                df, config.TIMEFRAME_MS[tf])
            if evaluation:
                # Refitting every cycle churns anchor_ts and drops the LLM verdict,
                # which re-judges the whole watchlist each scan (real API cost). If
                # the line is geometrically unchanged, keep the existing entry as-is
                # (verdict + tl_fit_ts intact) and skip the upsert.
                prev = watchlist.get(wl.entry_key(symbol, tf))
                if prev and analysis.same_line(prev["trendline"],
                                               evaluation["trendline"],
                                               config.TIMEFRAME_MS[tf]):
                    continue
                wl.upsert_entry(watchlist, symbol, tf, evaluation, corrs)
                log.info("QUALIFIED %s %s (%strend, dist %.3f%%)",
                         symbol, config.TIMEFRAMES[tf],
                         evaluation["direction"], evaluation["distance_pct"])


def prune_broken_entries(exchange, watchlist: dict, cache: dict,
                         active_symbols: set[str] | None) -> None:
    """
    Stage-3: re-check every watchlist entry and drop those where the
    3-candle breakout sequence has completed since its trendline was
    (re)fitted, plus entries whose symbol was delisted from the exchange.
    Surviving entries refresh their live distance-to-trendline for display.

    `active_symbols` is the FULL universe (never the --max-symbols-capped
    debug subset); None skips the delisting check entirely.
    """
    for key in list(watchlist.keys()):
        entry = watchlist[key]
        tf = entry["timeframe"]
        if tf not in config.TIMEFRAMES:      # tf removed from config (e.g. 15m)
            wl.remove_entry(watchlist, key, f"timeframe {tf} disabled")
            continue
        tf_ms = config.TIMEFRAME_MS[tf]

        if active_symbols is not None and entry["symbol"] not in active_symbols:
            wl.remove_entry(watchlist, key, "symbol delisted / inactive")
            continue

        df = fetch_cached(exchange, cache, entry["symbol"], tf)
        if df is None:
            continue                      # transient failure: keep, retry next cycle

        broken, reason = analysis.check_breakout(
            df, entry["trendline"], entry["direction"], tf_ms,
            # Old-schema entries (pre tl_fit_ts) fall back to added_ts.
            since_ts=entry.get("tl_fit_ts", entry["added_ts"]))
        if broken:
            history.log_closed(entry, reason)
            wl.remove_entry(watchlist, key, reason)
            continue

        # Refresh display fields against the newest closed candle.
        last = df.iloc[-1]
        tl_now = analysis.trendline_value_at(entry["trendline"],
                                             int(last["ts"]), tf_ms)
        entry["distance_pct"] = round(
            analysis.distance_to_trendline_pct(float(last["close"]), tl_now), 3)
        entry["close"] = float(last["close"])
        entry["updated_ts"] = int(time.time() * 1000)


# ============================================================================
# Terminal output
# ============================================================================

def render_watchlist(watchlist: dict, cycle_no: int, elapsed: float) -> None:
    """Pretty-print the current watchlist with rich."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = (f"CRYPTO SCREENER — cycle #{cycle_no} @ {now} "
             f"({elapsed:.0f}s) — {len(watchlist)} entries")

    table = Table(title=title, header_style="bold cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Coin", style="bold")
    table.add_column("TF", justify="center")
    table.add_column("Trend", justify="center")
    table.add_column("Dist→TL", justify="right")
    table.add_column("Touches", justify="right")
    table.add_column("Corr 30d", justify="right")
    table.add_column("Corr 90d", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("AI", justify="center")
    table.add_column("Added (UTC)", justify="center", style="dim")

    # .get() everywhere non-essential: entries restored from an older
    # watchlist.json may lack cosmetic fields — never crash the render.
    # Group by timeframe in config order (h1, h4, d1), then by distance.
    tf_order = {tf: i for i, tf in enumerate(config.TIMEFRAMES)}
    entries = sorted(watchlist.values(),
                     key=lambda e: (tf_order.get(e["timeframe"], 99),
                                    abs(e.get("distance_pct", 0.0))))
    for n, e in enumerate(entries, 1):
        trend_txt = ("[green]▲ UP[/green]" if e["direction"] == "up"
                     else "[red]▼ DOWN[/red]")
        added = datetime.fromtimestamp(
            e["added_ts"] / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
        coin = e["symbol"].split("/")[0] + "USDT"
        corr30, corr90 = e.get("corr_30d"), e.get("corr_90d")
        llm = e.get("llm")
        ai_txt = ("-" if llm is None
                  else "[green]✓[/green]" if llm.get("valid") else "[red]✗[/red]")
        table.add_row(
            str(n), coin, config.TIMEFRAMES[e["timeframe"]], trend_txt,
            f"{e.get('distance_pct', 0.0):+.3f}%",
            str(e["trendline"].get("touches", "-")),
            f"{corr30:.2f}" if corr30 is not None else "-",
            f"{corr90:.2f}" if corr90 is not None else "-",
            f"{e.get('close', 0.0):.6g}", ai_txt, added,
        )

    console.print()
    if entries:
        console.print(table)
    else:
        console.print(f"[yellow]{title}[/yellow]")
        console.print("[yellow]Watchlist is empty — no setups matched "
                      "this cycle.[/yellow]")
    console.print()


# ============================================================================
# Main loop
# ============================================================================

def run_cycle(exchange, watchlist: dict, cycle_no: int,
              max_symbols: int | None) -> None:
    started = time.time()
    log.info("===== Scan cycle #%d starting =====", cycle_no)

    symbols = ex.get_usdt_perp_symbols(exchange)
    # Delisting checks always use the FULL universe: a --max-symbols debug
    # cap must not make out-of-range entries look "delisted".
    active_symbols = None if max_symbols else set(symbols)
    if max_symbols:
        symbols = symbols[:max_symbols]
    log.info("Universe: %d USDT perpetual symbols", len(symbols))

    tradfi = ex.get_tradfi_symbols(exchange)
    candidates = build_correlation_universe(exchange, symbols, tradfi)
    log.info("Correlation filter passed: %d symbols", len(candidates))

    cache: dict = {}                     # per-cycle (symbol, tf) -> df
    scan_structures(exchange, candidates, watchlist, cache)
    prune_broken_entries(exchange, watchlist, cache, active_symbols)

    # Persist scan results before the (slow, failure-prone) LLM pass so an
    # exchange/LLM error or Ctrl+C mid-filter can't discard this cycle's work.
    wl.save_watchlist(watchlist)

    if config.LLM_FILTER:
        llm_filter.apply_filter(
            watchlist, cache,
            lambda sym, tf: fetch_cached(exchange, cache, sym, tf))

    wl.save_watchlist(watchlist)
    render_watchlist(watchlist, cycle_no, time.time() - started)


def main_loop(once: bool, interval: int, max_symbols: int | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[RichHandler(console=console,
                                              show_path=False)])
    # ccxt logs raw urls on DEBUG; keep third-party noise down.
    logging.getLogger("ccxt").setLevel(logging.WARNING)

    exchange = ex.build_exchange()
    watchlist = wl.load_watchlist()
    log.info("Loaded %d watchlist entries from %s",
             len(watchlist), config.WATCHLIST_FILE)

    cycle_no = 0
    while True:
        cycle_no += 1
        cycle_start = time.time()
        try:
            run_cycle(exchange, watchlist, cycle_no, max_symbols)
        except KeyboardInterrupt:
            # Save whatever the interrupted cycle managed to update before exit.
            wl.save_watchlist(watchlist)
            raise
        except Exception:
            # One bad cycle (exchange outage etc.) must not kill the loop.
            log.exception("Cycle #%d failed — retrying next interval", cycle_no)
            wl.save_watchlist(watchlist)

        if once:
            break
        # Sleep to the next interval boundary from cycle START, so the
        # cadence stays hourly instead of drifting by the scan duration.
        sleep_s = interval - ((time.time() - cycle_start) % interval)
        log.info("Sleeping %.1f minutes until next cycle…", sleep_s / 60)
        try:
            time.sleep(sleep_s)
        except KeyboardInterrupt:
            raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance futures crypto screener")
    p.add_argument("--once", action="store_true",
                   help="run a single scan cycle and exit")
    p.add_argument("--interval", type=int, default=config.SCAN_INTERVAL_SEC,
                   help="seconds between cycles (default 3600)")
    p.add_argument("--max-symbols", type=int, default=None,
                   help="cap the symbol universe (useful for quick tests)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip the Claude second-pass chart filter")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_llm:
        config.LLM_FILTER = False
    try:
        main_loop(args.once, args.interval, args.max_symbols)
    except KeyboardInterrupt:
        console.print("\n[bold]Screener stopped by user.[/bold]")
        sys.exit(0)
