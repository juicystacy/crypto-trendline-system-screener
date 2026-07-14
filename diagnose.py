"""
Per-stage diagnostic: why does/doesn't a symbol qualify?

Usage: python diagnose.py TSM XAUT MAVIA JELLYJELLY BTCDOM BZ LIT

For each symbol x timeframe, reports which pipeline stage kills it
(correlation, range, swings, line fit, distance) — and when the line
fit fails, relaxes one constraint at a time to name the blocking one.
"""

import sys

import config
import analysis
import exchange as ex


def fit_with(pivots, side, anchor_ts, tf_ms, bars, **overrides):
    """Run fit_pivot_line with temporary config overrides."""
    saved = {k: getattr(config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)
    try:
        return analysis.fit_pivot_line(pivots, side, anchor_ts, tf_ms, bars=bars)
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


# Relaxations tried in order — first one that yields a line names the blocker.
RELAXATIONS = [
    ("no candle-close validation", {"_no_bars": True}),
    ("CLOSE_PIERCE_TOL_PCT 0.75 -> 2.0", {"CLOSE_PIERCE_TOL_PCT": 2.0}),
    ("MIN_SLOPE_PCT_PER_BAR 0.04 -> 0", {"MIN_SLOPE_PCT_PER_BAR": 0.0}),
    ("LINE_PIERCE_TOL_PCT 0.4 -> 1.0", {"LINE_PIERCE_TOL_PCT": 1.0}),
    ("TOUCH_TOLERANCE_PCT 0.5 -> 1.0", {"TOUCH_TOLERANCE_PCT": 1.0}),
    ("MAX_SWINGS_CONSIDERED 12 -> 24", {"MAX_SWINGS_CONSIDERED": 24}),
    ("everything relaxed at once", {"MIN_SLOPE_PCT_PER_BAR": 0.0,
                                    "LINE_PIERCE_TOL_PCT": 0.5,
                                    "TOUCH_TOLERANCE_PCT": 1.0,
                                    "MAX_SWINGS_CONSIDERED": 24,
                                    "_no_bars": True}),
]


def diagnose_side(pivots, side, anchor_ts, tf_ms, df):
    tl = analysis.fit_pivot_line(pivots, side, anchor_ts, tf_ms, bars=df)
    if tl is not None:
        return tl, None
    for name, ov in RELAXATIONS:
        ov = dict(ov)
        bars = None if ov.pop("_no_bars", False) else df
        if fit_with(pivots, side, anchor_ts, tf_ms, bars, **ov) is not None:
            return None, name
    return None, "no line even fully relaxed (pivots don't align)"


def main(symbols: list[str]):
    xch = ex.build_exchange()
    btc_d = ex.fetch_ohlcv_df(xch, config.BTC_SYMBOL, "1d",
                              config.DAILY_CANDLE_LIMIT)
    if btc_d is None:
        sys.exit("cannot fetch BTC daily data — check proxy/VPS connection")

    for base in symbols:
        symbol = f"{base.upper()}/USDT:USDT"
        print(f"\n=== {symbol} ===")

        d1 = ex.fetch_ohlcv_df(xch, symbol, "1d", config.DAILY_CANDLE_LIMIT)
        if d1 is None:
            print("  cannot fetch daily candles — symbol not on Binance perps?")
            continue
        ok, corrs = analysis.passes_correlation_filter(d1["close"],
                                                       btc_d["close"])
        print(f"  corr filter: {'PASS' if ok else 'FAIL'} "
              f"{ {w: None if c is None else round(c, 2) for w, c in corrs.items()} }"
              f" (threshold {config.CORR_THRESHOLD})")

        for tf, label in config.TIMEFRAMES.items():
            df = ex.fetch_ohlcv_df(xch, symbol, tf, config.CANDLE_LIMIT)
            if df is None or len(df) < config.SWING_ORDER * 2 + 10:
                print(f"  {label}: not enough candles")
                continue
            tf_ms = config.TIMEFRAME_MS[tf]

            lo = float(df["low"].min())
            rng = (float(df["high"].max()) - lo) / lo * 100.0
            if rng < config.MIN_RANGE_PCT:
                print(f"  {label}: FAIL range guard ({rng:.2f}% < "
                      f"{config.MIN_RANGE_PCT}%)")
                continue

            highs, lows = analysis.find_swings(df)
            anchor_ts = int(df["ts"].iloc[0])
            print(f"  {label}: {len(lows)} swing lows, {len(highs)} swing highs")

            for side, pivots in (("support", lows), ("resistance", highs)):
                tl, blocker = diagnose_side(pivots, side, anchor_ts, tf_ms, df)
                if tl is not None:
                    tl_now = analysis.trendline_value_at(
                        tl, int(df["ts"].iloc[-1]), tf_ms)
                    dist = analysis.distance_to_trendline_pct(
                        float(df["close"].iloc[-1]), tl_now)
                    slope_pct = abs(tl["slope"]) / tl_now * 100.0
                    print(f"    {side}: LINE FOUND — {tl['touches']} touches, "
                          f"slope {slope_pct:.3f}%/bar, dist {dist:+.2f}% "
                          f"=> QUALIFIES")
                else:
                    print(f"    {side}: no line — blocker: {blocker}")


if __name__ == "__main__":
    # Reference setups the system must always find (validated by Bilal):
    # TSM/XAUT = TradFi corr-skip cases, MAVIA = 5-touch far-from-line case,
    # JELLYJELLY h1 = boundary case: 3 touches, slope 0.054%/bar (just above
    # the 0.04 floor), price at the line (2026-07-05, only valid hit of 196).
    # BTCDOM h1, BZ h4, LIT h4 = added 2026-07-05 with the body-swing rule
    # (HH/LL from open/close extremes, never wicks).
    main(sys.argv[1:] or ["TSM", "XAUT", "MAVIA", "JELLYJELLY",
                          "BTCDOM", "BZ", "LIT"])
