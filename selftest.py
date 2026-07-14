"""
Offline self-test for the screener's analysis engine.

Builds synthetic OHLCV data with a known structure (a clean uptrend whose
Higher Lows sit exactly on a straight support line, plus its mirrored
downtrend) and asserts that every stage of the pipeline behaves:

  * swing detection is non-repainting (no swing inside the last SWING_ORDER
    bars) and finds exactly the constructed pivots,
  * the pivot-resting trendline fit finds the constructed support line,
    counts its touches, rejects 2-touch lines and pierced lines, and
    accepts non-monotonic pullback sequences,
  * the 0.5% distance check passes near the line and fails away from it,
  * hammer / shooting-star geometry is recognized,
  * the 3-candle breakout fires on a high-volume break with a non-hammer
    second candle, and does NOT fire when candle 2 is a hammer,
  * the correlation filter rejects a clone of BTC and accepts an
    anti-correlated series.

Run:  python selftest.py     (exit code 0 = all good, no network required)
"""

import numpy as np
import pandas as pd

import analysis
import config

TF_MS = config.TIMEFRAME_MS["1h"]
T0 = 1_700_000_000_000          # arbitrary epoch-ms anchor for candle opens

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


# ----------------------------------------------------------------------------
# Synthetic data builders
# ----------------------------------------------------------------------------

def zigzag_path(n_bars: int = 91) -> np.ndarray:
    """
    Piecewise-linear close path: swing lows at bars 10/26/42/58/74 exactly on
    the line 100 + 0.05*bar, swing highs at 18/34/50/66/82 three points above
    it, ending near the support line at the final bar.
    """
    pivots = [(0, 101.0)]
    for k in range(5):
        lo_bar = 10 + 16 * k
        hi_bar = 18 + 16 * k
        pivots.append((lo_bar, 100 + 0.05 * lo_bar))          # on the line
        pivots.append((hi_bar, 100 + 0.05 * hi_bar + 3.0))    # above it
    pivots.append((n_bars - 1, 100 + 0.05 * (n_bars - 1) + 0.03))  # back to line

    path = np.empty(n_bars)
    for (b0, p0), (b1, p1) in zip(pivots, pivots[1:]):
        path[b0:b1 + 1] = np.linspace(p0, p1, b1 - b0 + 1)
    return path


def candles_from_path(path: np.ndarray, volume: float = 1000.0) -> pd.DataFrame:
    """open = previous close, small symmetric wicks, constant volume."""
    closes = path
    opens = np.concatenate([[path[0]], path[:-1]])
    rows = []
    for i, (o, c) in enumerate(zip(opens, closes)):
        rows.append({
            "ts": T0 + i * TF_MS,
            "open": o, "close": c,
            "high": max(o, c) + 0.02, "low": min(o, c) - 0.02,
            "volume": volume,
        })
    return pd.DataFrame(rows)


def append_candle(df: pd.DataFrame, close: float, volume: float,
                  open_: float | None = None, high: float | None = None,
                  low: float | None = None) -> pd.DataFrame:
    prev_close = float(df["close"].iloc[-1])
    o = prev_close if open_ is None else open_
    h = max(o, close) + 0.02 if high is None else high
    l = min(o, close) - 0.02 if low is None else low
    row = {"ts": int(df["ts"].iloc[-1]) + TF_MS, "open": o, "close": close,
           "high": h, "low": l, "volume": volume}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_swings_non_repainting() -> None:
    print("\n[1] Swing detection is non-repainting")
    # Monotonically rising highs: the last bar is the global maximum. A
    # repainting detector would flag it as a swing high even though it has
    # no right-hand confirmation bars yet.
    n = 40
    df = candles_from_path(np.linspace(100, 120, n))
    swing_highs, swing_lows = analysis.find_swings(df)
    limit = len(df) - config.SWING_ORDER
    check("no swing high inside the unconfirmed tail",
          all(s["i"] < limit for s in swing_highs),
          f"got indices {[s['i'] for s in swing_highs]} (limit {limit})")
    check("no swing low inside the unconfirmed tail",
          all(s["i"] < limit for s in swing_lows),
          f"got indices {[s['i'] for s in swing_lows]} (limit {limit})")


def test_uptrend_pipeline() -> pd.DataFrame:
    print("\n[2] Uptrend structure / trendline / distance")
    df = candles_from_path(zigzag_path())

    swing_highs, swing_lows = analysis.find_swings(df)
    lo_bars = [s["i"] for s in swing_lows]
    hi_bars = [s["i"] for s in swing_highs]
    check("finds the 5 constructed swing lows", lo_bars == [10, 26, 42, 58, 74],
          f"got {lo_bars}")
    check("finds the 5 constructed swing highs", hi_bars == [18, 34, 50, 66, 82],
          f"got {hi_bars}")

    def piv(bar: int, price: float) -> dict:
        return {"i": bar, "ts": T0 + bar * TF_MS, "price": price}

    anchor = int(df["ts"].iloc[0])
    tl_sup = analysis.fit_pivot_line(swing_lows, "support", anchor, TF_MS)
    check("support line rests on all 5 collinear lows (5 touches)",
          tl_sup is not None and tl_sup["touches"] == 5,
          f"got {tl_sup and tl_sup['touches']}")
    check("rising swing highs yield NO (falling) resistance line",
          analysis.fit_pivot_line(swing_highs, "resistance", anchor, TF_MS)
          is None)

    # Only 2 pivots near any rising line: below MIN_TOUCHES, no trendline.
    two_touch = [piv(0, 100.0), piv(16, 110.0), piv(32, 103.2)]
    check("2 touches are NOT enough for a trendline",
          analysis.fit_pivot_line(two_touch, "support", T0, TF_MS) is None)

    # Syrup-style pullbacks: 100 -> 97 -> 98 -> 96 is NOT monotonic, but all
    # pivots respect the falling line through 100/98/96 — must be valid.
    syrup = [piv(0, 100.0), piv(10, 97.0), piv(20, 98.0), piv(40, 96.0)]
    tl_nm = analysis.fit_pivot_line(syrup, "resistance", T0, TF_MS)
    check("non-monotonic pullbacks resting on a falling line are valid",
          tl_nm is not None and tl_nm["touches"] == 3,
          f"got {tl_nm and tl_nm['touches']}")

    # A pivot slicing >LINE_PIERCE_TOL_PCT through the line rejects it.
    pierced = [piv(0, 100.0), piv(16, 100.8), piv(24, 99.9), piv(32, 101.6)]
    check("a pivot piercing through the line rejects it",
          analysis.fit_pivot_line(pierced, "support", T0, TF_MS) is None)

    # Near-horizontal collinear pivots: a LEVEL, not a trend — rejected.
    flat_line = [piv(0, 100.0), piv(20, 100.02), piv(40, 100.04)]
    check("near-horizontal line (< MIN_SLOPE_PCT_PER_BAR) is rejected",
          analysis.fit_pivot_line(flat_line, "support", T0, TF_MS) is None)

    # Candle-close validation: same valid syrup line, but a NON-pivot bar
    # closes ~1% through it — beyond CLOSE_PIERCE_TOL_PCT (0.75, looser than
    # the pivot pierce tol) -> with `bars` given the line must be rejected.
    bars_ok = pd.DataFrame({
        "ts": [T0 + i * TF_MS for i in range(41)],
        "close": [100 - 0.1 * i - 1.0 for i in range(41)],   # safely below
    })
    bars_cut = bars_ok.copy()
    bars_cut.loc[30, "close"] = 98.0             # line@30 = 97 -> +1.03% above
    check("syrup line still valid when closes respect it",
          analysis.fit_pivot_line(syrup, "resistance", T0, TF_MS,
                                  bars=bars_ok) is not None)
    check("a candle CLOSE through the line rejects it",
          analysis.fit_pivot_line(syrup, "resistance", T0, TF_MS,
                                  bars=bars_cut) is None)

    result = analysis.evaluate_symbol_timeframe(df, TF_MS)
    check("full evaluation qualifies the symbol as uptrend",
          result is not None and result["direction"] == "up",
          f"got {result and result['direction']!r}")
    if result:
        tl = result["trendline"]
        check("trendline slope is positive (rising support)", tl["slope"] > 0)
        check(f"at least {config.MIN_TOUCHES} touches",
              tl["touches"] >= config.MIN_TOUCHES, f"got {tl['touches']}")
        check("distance to the line is reported",
              abs(result["distance_pct"]) <= 0.5,
              f"got {result['distance_pct']}%")

        # same_line: the SAME geometric line refit on a window shifted forward
        # k bars (new anchor_ts, adjusted intercept) must compare equal — this
        # is what stops the LLM from re-judging unchanged lines every scan.
        k = 3
        shifted = {**tl, "anchor_ts": tl["anchor_ts"] + k * TF_MS,
                   "intercept": tl["slope"] * k + tl["intercept"]}
        check("same_line: identical line with shifted anchor matches",
              analysis.same_line(tl, shifted, TF_MS))
        different = {**tl, "slope": tl["slope"] * 1.5}
        check("same_line: a clearly different slope does NOT match",
              not analysis.same_line(tl, different, TF_MS))

    # Last close pushed 2% above the line: still a valid line — distance
    # no longer gates qualification, it only sorts the table.
    far = zigzag_path()
    far[-1] *= 1.02
    far_res = analysis.evaluate_symbol_timeframe(candles_from_path(far), TF_MS)
    check("2% away from the line still qualifies (distance only sorts)",
          far_res is not None and far_res["distance_pct"] > 1.5,
          f"got {far_res and far_res['distance_pct']}%")

    # Same shape squeezed to a ~0.4% total range (a stablecoin chart):
    # collinear pivots and all, it must be rejected by the range guard.
    flat = 100 + (zigzag_path() - 100) * 0.05
    check("stablecoin-flat series (range < MIN_RANGE_PCT) is rejected",
          analysis.evaluate_symbol_timeframe(candles_from_path(flat), TF_MS)
          is None)
    return df


def test_downtrend_mirror() -> None:
    print("\n[3] Downtrend mirror")
    df = candles_from_path(210.0 - zigzag_path())
    result = analysis.evaluate_symbol_timeframe(df, TF_MS)
    check("mirrored path qualifies as downtrend",
          result is not None and result["direction"] == "down",
          f"got {result and result['direction']!r}")
    if result:
        check("resistance slope is negative", result["trendline"]["slope"] < 0)


def test_patterns() -> None:
    print("\n[4] Candlestick patterns")
    hammer = pd.Series({"open": 100.0, "close": 99.9, "high": 100.05,
                        "low": 98.5, "volume": 1})
    star = pd.Series({"open": 99.9, "close": 100.0, "high": 101.5,
                      "low": 99.85, "volume": 1})
    bear = pd.Series({"open": 100.0, "close": 99.0, "high": 100.02,
                      "low": 98.98, "volume": 1})
    check("long-lower-wick candle is a hammer", analysis.is_hammer(hammer))
    check("long-upper-wick candle is a shooting star",
          analysis.is_shooting_star(star))
    check("full-body bear candle is NOT a hammer",
          not analysis.is_hammer(bear))
    check("zero-range candle handled",
          not analysis.is_hammer(pd.Series({"open": 1.0, "close": 1.0,
                                            "high": 1.0, "low": 1.0})))


def test_breakout(df: pd.DataFrame) -> None:
    print("\n[5] 3-candle breakout / invalidation")
    result = analysis.evaluate_symbol_timeframe(df, TF_MS)
    if result is None:
        check("breakout test has a valid entry to work with", False)
        return
    tl = result["trendline"]

    def line_at(frame: pd.DataFrame) -> float:
        next_ts = int(frame["ts"].iloc[-1]) + TF_MS
        return analysis.trendline_value_at(tl, next_ts, TF_MS)

    # C1: closes 1% below support on 5x average volume.
    b = append_candle(df, close=line_at(df) * 0.99, volume=5000)
    # C2: plain bearish body (NOT a hammer).
    c2_close = float(b["close"].iloc[-1]) * 0.995
    b2 = append_candle(b, close=c2_close, volume=1000)
    # C3: still below the line.
    b3 = append_candle(b2, close=c2_close * 0.999, volume=1000)

    broken, reason = analysis.check_breakout(b3, tl, "up", TF_MS, since_ts=0)
    check("high-volume 3-candle break below support fires", broken, reason)

    # Same sequence but C2 is a textbook hammer: candle-2 exception blocks it.
    c1_close = float(b["close"].iloc[-1])
    hb = append_candle(b, close=c1_close - 0.05, volume=1000,
                       open_=c1_close, high=c1_close + 0.05,
                       low=c1_close - 2.0)
    hb3 = append_candle(hb, close=float(hb["close"].iloc[-1]) - 0.02,
                        volume=1000)
    broken_h, _ = analysis.check_breakout(hb3, tl, "up", TF_MS, since_ts=0)
    check("hammer on candle 2 blocks the invalidation", not broken_h)

    # No volume spike on C1: sequence must not fire either.
    nv = append_candle(df, close=line_at(df) * 0.99, volume=1000)
    nv2 = append_candle(nv, close=float(nv["close"].iloc[-1]) * 0.995,
                        volume=1000)
    nv3 = append_candle(nv2, close=float(nv2["close"].iloc[-1]) * 0.999,
                        volume=1000)
    broken_nv, _ = analysis.check_breakout(nv3, tl, "up", TF_MS, since_ts=0)
    check("no volume spike on candle 1 blocks the invalidation", not broken_nv)

    # Shallow fake-break: closes ~0.3% below the line — within
    # CLOSE_PIERCE_TOL_PCT, the same noise the line fitter tolerates.
    # Must NOT fire, or prune removes an entry the next scan re-adds.
    sh = append_candle(df, close=line_at(df) * 0.997, volume=5000)
    sh2 = append_candle(sh, close=line_at(sh) * 0.997, volume=1000)
    sh3 = append_candle(sh2, close=line_at(sh2) * 0.997, volume=1000)
    broken_sh, _ = analysis.check_breakout(sh3, tl, "up", TF_MS, since_ts=0)
    check("shallow break within CLOSE_PIERCE_TOL_PCT does NOT fire", not broken_sh)


def test_llm_chart_render() -> None:
    print("\n[6] LLM filter chart rendering (offline)")
    import llm_filter
    df = candles_from_path(zigzag_path())
    result = analysis.evaluate_symbol_timeframe(df, TF_MS)
    entry = {"symbol": "TEST/USDT:USDT", "timeframe": "1h",
             "direction": result["direction"], "trendline": result["trendline"]}
    png = llm_filter.render_chart(df, entry, TF_MS)
    check("chart renders to a PNG", png[:8] == b"\x89PNG\r\n\x1a\n",
          f"got {png[:8]!r}")


def test_correlation() -> None:
    print("\n[7] BTC correlation filter")
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.02, 120)
    btc = pd.Series(100 * np.cumprod(1 + rets))
    clone = pd.Series(50 * np.cumprod(1 + rets))          # corr = +1
    inverse = pd.Series(50 * np.cumprod(1 - rets))        # corr ~ -1

    ok_clone, corrs_clone = analysis.passes_correlation_filter(clone, btc)
    ok_inv, corrs_inv = analysis.passes_correlation_filter(inverse, btc)
    check("BTC clone (corr ~ +1) is rejected", not ok_clone,
          f"corrs {corrs_clone}")
    check("anti-correlated series is accepted", ok_inv, f"corrs {corrs_inv}")

    short = pd.Series(100 * np.cumprod(1 + rets[:40]))    # < 90 days history
    ok_short, _ = analysis.passes_correlation_filter(short, btc)
    check("insufficient history (<90d) is rejected", not ok_short)


if __name__ == "__main__":
    test_swings_non_repainting()
    df = test_uptrend_pipeline()
    test_downtrend_mirror()
    test_patterns()
    test_breakout(df)
    test_llm_chart_render()
    test_correlation()

    print(f"\n{'=' * 50}\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
