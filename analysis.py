"""
Technical-analysis engine: correlation filter, swing detection,
trend/structure classification, trendline fitting and the
3-candle breakout / invalidation logic.

Mathematical approach (summary):

* Swings   — a bar is a swing high when its BODY TOP (max of open/close)
             is the maximum of a symmetric +/- SWING_ORDER window
             (scipy.signal.argrelextrema); swing lows use the body bottom
             (min of open/close). Bodies, not wicks: a pivot is where a
             candle BODY got rejected by the trend — on a red candle at a
             top that's the OPEN, not the close. Wicks are noise.
             Bars within SWING_ORDER of either array edge lack a full
             confirmation window and are explicitly discarded, so a swing
             can never appear on the newest bars and later vanish
             (non-repainting).
* Trendline— drawn the way a trader draws it: the candidate line passes
             through a PAIR of swing pivots and must REST on the extremes —
             every pivot from the line's first touch onward stays on the
             correct side (above support / below resistance), give or take
             LINE_PIERCE_TOL_PCT of wick noise. Among all valid pair-lines
             the one touching the most pivots (ties: the longest) wins.
             A pivot counts as a "touch" when it sits within
             TOUCH_TOLERANCE_PCT of the line; >= MIN_TOUCHES are required
             and touches do NOT have to be monotonic — a pullback sequence
             like 0.62 / 0.71 / 0.40 on a falling resistance is valid.
* Trend    — simply the direction of the winning line: a rising support
             line (slope > 0) = uptrend, a falling resistance line
             (slope < 0) = downtrend. If both sides yield a valid line the
             one with more touches (ties: the more recent) wins.
             x-coordinates are candle timestamps converted to bar units,
             so the line can be projected onto future candles in later
             scan cycles.
* Breakout — sliding 3-candle sequence against the projected line:
             C1 closes through the line on >= 2x volume MA(20),
             C2 closes through it and is NOT a hammer (support break) /
             NOT a shooting star (resistance break), C3 closes through it.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

import config


# ============================================================================
# 1. BTC correlation filter
# ============================================================================

def get_correlation(coin_closes: pd.Series, btc_closes: pd.Series,
                    window: int) -> float | None:
    """
    Pearson correlation between a coin and BTC over the last `window`
    daily candles. Computed on daily returns when CORR_ON_RETURNS is set
    (recommended), otherwise on raw close prices.
    """
    # Returns lose one row to differencing, so they need window+1 closes;
    # raw-price mode uses exactly `window` closes.
    take = window + 1 if config.CORR_ON_RETURNS else window
    n = min(len(coin_closes), len(btc_closes))
    if n < take:
        return None

    coin = coin_closes.iloc[-take:].reset_index(drop=True)
    btc = btc_closes.iloc[-take:].reset_index(drop=True)

    if config.CORR_ON_RETURNS:
        coin = coin.pct_change().dropna()
        btc = btc.pct_change().dropna()

    if coin.std() == 0 or btc.std() == 0:    # flat series => corr undefined
        return None
    return float(coin.corr(btc))


def passes_correlation_filter(coin_closes: pd.Series,
                              btc_closes: pd.Series) -> tuple[bool, dict]:
    """
    Pass when Pearson correlation with BTC is <= CORR_THRESHOLD for BOTH
    lookback windows (30d and 90d). Missing data => fail (conservative).
    """
    corrs = {}
    for window in config.CORR_WINDOWS:
        corr = get_correlation(coin_closes, btc_closes, window)
        corrs[window] = corr
        if corr is None or corr > config.CORR_THRESHOLD:
            return False, corrs
    return True, corrs


# ============================================================================
# 2. Swing detection & trend structure
# ============================================================================

def find_swings(df: pd.DataFrame,
                order: int = config.SWING_ORDER) -> tuple[list, list]:
    """
    Detect confirmed swing highs/lows with a symmetric rolling-extreme test
    on candle BODY extremes — highs on max(open, close), lows on
    min(open, close). Wicks are noise; a pivot is where a candle body was
    rejected by the trend, and on a red candle at a top the body extreme
    is the OPEN, not the close.

    Returns two lists of dicts {i, ts, price} (bar index, timestamp, price),
    oldest first.

    argrelextrema's default mode='clip' compares edge bars against
    themselves (np.take clips out-of-range indices), and greater_equal /
    less_equal pass that self-comparison — so raw output CAN contain bars
    inside the first/last `order` positions that have no full confirmation
    window. Those are filtered out below; that filter is what makes the
    swings non-repainting.
    """
    body_hi = np.maximum(df["open"].values, df["close"].values)
    body_lo = np.minimum(df["open"].values, df["close"].values)
    n = len(df)

    hi_idx = [i for i in argrelextrema(body_hi, np.greater_equal, order=order)[0]
              if order <= i < n - order]
    lo_idx = [i for i in argrelextrema(body_lo, np.less_equal, order=order)[0]
              if order <= i < n - order]

    # np.greater_equal marks every bar of a flat plateau — keep the first.
    def dedupe(indices: list[int], values: np.ndarray) -> list[int]:
        out = []
        for i in indices:
            if out and i - out[-1] <= order and values[i] == values[out[-1]]:
                continue
            out.append(int(i))
        return out

    swing_highs = [{"i": i, "ts": int(df["ts"].iloc[i]), "price": float(body_hi[i])}
                   for i in dedupe(hi_idx, body_hi)]
    swing_lows = [{"i": i, "ts": int(df["ts"].iloc[i]), "price": float(body_lo[i])}
                  for i in dedupe(lo_idx, body_lo)]
    return swing_highs, swing_lows


# ============================================================================
# 3. Trendline construction & distance check
# ============================================================================

def fit_pivot_line(pivots: list, side: str, anchor_ts: int,
                   tf_ms: int, bars: pd.DataFrame | None = None) -> dict | None:
    """
    Find the best trader-style trendline through the given swing pivots
    (swing LOWS with side='support', swing HIGHS with side='resistance').

    Every pair of pivots defines a candidate line. A candidate is valid when:
      * its slope has the right sign (support rises, resistance falls) AND
        is steeper than MIN_SLOPE_PCT_PER_BAR — a near-horizontal line is a
        level, not a trend,
      * every pivot from the line's FIRST touch onward sits on the correct
        side of it — above support / below resistance — allowing at most
        LINE_PIERCE_TOL_PCT of pierce for wick noise,
      * when `bars` (the full OHLCV frame) is given: every candle CLOSE from
        the first touch onward also respects the line within the same pierce
        tolerance. Pivots alone miss price cutting through the line on bars
        that never confirmed as swings (incl. the last SWING_ORDER bars) —
        a line price has closed through is not a trendline,
      * at least MIN_TOUCHES pivots sit within TOUCH_TOLERANCE_PCT of it.
    Touches need NOT be monotonic: a pullback that undercuts the previous
    one but still respects the line is a perfectly good touch.

    The valid candidate with the most touches wins (ties: the one spanning
    the most bars). x is expressed in BAR UNITS relative to `anchor_ts`
    (x = (ts - anchor_ts) / tf_ms) so later scan cycles can project the
    line onto brand-new candles using timestamps alone.

    Returns {slope, intercept, anchor_ts, touches, n_points} or None.
    """
    pts = pivots[-config.MAX_SWINGS_CONSIDERED:]
    if len(pts) < config.MIN_TOUCHES:
        return None

    sign = 1.0 if side == "support" else -1.0   # +1: pivots must sit above
    xs = [(s["ts"] - anchor_ts) / tf_ms for s in pts]
    ys = [s["price"] for s in pts]

    if bars is not None:
        bar_x = ((bars["ts"] - anchor_ts) / tf_ms).to_numpy(dtype=float)
        bar_close = bars["close"].to_numpy(dtype=float)

    best: dict | None = None
    best_key = (0, 0.0)                          # (touches, span in bars)
    for a in range(len(pts) - 1):
        for b in range(a + 1, len(pts)):
            slope = (ys[b] - ys[a]) / (xs[b] - xs[a])
            if (side == "support") != (slope > 0) or slope == 0:
                continue
            # Near-horizontal = an S/R level, not a trend.
            if abs(slope) / ys[a] * 100.0 < config.MIN_SLOPE_PCT_PER_BAR:
                continue
            intercept = ys[a] - slope * xs[a]

            # The line only exists from its first touch onward — pivots from
            # before the trend began don't get a vote.
            touches, ok = 0, True
            for x, y in zip(xs[a:], ys[a:]):
                line = slope * x + intercept
                if line <= 0:                    # nonsense projection
                    ok = False
                    break
                diff_pct = (y - line) / line * 100.0 * sign
                if diff_pct < -config.LINE_PIERCE_TOL_PCT:
                    ok = False                   # pivot through the line
                    break
                if abs(diff_pct) <= config.TOUCH_TOLERANCE_PCT:
                    touches += 1
            if not ok or touches < config.MIN_TOUCHES:
                continue

            # Candle-close validation: price must never have CLOSED through
            # the line since its first touch. Uses its own (looser) tolerance:
            # a shallow fake-break close is trader noise, a real cut-through
            # goes far past CLOSE_PIERCE_TOL_PCT.
            if bars is not None:
                m = bar_x >= xs[a]
                line_v = slope * bar_x[m] + intercept
                if np.any(line_v <= 0) or np.min(
                        (bar_close[m] - line_v) / line_v * 100.0 * sign
                ) < -config.CLOSE_PIERCE_TOL_PCT:
                    continue

            key = (touches, xs[-1] - xs[a])
            if key > best_key:
                best_key = key
                best = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "anchor_ts": int(anchor_ts),
                    "touches": touches,
                    "n_points": len(pts) - a,
                }
    return best


def trendline_value_at(tl: dict, ts: int, tf_ms: int) -> float:
    """Project the stored trendline onto the candle with timestamp `ts`."""
    x = (ts - tl["anchor_ts"]) / tf_ms
    return tl["slope"] * x + tl["intercept"]


def same_line(a: dict, b: dict, tf_ms: int, tol_pct: float = 0.1) -> bool:
    """
    True if two fitted trendlines are geometrically ~identical, ignoring anchor.

    Each scan refits the line on a fresh window, so anchor_ts/intercept shift even
    when the line hasn't visually moved. Compare anchor-independently: project both
    to a common ts and require the same level AND the same per-bar slope.
    """
    ref = b["anchor_ts"]
    va, vb = trendline_value_at(a, ref, tf_ms), trendline_value_at(b, ref, tf_ms)
    if vb == 0:
        return False
    same_level = abs(va - vb) / abs(vb) * 100 <= tol_pct
    denom = abs(b["slope"]) or 1e-12
    same_slope = abs(a["slope"] - b["slope"]) / denom <= 0.01
    return same_level and same_slope


def distance_to_trendline_pct(close: float, tl_value: float) -> float:
    """Signed distance of close from the line, in % (positive = above)."""
    if tl_value == 0:
        return float("inf")
    return (close - tl_value) / abs(tl_value) * 100.0


# ============================================================================
# 4. Candlestick patterns
# ============================================================================

def is_hammer(candle: pd.Series) -> bool:
    """
    Hammer: small body in the upper part of the range with a long lower
    wick — buyers rejected the breakdown. Geometry thresholds in config.
    """
    rng = candle["high"] - candle["low"]
    if rng <= 0:
        return False
    body = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return (body <= config.HAMMER_BODY_MAX_PCT * rng
            and lower_wick >= config.HAMMER_WICK_BODY_MULT * max(body, rng * 0.05)
            and upper_wick <= lower_wick * 0.5)


def is_shooting_star(candle: pd.Series) -> bool:
    """Inverse of the hammer — long upper wick, body near the low."""
    rng = candle["high"] - candle["low"]
    if rng <= 0:
        return False
    body = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return (body <= config.HAMMER_BODY_MAX_PCT * rng
            and upper_wick >= config.HAMMER_WICK_BODY_MULT * max(body, rng * 0.05)
            and lower_wick <= upper_wick * 0.5)


# ============================================================================
# 5. Breakout / invalidation
# ============================================================================

def check_breakout(df: pd.DataFrame, tl: dict, direction: str,
                   tf_ms: int, since_ts: int = 0) -> tuple[bool, str]:
    """
    Scan every consecutive 3-candle window (closed candles only, newest
    window last) for the invalidation sequence.

    Uptrend (support break — all closes BELOW the projected line):
      C1: close < TL  AND  volume >= VOL_BREAK_MULT x VolMA(VOL_MA_LEN)
      C2: close < TL  AND  NOT a hammer
      C3: close < TL
    Downtrend (resistance break) is the mirror image: closes ABOVE the
    line and C2 must NOT be a shooting star.

    Every 3-candle window in the fetched history is scanned (bounded only
    by volume-MA warm-up and `since_ts`) — not just the last 3 candles — so
    a sequence that completed between hourly scan cycles (e.g. on the 15m
    timeframe) or during screener downtime is still caught.

    `since_ts` is the OPEN timestamp of the last closed candle at the time
    the trendline was (re)fitted: only candles strictly AFTER that one can
    start a breakout, so a refreshed line is never back-projected onto
    candles that predate its own fit.

    Returns (broken, reason).
    """
    if len(df) < config.VOL_MA_LEN + 3:
        return False, ""

    vol_ma = df["volume"].rolling(config.VOL_MA_LEN).mean()
    tl_vals = df["ts"].apply(lambda ts: trendline_value_at(tl, int(ts), tf_ms))

    if direction == "up":
        beyond = df["close"] < tl_vals          # closed below support
        pattern_exception = is_hammer
    else:
        beyond = df["close"] > tl_vals          # closed above resistance
        pattern_exception = is_shooting_star

    for i in range(config.VOL_MA_LEN, len(df) - 2):
        c1, c2, c3 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2]
        if c1["ts"] <= since_ts:
            continue

        ma = vol_ma.iloc[i]
        cond1 = beyond.iloc[i] and pd.notna(ma) and \
            c1["volume"] >= config.VOL_BREAK_MULT * ma
        cond2 = beyond.iloc[i + 1] and not pattern_exception(c2)
        cond3 = beyond.iloc[i + 2]

        if cond1 and cond2 and cond3:
            side = "below support" if direction == "up" else "above resistance"
            reason = (f"3-candle breakout {side} "
                      f"(C1 vol {c1['volume']:.0f} >= "
                      f"{config.VOL_BREAK_MULT:.0f}x MA {ma:.0f})")
            return True, reason

    return False, ""


# ============================================================================
# 6. Full per-timeframe qualification
# ============================================================================

def evaluate_symbol_timeframe(df: pd.DataFrame,
                              tf_ms: int) -> dict | None:
    """
    Run the full structure pipeline on one symbol/timeframe:
      swings -> best pivot-resting trendline (>= MIN_TOUCHES touches)
      -> 0.5% distance check on the latest close.

    A rising support line = uptrend, a falling resistance line = downtrend;
    when both sides produce a valid line the one with more touches wins.

    Returns a watchlist-ready dict or None if any rule fails.
    """
    if df is None or len(df) < config.SWING_ORDER * 2 + 10:
        return None

    # Stablecoins (USDC…) move so little that every pivot sits within the
    # %-based touch tolerance — reject flat markets before "finding" lines.
    lo = float(df["low"].min())
    if lo <= 0 or (float(df["high"].max()) - lo) / lo * 100.0 < config.MIN_RANGE_PCT:
        return None

    swing_highs, swing_lows = find_swings(df)
    anchor_ts = int(df["ts"].iloc[0])
    support = fit_pivot_line(swing_lows, "support", anchor_ts, tf_ms, bars=df)
    resistance = fit_pivot_line(swing_highs, "resistance", anchor_ts, tf_ms,
                                bars=df)

    candidates = [(d, tl) for d, tl in (("up", support), ("down", resistance))
                  if tl is not None]
    if not candidates:
        return None
    direction, tl = max(candidates, key=lambda c: c[1]["touches"])

    # No max-distance gate: a valid line far from price is still a setup
    # worth tracking — the table sorts by |distance| so near ones float up.
    last = df.iloc[-1]
    tl_now = trendline_value_at(tl, int(last["ts"]), tf_ms)
    dist_pct = distance_to_trendline_pct(float(last["close"]), tl_now)

    return {
        "direction": direction,
        "trendline": tl,
        "distance_pct": round(dist_pct, 3),
        "close": float(last["close"]),
        "last_candle_ts": int(last["ts"]),
    }
