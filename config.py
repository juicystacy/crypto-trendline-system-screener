"""
Central configuration for the Binance Futures crypto screener.

All tunable parameters of the trading logic live here so the strategy
can be adjusted without touching the engine code.
"""

# ----------------------------------------------------------------------------
# Exchange / data
# ----------------------------------------------------------------------------
QUOTE_ASSET = "USDT"                 # only USDT-margined perpetual futures
BTC_SYMBOL = "BTC/USDT:USDT"         # ccxt unified symbol for BTCUSDT perp

# Timeframes scanned independently. Keys are ccxt timeframe strings,
# values are the labels shown in the terminal output.
TIMEFRAMES = {
    "1h":  "h1",
    "4h":  "h4",
    "1d":  "d1",
}

# Milliseconds per timeframe — used to project the trendline forward in time
# (Binance perps trade 24/7 so candle timestamps are perfectly regular).
TIMEFRAME_MS = {
    "15m": 15 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}

CANDLE_LIMIT = 300                   # candles fetched per symbol/timeframe
DAILY_CANDLE_LIMIT = 120             # daily candles for the correlation filter

# ----------------------------------------------------------------------------
# BTC correlation filter
# ----------------------------------------------------------------------------
CORR_WINDOWS = (30, 90)              # lookbacks in days
CORR_THRESHOLD = 0.5                 # pass if corr <= 0.5 for BOTH windows

# Pearson correlation on raw close prices is statistically unreliable
# (two trending series always look correlated). Computing it on daily
# RETURNS is the standard quant approach and is the default here.
# Set to False to reproduce the literal "close prices" spec.
CORR_ON_RETURNS = True

# ----------------------------------------------------------------------------
# Swing / trend structure
# ----------------------------------------------------------------------------
SWING_ORDER = 5                      # bar must be the extreme of +/- N bars
MAX_SWINGS_CONSIDERED = 12           # recent swing pivots handed to the fitter
                                     # (it searches them for the best sub-line)
# Stablecoin/dead-market guard: a symbol whose ENTIRE fetched history spans
# less than this much high-to-low is flat noise (USDC & friends) — any
# "trendline" found there is an artifact of the %-based touch tolerance.
MIN_RANGE_PCT = 1.0

# ----------------------------------------------------------------------------
# Trendline — drawn the way a trader draws it: the line RESTS ON the extreme
# pivots (under the swing lows for support, over the swing highs for
# resistance) and every later pivot must stay on the correct side of it.
# NOT a least-squares fit through the middle of the swings.
# A valid trend = such a line with >= MIN_TOUCHES touches + correct slope sign.
# ----------------------------------------------------------------------------
MIN_TOUCHES = 3                      # pivots hugging the line to confirm it
TOUCH_TOLERANCE_PCT = 0.5            # a pivot within 0.5% of the line = a touch
# How far a pivot may poke through the WRONG side of the line before the line
# is rejected (0 = perfectly strict). Calibrated on reference setups
# 2026-07-05: BTCDOM h1 & BZ h4 both have a body pivot ~0.3-0.4% through
# lines Bilal calls valid — 0.15 rejected them.
LINE_PIERCE_TOL_PCT = 0.4
# Same idea but for the candle-CLOSE validation pass: how far a CLOSE may
# poke through the line without killing it. Looser than the pivot tolerance —
# a lone fake-break close that price immediately reclaimed is trader-tolerable
# noise, while a real cut-through goes far past this.
CLOSE_PIERCE_TOL_PCT = 0.75
# A near-horizontal line is a support/resistance LEVEL, not a trend. Reject
# candidates flatter than this (% of price per bar). Validated examples:
# OPEN h1 0.056, ARC h4 0.079 %/bar; rejected flat junk: USDC d1 0.0001,
# SUN h1 0.001, FLOW h1 0.019.
MIN_SLOPE_PCT_PER_BAR = 0.04

# ----------------------------------------------------------------------------
# Breakout / invalidation
# ----------------------------------------------------------------------------
VOL_MA_LEN = 20                      # volume moving-average length
VOL_BREAK_MULT = 2.0                 # candle-1 volume >= 2x volume MA

# Hammer geometry (candle-2 exception on support breaks).
HAMMER_BODY_MAX_PCT = 0.35           # body <= 35% of full candle range
HAMMER_WICK_BODY_MULT = 2.0          # dominant wick >= 2x body

# ----------------------------------------------------------------------------
# LLM second-pass filter — Claude vision judges each qualified chart like a
# trader; verdict shown in the AI column. Needs ANTHROPIC_API_KEY in the env.
# ----------------------------------------------------------------------------
LLM_FILTER = True                    # on — Sonnet judge separates clean lines from
                                     # geometry-passing junk (which thresholds can't)
LLM_MODEL = "claude-sonnet-5"        # ~5x cheaper than Opus; was off for cost
LLM_CHART_CANDLES = 160              # candles drawn on the chart image

# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
SCAN_INTERVAL_SEC = 3600             # 1-hour scan cycle
WATCHLIST_FILE = "watchlist.json"    # persistent state between cycles
HISTORY_FILE = "history.csv"         # persistent log of LLM-passed setups
HISTORY_KEEP_DAYS = 180              # ~6 months; older rows dropped on each write
MAX_API_RETRIES = 3
RETRY_BACKOFF_SEC = 5
