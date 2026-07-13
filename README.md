# Binance Futures Crypto Screener

Scans all Binance USDT-M perpetual futures every hour, keeps a persistent
watchlist of coins that are **uncorrelated with BTC** and sitting on a valid
trendline, and removes them when a confirmed 3-candle breakout invalidates the
setup. A second pass has Claude judge each qualified chart like a trader.

## Logic pipeline

1. **BTC-correlation filter** — Pearson correlation with BTCUSDT over the
   last 30 **and** 90 daily candles must both be `<= 0.5`. By default the
   correlation is computed on daily *returns* (the statistically sound
   approach — raw-price correlation between two trending assets is almost
   always spuriously high). Set `CORR_ON_RETURNS = False` in `config.py` to
   use raw close prices instead. TradFi underlyings (stocks/gold/commodities)
   skip this filter — BTC correlation is meaningless for them.
2. **Swings** — a bar is a swing high when its candle **body top**
   (`max(open, close)`) is the extreme of a symmetric ±5-bar window, swing
   lows on the body bottom. Bodies, not wicks: a pivot is where a candle body
   got rejected. Bars within 5 of either data edge lack a full confirmation
   window and are discarded, so a swing can never appear on the newest bars
   and later vanish (non-repainting).
3. **Trendline + trend** — drawn the way a trader draws it: the line **rests
   on** the extreme pivots (under the swing lows for support, over the swing
   highs for resistance), it is **not** a least-squares fit through the
   middle. Every pivot pair defines a candidate; a candidate is valid when
   - its slope has the right sign (support rises, resistance falls) and is
     steeper than `MIN_SLOPE_PCT_PER_BAR` (a near-flat line is an S/R *level*,
     not a trend),
   - every pivot from its first touch onward stays on the correct side,
     allowing `LINE_PIERCE_TOL_PCT` of wick noise,
   - no candle **close** has cut through it (looser `CLOSE_PIERCE_TOL_PCT`),
   - at least `MIN_TOUCHES` (3) pivots sit within `TOUCH_TOLERANCE_PCT`
     (0.5%) of it. Touches need **not** be monotonic.

   The valid line with the most touches wins (ties: longest span). Its
   direction *is* the trend — rising support = uptrend, falling resistance =
   downtrend. Stablecoins / dead markets (whole history spans less than
   `MIN_RANGE_PCT`) are rejected before any line is fitted.
4. **Distance** — the latest close's signed distance to the projected line is
   recorded for display; there is **no** max-distance gate. A valid line far
   from price is still tracked, and the table sorts by |distance| so near
   setups float to the top.
5. **LLM second pass** — each qualified chart is rendered as a candlestick
   image and Claude vision judges the trendline like a trader on TradingView
   (clean stepping pullbacks, visible slope, no closes cutting through).
   Geometry gives recall; this gives precision. The verdict shows in the
   **AI** column (✓ / ✗). Only entries whose line was (re)fitted since their
   last verdict are re-judged, so a scan costs a few cents. Needs
   `ANTHROPIC_API_KEY`; disable with `--no-llm` or `LLM_FILTER = False`.
6. **Invalidation** — a coin is removed when this 3-candle sequence **closes**
   through the line (below support / above resistance):
   - **C1** closes through on volume `>= 2× Volume-MA(20)`
   - **C2** closes through and is **not** a hammer (shooting star for
     downtrends)
   - **C3** closes through

   Every 3-candle window in the fetched history is scanned (not just the last
   3), so a sequence that completed between scan cycles or during downtime is
   still caught.

The watchlist is persisted to `watchlist.json` between cycles and restarts.
Trendlines are stored as `slope/intercept` anchored to a candle timestamp, so
they are projected forward onto new candles in later cycles.

## History log

Every setup that **passes the LLM filter** is recorded to `history.csv` — one
row per setup lifecycle. A row is written when the LLM first validates the setup
(with the LLM-pass date and the watchlist columns: coin, timeframe, trend,
distance, touches, correlations, close, LLM reason) and completed with the
**breakout date** and outcome when the setup later breaks out and leaves the
watchlist. Only LLM-passed setups are logged, so the file stays small.

Rows are deduped per `coin|timeframe` (one open row per pair) and auto-pruned:
anything older than `HISTORY_KEEP_DAYS` (180 days ≈ 6 months) is dropped on every
write. CSV, so it opens directly in Excel. Dates are screener events (detected on
the next scan cycle), not exact candle times.

## Install

```bash
pip install -r requirements.txt
```

(The LLM second pass runs by default; use `--no-llm` to skip it.)

> `pandas_ta` is intentionally **not** used — it is unmaintained and breaks
> on modern numpy/Python. The only indicators needed (volume SMA, candle
> patterns) are implemented directly in pandas/numpy.

## Run

```bash
python screener.py                 # loop forever, scan every hour
python screener.py --once          # single scan cycle, then exit
python screener.py --once --max-symbols 30   # quick smoke test
python screener.py --interval 1800 # custom cycle length (seconds)
python screener.py --no-llm        # skip the Claude chart filter
```

No exchange API keys are required — only public market-data endpoints are
used. The LLM pass needs `ANTHROPIC_API_KEY` in the environment.

### Network access

Binance API endpoints are unreachable from some countries — either Binance
geo-blocks the region (HTTP 451, e.g. the US) or the local ISP blocks the
exchange domains (connection/SSL errors during the handshake, common in
Indonesia). The screener survives this gracefully (it retries and waits for
the next cycle), but it can't screen anything without data. Options:

- run it on a VPS in a supported region, or
- route it through a VPN/proxy. `ccxt` honors the standard proxy
  environment variable, so no code change is needed:

  ```powershell
  $env:HTTPS_PROXY = "http://127.0.0.1:7890"   # your proxy address
  python screener.py
  ```

A full scan of ~300 symbols takes several minutes because Binance rate
limits are respected (`enableRateLimit` in ccxt).

## Files

| File | Purpose |
|---|---|
| `config.py` | every tunable parameter of the strategy |
| `exchange.py` | ccxt data access with retry/rate-limit handling |
| `analysis.py` | swings, trendline, patterns, breakout logic |
| `llm_filter.py` | renders charts and asks Claude vision for a verdict |
| `history.py` | appends LLM-passed setups to the history CSV, auto-prunes |
| `watchlist.py` | JSON-persisted watchlist state |
| `screener.py` | scan cycle orchestration, rich output, main loop |
| `selftest.py` | offline test suite (no network needed) |
| `diagnose.py` | one-off connectivity / data debugging helper |
| `watchlist.json` | generated at runtime — current watchlist state |
| `history.csv` | generated at runtime — log of LLM-passed setups (auto-pruned to 180 days) |

## Tuning

All thresholds live in `config.py`: swing sensitivity (`SWING_ORDER`),
required touches (`MIN_TOUCHES`), the 0.5% touch tolerance, pivot/close pierce
tolerances, minimum slope, stablecoin range guard, volume-spike multiple,
hammer geometry, the LLM model, the scan interval, and how long history rows are
kept (`HISTORY_KEEP_DAYS`).
