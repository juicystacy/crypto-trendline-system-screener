"""
LLM second-pass filter: render each qualified trendline as a candlestick
chart image and have Claude judge it the way the trader judges a chart on
TradingView. Geometry rules give recall; this gives precision.

Only entries whose trendline was (re)fitted since their last judgment are
sent — a scan typically costs a few cents. Requires ANTHROPIC_API_KEY.
"""

import base64
import io
import json
import logging

import matplotlib
matplotlib.use("Agg")                      # headless — never open a window
import matplotlib.pyplot as plt

import analysis
import config
import history

log = logging.getLogger("screener")

PROMPT = """You are a discretionary crypto trader judging a trendline drawn on a candlestick chart.

The blue line is a {side} trendline for a claimed {direction}trend. Judge it like a trader eyeballing a chart, using these rules:
- Pivots are candle CLOSES, not wicks: the line must rest on the pullback closes (>= 3 rejections touching the line). Wicks poking through the line are acceptable noise.
- Each pullback close touching the line must ALSO be a Higher Low (uptrend) / Lower High (downtrend) versus the previous pullback — the line connects a strictly {direction}ward-stepping sequence of pullback closes, not arbitrary touches that happen to line up.
- After each pullback, the following swing should make a new Higher High (uptrend) / Lower Low (downtrend). One swing failing to make a new extreme is acceptable; more than one means the trend is stalling and the setup is invalid.
- A new Higher High / Lower Low counts ONLY when it comes from a consistent stepping swing. A single spike candle that juts far out and immediately reverts does NOT count as a valid extreme — discount one-off spikes and judge the consistent structure. Be strict here: an otherwise flat/ranging chart with one spike is NOT a trend.
- The line must NOT be cut through by candle closes anywhere along its length. Reject if closes pierce it repeatedly (a mid-channel regression line that price closes above/below many times is not a trendline), even if the overall drift matches the claimed direction.
- It must have a clear visible slope — a near-horizontal line is a support/resistance level, not a trendline.
- The line must span the trend: touches only clustered at the right edge (line detached from all earlier structure) are invalid.
- Do NOT judge whether price is currently near the line — a valid line that price has moved away from is still a valid line; distance is tracked separately.

Answer strictly as JSON."""

SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "valid": {"type": "boolean"},
            "reason": {"type": "string", "description": "one short sentence"},
        },
        "required": ["valid", "reason"],
        "additionalProperties": False,
    },
}


def render_chart(df, entry, tf_ms: int) -> bytes:
    """Candlestick chart (last LLM_CHART_CANDLES bars) + projected trendline -> PNG bytes."""
    d = df.tail(config.LLM_CHART_CANDLES).reset_index(drop=True)
    tl = entry["trendline"]
    line = [analysis.trendline_value_at(tl, int(ts), tf_ms) for ts in d["ts"]]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    up = d["close"] >= d["open"]
    ax.vlines(d.index, d["low"], d["high"],
              color=["green" if u else "red" for u in up], linewidth=0.7)
    ax.bar(d.index, (d["close"] - d["open"]).abs(),
           bottom=d[["open", "close"]].min(axis=1), width=0.7,
           color=["green" if u else "red" for u in up])
    ax.plot(d.index, line, color="blue", linewidth=1.5)
    ax.set_title(f"{entry['symbol']} {entry['timeframe']} — claimed {entry['direction']}trend")
    ax.margins(x=0.01)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def judge_entry(client, df, entry, tf_ms: int) -> dict | None:
    """Ask Claude for a verdict on one chart. Returns {valid, reason} or None on failure."""
    png = render_chart(df, entry, tf_ms)
    side = "support" if entry["direction"] == "up" else "resistance"
    prompt = PROMPT.format(side=side, direction=entry["direction"])
    # Sonnet 5 runs adaptive thinking by default when `thinking` is omitted, and
    # those tokens share the max_tokens budget — a chart judgment can think well
    # past 1024 and truncate the JSON. Keep thinking (it's what makes the visual
    # call good) but cap it at low effort and give the budget real headroom.
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": SCHEMA},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png",
                            "data": base64.standard_b64encode(png).decode()}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def apply_filter(watchlist: dict, cache: dict, fetch) -> None:
    """
    Judge every watchlist entry whose trendline was (re)fitted since its last
    verdict. Verdicts are stored on the entry (entry["llm"]) and shown in the
    table — entries are never removed here, the trader decides.

    `fetch(symbol, tf)` returns the cached OHLCV df for the cycle.
    """
    import anthropic

    # Judge only when the LINE ITSELF changed. tl_fit_ts refreshes every
    # cycle a symbol re-qualifies (even with an identical line), so keying
    # on it re-judged the whole watchlist every scan — that's what made
    # Opus judging cost real money.
    def line_sig(e):
        tl = e["trendline"]
        return [tl["slope"], tl["intercept"], tl["anchor_ts"]]

    pending = [e for e in watchlist.values()
               if e.get("llm", {}).get("line") != line_sig(e)]
    if not pending:
        return

    try:
        client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    except Exception as exc:             # missing key etc. — scan still works
        log.warning("LLM filter disabled: %s", exc)
        return

    log.info("LLM filter: judging %d new/refit entries…", len(pending))
    for entry in pending:
        df = fetch(entry["symbol"], entry["timeframe"])
        if df is None:
            continue
        try:
            verdict = judge_entry(client, df, entry,
                                  config.TIMEFRAME_MS[entry["timeframe"]])
        except Exception as exc:         # API error: keep unjudged, retry next cycle
            log.warning("LLM filter error on %s %s: %s",
                        entry["symbol"], entry["timeframe"], exc)
            continue
        entry["llm"] = {**verdict, "line": line_sig(entry)}
        if verdict["valid"]:
            history.log_validated(entry)
        log.info("LLM %s %s %s — %s",
                 "VALID" if verdict["valid"] else "REJECT",
                 entry["symbol"], entry["timeframe"], verdict["reason"])
