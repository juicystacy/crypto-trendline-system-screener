"""
Live smoke-test for the LLM judge (llm_filter.judge_entry).

Unlike selftest.py this DOES hit the Anthropic API — it renders one clean
synthetic uptrend, fits the real support line, and asks Sonnet to judge it.
A clean textbook line should come back valid=True.

Run:  python test_llm.py        (needs ANTHROPIC_API_KEY in the env)
"""

import sys

import analysis
import config
import llm_filter
from selftest import zigzag_path, candles_from_path, TF_MS


def main() -> int:
    df = candles_from_path(zigzag_path())
    _, swing_lows = analysis.find_swings(df)
    anchor = int(df["ts"].iloc[0])
    tl = analysis.fit_pivot_line(swing_lows, "support", anchor, TF_MS, bars=df)
    assert tl is not None, "engine failed to fit the textbook support line"

    entry = {
        "symbol": "TEST/USDT:USDT",
        "timeframe": "1h",
        "direction": "up",
        "trendline": tl,
    }

    import anthropic
    client = anthropic.Anthropic()   # auth is resolved lazily at request time

    try:
        verdict = llm_filter.judge_entry(client, df, entry, TF_MS)
    except anthropic.AuthenticationError:
        print("no credentials — set ANTHROPIC_API_KEY in this shell and re-run.")
        return 1

    # Contract the screener relies on: a dict with a bool `valid` + string reason.
    assert isinstance(verdict, dict), f"expected dict, got {type(verdict)}"
    assert isinstance(verdict.get("valid"), bool), f"bad valid: {verdict!r}"
    assert isinstance(verdict.get("reason"), str) and verdict["reason"], \
        f"bad reason: {verdict!r}"

    print(f"model:   {config.LLM_MODEL}")
    print(f"valid:   {verdict['valid']}")
    print(f"reason:  {verdict['reason']}")
    # Clean textbook line — warn (don't fail) if the judge rejects it, since a
    # borderline call is a prompt/model matter, not a wiring bug.
    if not verdict["valid"]:
        print("WARN: judge rejected a textbook-clean line — inspect the prompt.")
    print("\nOK — live judge call returned a schema-valid verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
