Opus 5 bridge probe, author-directed post-review run.
Protocol/code: e10_api_bridge/e10bridge.py imported verbatim; only
OUT_DIR, CELLS (2 fig2 cells), roster entry, and a standalone
cost ledger overridden (this ledger is NOT summed into the E10
cumulative budget files; add it to _extra_ledgers if extended).
Worlds: regime2 kr1+kc1 subset (130 programs). Mode: bridge,
temperature omitted, thinking disabled, max gold 512 / cont 384.

## Outcome (2026-07-30)
Gold stage: 130/130 requests returned stop_reason=refusal (input ~790 tok, output 1 tok,
instant classifier). Same outcome class as claude-fable-5 (360/360, cat=cyber, E10_TOPUP_FABLE_REPORT).
No manifest/cells produced; spend $1.55 at assumed opus pricing.

## Framing probe (2 programs x 3 framings, saved verbatim in chat log + LOG.md)
- harness prompt (tf.make_user_message, shipped protocol): refused 2/2, stop_details.category=cyber
- Fable-report arithmetic-worksheet rewording:            refused 2/2, category=cyber
- plain "write its execution trace line by line" phrasing: OK 2/2 (model writes correct trace,
  wraps code in backticks)
Interpretation: the classifier keys on the harness INSTRUCTION/EXEMPLARS block (and the worksheet
rewording), not on the task itself. Opus 5 is measurable only under a modified prompt = disclosed
protocol deviation; would need a calibration model re-run under the same modified prompt.
