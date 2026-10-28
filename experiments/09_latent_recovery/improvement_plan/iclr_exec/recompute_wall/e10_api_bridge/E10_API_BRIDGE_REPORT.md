# E10 Substrate-Gap Bridge — the recompute wall on OpenAI + the NEWEST Anthropic models, on TRUE computed cells

**Run date:** 2026-07-27 · Lead: Elyas Obbad · Harness `e10bridge.py` (EXPG generator/injector/two-world validator reused verbatim).
**This run's API spend: $21.84.** Cumulative project ledger now **$52.69** (base $30.85 + this run). $120 stop-and-report never approached; $150 run cap and $100 code hard-stop both clear.
**Outputs:** `improvement_plan/iclr_exec/recompute_wall/e10_api_bridge/` (raw, validated, `summary_bridge.json`, `bridge_analysis.json`, `cost_ledger.json`, `worlds_provenance.json`).

## 0. What this closes

Track A (`e10_api/E10_API_REPORT.md`) established the wall on program-trace cells for the two OLDER Anthropic prefill models (haiku-4-5, sonnet-4-5) and flagged the **SUBSTRATE GAP**: the OpenAI models and the newest Anthropic models (claude-opus-4-8, claude-sonnet-5) had only **NL-substrate** data, and NL plants are not computed values, so those rows could not speak to the recompute wall. On NL, opus-4-8/sonnet-5 *rejected* the derivational plant (0.00 absorption) — "checks at this tier." This run runs all of them on the **EXPG program-trace substrate (genuinely computed cells)**, on the **identical worlds** as the two prefill models (regime2 `programs.jsonl`, fp `64583365…`, kr1:70 kr8:60 kc1:60 kc5:60).

## 1. A hard constraint discovered up front (must be read)

**claude-opus-4-8 and claude-sonnet-5 REJECT assistant-message prefill via the API** — verified 2026-07-27, error: *"This model does not support assistant message prefill. The conversation must end with a user message."* The EXPH2 true-prefill harness therefore **cannot be run on the two frontier models at all.** The only API path to the program-trace substrate for them is the **chat bridge**: the partial trace (with the plant) is presented in the *user* turn with an explicit "continue this trace" instruction (disclosed format shift, per E10 §2 — chat re-presentation can substitute re-planning for continuation). The anchor metric is `final_output_absorbed` from validator **re-execution of the continuation's numbers**, which is format-robust.

**Bridge-confound control (the key to trusting the frontier rows):** haiku-4-5 and sonnet-4-5 were *also* run in bridge mode here, and they already have TRUE-PREFILL numbers in regime2 on these same worlds. On the computed cells the two regimes **coincide**: haiku bridge onehop/deep = **1.00/1.00** vs prefill **1.00/1.00**; sonnet-4-5 bridge **1.00/1.00** vs prefill **0.98/1.00**; readable `adjacent_contradiction` = **0.00** in both regimes for both. **The bridge does not destroy absorption on computed cells** — it only nudges the copy/re-read cells (`opfree`) modestly upward. So opus-4-8/sonnet-5 bridge ≈1.0 on computed cells is a genuine wall signal, not a chat-replanning artifact.

## 2. The result — the wall holds across the entire roster, frontier included

3-way outcome on cf cells (format-robust): **absorbed** = `final_output_absorbed` & not flagged; **silently_corrected** = correct output, plant overwritten, not flagged; **flagged** = frozen code-doubt lexicon fired. `absorbed` rates shown; readable = `adjacent_contradiction`; recompute = pooled `opfree_kr1/kr8, onehop_kc1, deep_kc5`. n = 58–70/cell.

| model | mode | readable (adj) | opfree_kr1 | opfree_kr8 | **onehop_kc1** | **deep_kc5** | pooled recompute | margin | prefill ref (1hop/deep) |
|---|---|---|---|---|---|---|---|---|---|
| **claude-opus-4-8** | bridge | **0.00** | 0.31 | 0.15 | **1.00** | **0.98** | 0.60 | **+0.60** | — (prefill blocked) |
| **claude-sonnet-5** | bridge | **0.01** | 0.23 | 0.43 | **1.00** | **1.00** | 0.65 | **+0.63** | — (prefill blocked) |
| claude-haiku-4-5 | bridge | 0.00 | 0.36 | 0.28 | 1.00 | 1.00 | 0.65 | +0.65 | **1.00 / 1.00** |
| claude-sonnet-4-5 | bridge | 0.00 | 0.09 | 0.13 | 1.00 | 1.00 | 0.54 | +0.54 | **0.98 / 1.00** |
| gpt-4.1 | bridge | 0.01 | 0.94 | 0.97 | 1.00 | 1.00 | 0.98 | +0.96 | — |
| gpt-4o | bridge | 0.01 | 0.17 | 0.15 | 1.00 | 0.95 | 0.55 | +0.54 | — |
| gpt-5.1 | bridge | 0.09 | 0.89 | 0.92 | 1.00 | 1.00 | 0.95 | +0.86 | — |
| gpt-3.5-turbo-instruct | completion | 0.15 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.85 | — |

Wilson-95 on the decisive cells (non-overlapping, tight): opus-4-8 onehop **1.00 [0.94, 1.0]**, deep **0.98 [0.91, 1.0]**, readable **0.00 [0.0, 0.05]**; sonnet-5 onehop **1.00 [0.94, 1.0]**, deep **1.00 [0.94, 1.0]**, readable **0.01 [0.003, 0.08]**; gpt-4o onehop **1.00 [0.94, 1.0]**, deep **0.95 [0.86, 0.98]**.

### Findings, straight

1. **The wall holds for the OpenAI models AND for the newest Anthropic models on true computed cells.** Every one of the 8 models absorbs the genuinely computed cells (onehop_kc1 = 1 op, deep_kc5 = 5 ops) at **≥0.95** while catching the readable contradiction at **≤0.15**. The recompute-minus-readable margin is **+0.54 to +0.96** on every model. This is the readable-vs-recompute asymmetry replicating across the full frontier/API roster.

2. **The asymmetry is at its sharpest exactly at the frontier — the flagged case the task named.** claude-opus-4-8 and claude-sonnet-5, which *rejected* the NL derivational plant (0.00 absorption, "checks at this tier" in the E10 NL panel), **absorb the genuinely computed cells at ~1.0** while catching the re-readable contradiction at ~0.00 and the copy cells partly (opfree 0.15–0.43). doubt_lex fired **0×** on the computed cells for both. The frontier models catch readable/copied falsehoods but are **blind to their own computed values.** The substrate gap was masking this: the NL plants were not computed, so the frontier models rejected them; computed plants are absorbed wholesale. This is the "sharpest asymmetry at the frontier" outcome, reported as measured, not massaged.

3. **Two wall *shapes*, both recompute ≫ readable.** (a) *Sharp* wall — boundary exactly at "requires an operation": opus-4-8, sonnet-5, sonnet-4-5, haiku-4-5, **gpt-4o** catch copies/re-reads (opfree low 0.09–0.43) but absorb computed cells ~1.0. (b) *Blunt* wall — absorbs anything not immediately re-readable: **gpt-4.1, gpt-5.1, gpt-3.5-turbo-instruct** absorb both copies and computed cells at ~1.0, catching only the adjacent contradiction. gpt-3.5-turbo-instruct via legacy completions (the one true single-token-stream API model) is the extreme: recompute **1.00 across the board**, readable 0.15 — the cleanest API analogue of the Qwen-7B open-weights pilot (which also absorbed opfree at 0.83–0.91).

4. **No result against the wall.** The honest-finding-against-universality branch (frontier models that DON'T absorb computed cells) did not occur — every frontier model absorbs. Reported so it is on record that this branch was live and did not fire.

## 3. Caveats (pre-stated, honored)

- **Frontier models are bridge-only.** opus-4-8/sonnet-5 have no true-prefill API path; every number for them is chat-bridge. The confound is bounded by the haiku/sonnet-4-5 prefill↔bridge calibration (§1): computed cells coincide across regimes, so the frontier computed-cell numbers are trustworthy; the copy cells (opfree) are the confound-sensitive ones and are *not* the wall's primary channel.
- **n = 58–70/cell**, the E10 §5.2 "50≤n<100: scored, CI-widened, flagged" band, below the n≥150 target. CIs reported. A top-up to n≥150 requires regenerating a **superset** of worlds (backward-compatible seed bases) — flagged, not silently assumed; not done here to keep worlds byte-identical to the two prefill models for the frontier comparison.
- **Gold attrition negligible:** solve rate 1.00 for opus/haiku/sonnet-4-5, 0.996 sonnet-5 (1 parse), 0.996 gpt-3.5 (1 wrong_value); OpenAI chat & gpt-4o 1.00. Full eligible cohorts.
- **Never pooled improperly:** bridge and prefill rows are labeled; the frontier bridge rows are compared to the calibration-anchored bridge rows, not silently merged with regime2 prefill.
- **Phrasing:** all rates are measured absorption on the named roster, never a universal law. The wall replicates on all 8; it is reported per model with Wilson CIs.

## 4. Provenance / repro

`e10bridge.py {prepare,parity,gold,perturb,validate,summarize} --model KEY`, keys from env only. Worlds = regime2 programs.jsonl (identical to the prefill models). Analyzer `e10_bridge_analyze.py`. Parity check passed (`parity_check.json`): prefill user turn byte-identical to `tf.make_user_message`; completion = user + '\n' + prefix single stream; bridge = same instruction/exemplars/program with prefix in the user turn. Cost ledger `e10_api_bridge/cost_ledger.json` (this run $21.84; per-model: opus $5.94, sonnet-5 $3.62, sonnet-4-5 $3.51, gpt-4o $2.61, gpt-4.1 $2.06, gpt-5.1 $1.87, haiku $1.16, gpt-3.5 $1.07).
