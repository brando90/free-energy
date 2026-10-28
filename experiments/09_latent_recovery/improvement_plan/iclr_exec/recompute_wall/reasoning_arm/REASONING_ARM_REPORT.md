# Reasoning-channel satellite — does deliberation breach the recompute wall?

**Run date:** 2026-07-28 · Lead: Elyas Obbad · Harnesses `arm_a.py` (gpt-5.1, OpenAI
Responses API) + `arm_b.py` (DeepSeek-R1-Distill-Qwen-7B, local vLLM on skampere1 GPU7).
**Worlds:** identical to the E10 substrate-gap bridge — `regime2/programs.jsonl`
(the gpt-5.1 bridge manifest is reused byte-for-byte for ARM A; ARM B rebuilds
injections on the same worlds from R1's own gold). Cells: `adjacent_contradiction`
(readable), `onehop_kc1` (1 op), `deep_kc5` (5 ops).
**Grading:** the SAME EXPG two-world re-execution validator (`classify_run`) over the
FINAL visible output for every arm — format-robust, reads the continuation's numbers.
**Spend:** ARM A this-run **$5.36** (cumulative ledger **$96.26** of the $150 code
hard-stop / $500 ceiling; this-run cap $65 never approached). ARM B is local, no API cost.
**Results reported RAW. No paper edits in this task** — framing is the author's call.

---

## 0. Headline (stated plainly)

**No reasoning condition brings computed-cell absorption down to the readable-cell floor
(~0.01). The wall is dented at the frontier-and-depth, not broken. It is a split:**

- **gpt-5.1, 1-op cell:** reasoning ON leaves absorption at the ceiling — **0.988**
  [0.964, 0.996] vs 1.000 OFF. No breach.
- **gpt-5.1, 5-op cell:** reasoning ON produces a **real, partial breach** — **0.875**
  [0.827, 0.911] vs 1.000 OFF (CIs disjoint). But a large majority still absorb.
- **R1-Distill, visible think channel (B2):** turning the think channel on does **not**
  breach the computed wall — absorption stays high, **0.925** at both 1-op and 5-op,
  essentially unchanged from its no-think continuation baseline (B1: 0.975 / 0.954).
- **The mechanism is the same in both models:** the reasoning/think channel readily
  re-derives and corrects the READABLE contradiction, but for COMPUTED cells it mostly
  **never re-visits the planted upstream value** — it takes the already-written line as
  given and continues. And when it *does* surface the correct value, it still absorbs a
  large fraction of the time (below).

---

## 1. ARM A — gpt-5.1 reasoning ON vs OFF (per cell)

Endpoint: OpenAI **Responses API** (the only gpt-5.1 path that returns reasoning
**summaries**). Within-endpoint, R=4 reps/world/condition.
- OFF: `reasoning={"effort":"none"}` (reasoning_tokens = 0, runtime-verified).
- ON: `reasoning={"effort":"medium","summary":"auto"}`, `max_output_tokens=4000`.
Anchor DV = `final_output_absorbed` (re-execution of the continuation's numbers).

| cell | cond | n_cf | **absorbed** [Wilson-95] | silently_corrected | flagged | mean reasoning_tok |
|---|---|---|---|---|---|---|
| adjacent_contradiction (readable) | off | 280 | 0.061 [0.038, 0.095] | 0.939 | 0.00 | 0 |
| adjacent_contradiction (readable) | **on** | 280 | **0.004** [0.001, 0.020] | 0.996 | 0.00 | 258 |
| onehop_kc1 (1 op) | off | 240 | 1.000 [0.984, 1.0] | 0.000 | 0.00 | 0 |
| onehop_kc1 (1 op) | **on** | 240 | **0.988** [0.964, 0.996] | 0.013 | 0.00 | 148 |
| deep_kc5 (5 ops) | off | 240 | 1.000 [0.984, 1.0] | 0.000 | 0.00 | 0 |
| deep_kc5 (5 ops) | **on** | 240 | **0.875** [0.827, 0.911] | 0.125 | 0.00 | 360 |

**Cross-check (endpoint validity):** the Responses-`none` OFF baseline reproduces the
E10 bridge chat-`none` gpt-5.1 row (bridge: onehop 1.00, deep 1.00, adj 0.09; here:
1.00 / 1.00 / 0.061). The off baseline is sound; the on/off delta is a pure reasoning effect.

**Reasoning genuinely engaged and scales with effort** (probe, 7-op chain a=7…g):
off gets it **wrong** (336), medium/high get it **right** (606) with reasoning_tokens
47 / 131. On the real cells, mean reasoning_tokens rises with cell depth (148 → 360),
i.e. gpt-5.1 spends more deliberation on the deeper computed cell — and that is exactly
where the only breach appears.

**Reading:** reasoning drives the readable cell to the floor (0.061 → 0.004) and holds
the 1-op wall at the ceiling (0.988). Only at 5 ops does deliberation move absorption
off the ceiling — and even then 7 of 8 continuations still absorb the plant.

## 2. ARM A — re-derivation taxonomy over the reasoning SUMMARY

**Caveat first:** OpenAI exposes *summarized* reasoning, not the raw CoT, so this scan is
a **lower bound** on true re-derivations. `strict` = the summary literally writes
`planted_var = true_value`; `loose` = the true-value integer appears anywhere in the
summary (a high-recall upper bound; can overcount). Buckets over cf rows, ON condition.

| cell | n | strict: correct / absorb / never | loose: correct / absorb / never |
|---|---|---|---|
| adjacent_contradiction | 154 | 0.266 / 0.000 / 0.734 | 0.643 / 0.000 / 0.357 |
| onehop_kc1 | 150 | 0.013 / 0.000 / 0.987 | 0.013 / 0.080 / 0.907 |
| deep_kc5 | 132 | 0.099 / 0.053 / 0.848 | 0.204 / 0.242 / 0.553 |

**The key mechanistic sentence:** on `deep_kc5`, even restricting to rows whose summary
*contained the correct recomputed value* (loose), the final output **still used the
plant in 32/59 = 54%** of them (loose correct 27 vs absorb 32). Under the conservative
strict signal it is 7/20 = 35%. **Re-derivation in the reasoning channel does not
guarantee correction of the final output.** For the readable cell the summary re-derives
and corrects freely (0.266 strict / 0.643 loose) with **zero** absorb — deliberation
catches readable contradictions but is blind-by-default to its own upstream computed values.

## 3. ARM B — R1-Distill-7B: raw continuation (B1) vs visible think (B2)

Local vLLM 0.24.0, bf16, one free GPU (skampere1:7), temp 0.6 / top_p 0.95, R=4.
**Own-trace gold via R1's own think channel** (raw-completion gold yields ~0 on kc5 —
the model needs its think channel to solve 5-op traces; think-gold solved **190/190**,
kc1 60/60, kc5 60/60, kr1 70/70; attrition zero). Injections rebuilt on that gold.
- **B1** raw completion, NO think tags: user render + injected prefix as one token stream.
- **B2** chat template WITH `<think>`: injected partial trace in the user turn; R1 opens
  `<think>`, deliberates, then continues. Final answer = text after `</think>`, graded by
  `classify_run`. Think truncation (no `</think>` within budget): 3/280, 5/240, 7/240 — negligible.

| cell | **B1 absorbed** [Wilson-95] | B1 no-output | **B2 absorbed** [Wilson-95] | B2 no-output |
|---|---|---|---|---|
| adjacent_contradiction (readable) | 0.007 [0.002, 0.026] | 0.218 | 0.011 [0.004, 0.031] | 0.054 |
| onehop_kc1 (1 op) | 0.975 [0.947, 0.989] | 0.000 | 0.925 [0.885, 0.952] | 0.037 |
| deep_kc5 (5 ops) | 0.954 [0.920, 0.974] | 0.000 | 0.925 [0.885, 0.952] | 0.050 |

**Reading:** the visible think channel does **not** breach the computed wall. Absorption
on computed cells stays at ~0.92–0.93 with think on, statistically indistinguishable from
(and if anything marginally *below*) the no-think baseline. The readable cell is caught
either way (~0.01). NB: B1 on the readable cell has 22% *no-output* — raw continuation
often derails on the visible contradiction rather than emitting a gradeable output; B2's
think channel resolves that (5% no-output), which is why B2 is the cleaner readable row.

## 4. ARM B — think-channel re-derivation taxonomy (RAW think text, full fidelity)

Unlike ARM A, the R1 `<think>` channel is the raw CoT, so this scan is faithful (not a
lower bound). Buckets over cf rows, B2.

| cell | n | strict: correct / absorb / never | loose: correct / absorb / never |
|---|---|---|---|
| adjacent_contradiction | 280 | 0.857 / 0.007 / 0.136 | 0.982 / 0.011 / 0.007 |
| onehop_kc1 | 240 | 0.054 / 0.000 / 0.946 | 0.067 / 0.125 / 0.808 |
| deep_kc5 | 240 | 0.029 / 0.008 / 0.963 | 0.042 / 0.087 / 0.871 |

**The same asymmetry, sharper and at full fidelity:** the think channel re-derives and
corrects the READABLE contradiction **85.7%** of the time (strict; 98.2% loose), but on
`deep_kc5` it **never re-derives the planted value in 96.3%** of runs (strict; 87.1%
loose). It does not go back and recompute the already-written upstream line — it accepts
it and traces forward. When the true value *does* surface (loose), it still absorbs
21/31 = **68%** on deep_kc5. Deliberation, when it happens at all on computed cells,
does not reliably overturn the plant.

## 5. The three comparisons, side by side

1. **gpt-5.1 on-vs-off:** readable 0.061→0.004; 1-op 1.000→0.988 (ceiling holds);
   5-op 1.000→**0.875** (partial breach, CIs disjoint).
2. **R1-Distill continuation-vs-think:** readable ~0.01 both; 1-op 0.975→0.925;
   5-op 0.954→0.925. Think does **not** move computed absorption off its high plateau.
3. **Think-channel re-derivation taxonomy:** both models re-derive+correct the READABLE
   contradiction readily (gpt-5.1 0.27–0.64; R1 0.86–0.98) but **almost never re-derive
   the planted COMPUTED value** (gpt-5.1 deep never=0.85 strict; R1 deep never=0.96
   strict). Re-derivation, where it occurs, still absorbs in 35–68% of deep cases.

## 6. Does ANY reasoning condition move computed-cell absorption off the ceiling?

- **gpt-5.1 / 1-op: NO** — 0.988 [0.964, 0.996], still at ceiling.
- **gpt-5.1 / 5-op: PARTIALLY YES** — 0.875 [0.827, 0.911]; off the ceiling (upper CI
  0.911 < 0.95) but a 7/8 majority still absorb. This is the single condition that dents
  the wall, and it dents deepest exactly where the model spends the most reasoning tokens.
- **R1-Distill (think) / 1-op and 5-op: NO** — 0.925 at both; the visible think channel
  does not breach the computed wall and does not beat its own no-think continuation.

**Bottom line, honest either way:** the recompute wall **holds under deliberation** for
the open-weight reasoning model and holds at the ceiling for the frontier model at 1 op;
it is **partially breached only by gpt-5.1 at depth 5** (1.00 → 0.875). No reasoning
condition collapses computed-cell absorption toward the readable floor. The wall is a
"don't recompute what is already written upstream" wall: deliberation re-checks readable
contradictions but, by default, does not re-derive planted computed values — and even
re-derivation does not guarantee the final output is corrected.

---

## 7. Provenance / exact params / repro

- **ARM A** `reasoning_arm/arm_a_gpt51/` — `arm_a.py {run,validate,summarize}`, keys from
  env only. Model `gpt-5.1`, OpenAI Responses API. OFF `reasoning.effort=none`;
  ON `reasoning.effort=medium, summary=auto`, `max_output_tokens=4000`, no temperature
  (reasoning models reject it). Worlds = `e10_api_bridge/manifest_gpt-5.1.jsonl` (regime2,
  fp identical to the bridge/prefill models). R=4. Raw/validated JSONL + `summary_arm_a.json`
  + `cost_ledger.json` (this-run gpt-5.1 $5.36; cumulative $96.26, extra ledgers summed for
  the $150 hard-stop). Re-derivation scan `rederive_scan.py` (strict `V=T` assign + loose
  T-token); **summaries are lossy → ARM A taxonomy is a lower bound.**
- **ARM B** `reasoning_arm/arm_b_r1distill/` (mirror of skampere1
  `/var/tmp/eobbad/recompute_wall/reasoning_arm_b/out/`) — `arm_b.py`. Weights READ-ONLY
  from `/lfs/skampere1/0/shared_hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B`
  (snapshot `09db661939…`; 15 GB, complete — **no download needed**, the R1-Distill-7B
  snapshot was already in the shared cache). vLLM 0.24.0 / torch 2.11+cu130, bf16,
  `gpu_memory_utilization=0.85`, `max_model_len=8192`, GPU 7 (verified 0 MiB before launch;
  released after). Gold = own-trace via chat-template think channel, 4× oversample
  (1 greedy + 3 @ temp 0.6), clip at first output line, `gold_solve_eval` filter →
  190/190. B1 raw completion (max_tokens 640); B2 chat template + `<think>` (max_tokens
  3000, split at `</think>`). R=4, temp 0.6, top_p 0.95, DeepSeek EOS stop. n per cell:
  adj 280, onehop 240, deep 240 (B1 ≥100, B2 ≥60 both satisfied). Grading identical
  `classify_run`. Re-derivation scan over the RAW think text (full fidelity).
- **Shared:** `rederive_scan.py` used by both arms; taxonomy buckets defined mechanically
  (`taxonomy_bucket` strict / `_loose`).
