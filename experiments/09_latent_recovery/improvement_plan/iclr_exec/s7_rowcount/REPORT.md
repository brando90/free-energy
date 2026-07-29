# S7 solved()-fix + strict-pass row-count reconciliation + E1 cache inventory — REPORT

ICLR revision plan item **S7** (§3 SHOULD) + the plan's flag on STRICT_METRICS row totals
(W2/Q2 row of the verdict table) + E1 checkpoint-cache prep. Zero GPU. Executed 2026-07-23.

Slug dir (all new files here; nothing under `results/` or existing `improvement_plan/` subdirs
was modified): `$EXP/improvement_plan/iclr_exec/s7_rowcount/`
where `$EXP = /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`.

Note: two earlier aborted attempts exist (`iclr_exec/s7_solved_rowcount/`,
`iclr_exec/s7_solved_fix_rowcount/`); I did not touch them. This is an independent re-run and
reaches the same conclusions, re-verified from source.

---

## PART A — solved() discourse-marker fix

### The bug (located exactly)
The stored per-rollout `solved` flag is produced upstream by `src/common.py::solved()` (and the
identical `src/expa_global_expansion.py::solved()`), which matches the **raw** final sentence to
the target and never strips leading discourse markers ("So,", "Therefore,"). A completion ending
`"So, Alex is a tumpus."` therefore fails the match even though the target was reached. The formal
validator DOES strip markers — `validator.py:131` `sents = [strip_marker(s) for s in sents]`, then
`validator.py:153` `final_ok = bool(sents) and _norm(sents[-1]) == _norm(target)` — so these
rollouts re-validate as `valid_rederivation`. `NATURAL_ERRORS_REPORT.md` §1 flags exactly this
("Nine 'failures' re-validate as valid_rederivation … pipeline `solved()` did not strip discourse
markers; footnote-level").

The consuming line in `natural_errors/extract_natural_errors.py` is **line 391**:
```python
is_fail = (not r["solved"]) or v["class"] != "valid_rederivation"   # r["solved"] = stale, marker-blind
```

### The one-line fix
`v` is already the marker-aware validator result (`v = validate_continuation(...)`, line 389), and
`v["final_ok"]` is *precisely* `solved()` with discourse markers stripped. So the fix is a
one-token substitution — strip-markers-before-the-validity-check without recomputing anything:
```python
is_fail = (not v["final_ok"]) or v["class"] != "valid_rederivation"   # S7 fix: marker-aware
```
The original locked script was **not edited in place**. A patcher (`patch_s7.py`) reads the locked
copy and writes two runnable variants into the slug dir with `OUT_DIR` redirected so the locked
`natural_errors/` outputs are never touched: `extract_baseline.py` (redirect only) and
`extract_fixed.py` (redirect + the one-line fix).

### Baseline reproduction (HARD RULE 5 — validated against published anchors)
`extract_baseline.py` reproduces `natural_errors/summary.json` to the digit — every anchor OK:
n=236 combined fresh invalid entity-facts (135 PrOntoQA + 101 EXPD); PrOntoQA modal cell
`false_cwa_unentailed` 115/135 = 85%; usable 15/28 = 0.54 vs inert 5/107 = 0.047; control
false-positive 0/486; the 9 revalidate-valid anomalies; auditable failures 525 (225 PrOntoQA + 300
EXPD); combined fabricated-rule raw 1184 (= 1118 unentailed + 66 entailed-true restatements).

### Result of the fix — what reclassifies, what moves
**Exactly 9 rollouts reclassify failed → valid** (all had stored `solved=False` but validator
class `valid_rederivation`): 3 legacy + 3 EXPA + 3 EXPD —
`1hop_…4testhops…example79`, `…example91`, `1hop_…5testhops…example1` (legacy);
`3hop…example2`, `3hop…example93`, `4hop…example97::in_context_example5` (expa);
`attr_f090_d5_B`, `cat_f196_d1_usable`, `cat_f196_d1_inert` (expd). Control valid cohort 486 → 495,
still 0 false positives; the anomaly list is empty after the fix.

**None of the four report headline numbers move:**

| headline | baseline | fixed |
|---|---|---|
| n natural invalid entity-facts (comb / PrOntoQA / EXPD) | 236 / 135 / 101 | **236 / 135 / 101** |
| 93% coverage (126/135) — entity-fact axes coverage | invariant | **invariant** |
| 85% modal cell (PrOntoQA `false_cwa_unentailed` 115/135) | 115/135 | **115/135** |
| 54% vs 5% usability (PrOntoQA usable vs inert) | 15/28 vs 5/107 | **15/28 vs 5/107** |
| control false-positive | 0/486 | 0/495 |

**Why it cannot move (proof):** a `valid_rederivation` rollout has every entity-fact derivable
from the true closure and reaches the target, so it contributes **zero** fresh natural errors. The
fresh-error inventory is therefore invariant — verified directly: the 236 `entity_fact` rows in
`natural_errors.jsonl` are **byte-identical** across baseline and fixed (SHA256 `1c6a046d5172af7f`
both), and `natural_rollouts.jsonl` differs by exactly the 9 removed rows, 0 added. The 93%/85%/54%
vs 5% statistics are all functions of those 236 invariant rows.

**Secondary / denominator counts that DO shift (footnote-level only):**
- auditable failures 525 → **516** (PrOntoQA 225 → 219; EXPD 300 → 297)
- valid control cohort 486 → **495**
- unentailed fabricated-rule statements: combined 1,118 → **1,112** (PrOntoQA 369 → 365;
  EXPD 749 → 747) — the 9 valid rollouts each restated a few novel rules the validator ignores but
  the step-walk tallies as rule objects (raw fabricated 1184 → 1177; entailed-true 66 → 65)
- PrOntoQA rollout taxonomy: `other_error_kinds_only` 81 → 78; `truncation/derail_no_falsehood`
  16 → 13 (the 107 has-fresh-error rollouts and the downstream-use rollout counts are unchanged)

The report/defense sentence "3,715 gold rollouts; 525 failures … zero false positives on the 486
valid rollouts" should read **516 failures / 495 valid** after the fix; nothing else changes.

### One-paragraph appendix disclosure (drop-in)
> The pipeline's `solved()` predicate matches the final continuation sentence to the target without
> stripping leading discourse markers ("So,", "Therefore,"), whereas our formal validator strips
> them before matching. Nine gold rollouts (3 legacy, 3 EXPA, 3 EXPD) were consequently stored as
> unsolved yet re-validate as valid rederivations under the marker-aware validator. Reclassifying
> them (using the validator's marker-stripped final-answer check as the solved criterion) removes
> them from the failed cohort. The natural-error inventory is unchanged — every statistic derived
> from fresh natural errors is invariant, because a valid rederivation contains no fresh error by
> construction (the 236 extracted error rows are bit-identical before and after). Only
> failure-cohort denominators shift by nine (525 → 516 auditable failures; 225 → 219 on PrOntoQA;
> the count of naturally fabricated rule statements drops 1,118 → 1,112), none of which affects the
> coverage (93%), modal-cell (85%), or usable-vs-inert absorption (54% vs 5%) results.

---

## PART B — strict-pass row-count reconciliation

### Definitive count
The strict hop-sound / strict-repair pass (`stage0_strict/strict_metrics.py`, `strict_metrics.json`
generated 2026-07-02T06:53:30) consumed **6,147 rows**, confirmed three independent ways:
1. `strict_rows.jsonl` contains exactly **6,147** lines.
2. The per-experiment `rows_used` in the report metadata block sum to 6,147:
   EXPA 1,800 + EXPC_POLARITY_CONTROL 84 + EXPC_POLARITY_CONTROL_FULL 1,524 + legacy 2,739
   (contradiction 390 + distractor 390 + falsehood 105 + neghop2 260 + neghop3 199 + neghop4 134
   + neghop5 91 + negstep 390 + paraphrase 390 + wrong 390).
3. The 21 per-family tables' pooled `n` sum to 6,147.

Denominator convention = "all rows retained (generation_failed and unparsed included)" (report
header + script). **EXPB (900 rows) is excluded by schema** (`assistant_prefill_*` prefix format)
and is in neither total.

### What each published/candidate total refers to (all reproduced from `strict_rows.jsonl`)
| figure | source | meaning | reproducible? |
|---|---|---|---|
| **6,147** | STRICT tables / `strict_rows.jsonl` / metadata sum | all retained rows across the 21 experiment×family cells — the strict-pass denominator (the W2/Q2 verdict already cites this) | YES (3 ways) |
| 4,112 | `truth=="false"` | audited-false plants — the M1/M3 SRR & stated-reliance denominator (remainder: 1,645 true plants + 390 unparsed distractor-rule plants) | YES |
| 5,938 | `class≠unparsed` | parseable rows (209 unparsed-class rows dropped) | YES |
| 4,090 | `valid_rederivation` | closure-valid completions | YES |
| 4,539 | **M5 retro-distance audit** (STATUS L25, `stage0_distance/`) | a **different pass**: EXPA (1,800) + legacy (2,739); the EXPC polarity controls are not in the distance audit | YES |
| **4,857** | **STATUS.md L32 only** | matches **no** data artifact and **no principled subset** | **NO** |

### Verdict on 4,857
"4,857 rows" appears **only** in the STATUS.md running-log entry for the strict pass; a grep of the
whole experiment tree finds it in no artifact (`strict_metrics.json`, `strict_rows.jsonl`, report,
or any summary). It equals none of the meaningful totals above — not all-rows 6,147, audited-false
4,112, parseable 5,938, valid 4,090, the M5 subset 4,539, nor the pre-EXPC_FULL figure 4,623.
(A brute-force subset-sum can hit 4,857 only with arbitrary 14-of-22 cell mixes that split
EXPC_FULL's four conditions incoherently — combinatorial coincidence, not a principled grouping;
by contrast 4,539 = EXPA+legacy is principled and STATUS-corroborated.) The same STATUS entry
confirms EXPC_FULL was included and EXPB skipped, so 4,857 is not a pre-EXPC_FULL snapshot either.
**Conclusion: 4,857 is a stale/erroneous hand-count in the log and must not be cited; the strict
pass ran over 6,147 rows.**

Robustness note: Q1's headline 8.7pp strict compression is a **within-cell** comparison on the two
450-row EXPA cells and is unaffected by any total-count confusion — verified directly from
`strict_rows.jsonl`: one-hop hop-sound valid 135/450 = 0.300 (closure 302/450 = 0.671, SRR
113/450 = 0.251) vs global 96/450 = 0.213 (closure 177/450 = 0.393, SRR 1/450 = 0.002).

### One-paragraph reconciliation (drop-in)
> The strict hop-sound / strict-repair pass was run over **6,147 stored generations** — the sum of
> the per-experiment row counts EXPA (1,800), EXPC_POLARITY_CONTROL (84),
> EXPC_POLARITY_CONTROL_FULL (1,524) and the ten legacy perturbation families (2,739); the released
> `strict_rows.jsonl` contains exactly 6,147 rows and the per-family tables sum to the same total,
> all under the "all rows retained" denominator (generation-failed and unparsed rows included).
> EXPB (900 rows) is excluded for schema reasons. Within this pass, 4,112 rows are audited-false
> plants (the denominator for the strict-repair and stated-reliance metrics) and 5,938 are
> parseable. A separate retro refutation-distance audit ran over 4,539 rows (EXPA plus the legacy
> families only). The figure "4,857" that appears in an internal running log corresponds to no
> artifact or principled subset and is superseded by 6,147.

---

## PART C — E1 checkpoint cache inventory (skampere2)

Cache: `HF_HOME=/lfs/skampere2/0/eobbad/.cache/huggingface` → hub at
`…/huggingface/hub` (`env.sh` sets it; `HF_HUB_OFFLINE=0`). "Complete" below = the local snapshot's
`*.safetensors` symlinks all resolve to real blobs on disk. **No downloads attempted; none made.**

E1's four named models — **all present and complete** (base + instruct where relevant):

| E1 model | cached repo (complete) | size | snapshot rev | shards |
|---|---|---|---|---|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` (base) / `-1.5B-Instruct` | 2.9G / 2.9G | 8faed761 / 989aa798 | 1 / 1 |
| Qwen2.5-32B | `Qwen/Qwen2.5-32B-Instruct` | 62G | 5ede1c97 | 17 |
| OLMo-2-7B | `allenai/OLMo-2-1124-7B` (base) / `-7B-Instruct` | 28G / 14G | 7df9a825 / 470b1fba | 6 / 3 |
| Llama-3.1-8B | `NousResearch/Meta-Llama-3.1-8B` (base) / `-8B-Instruct` | 15G / 30G | 1f47e50c / d10aef79 | 4 / 4 |

Anchor `Qwen/Qwen2.5-7B-Instruct` present & complete (15G, rev **a09a3545**, 4 shards — matches the
rev in NATURAL_ERRORS_REPORT and EXPC replay).

**E1 would need to download: NOTHING for the four core models**, with two caveats:
1. **`meta-llama/Llama-3.1-8B-Instruct` is a 76K gated STUB** (rev 0e9e39f2, 0 safetensors — refs
   only). E1 launch scripts must point Llama-Instruct at the complete
   **`NousResearch/Meta-Llama-3.1-8B-Instruct`** mirror (rev d10aef79) or supply an HF token to
   finish the gated download. The Llama **base** (`NousResearch/Meta-Llama-3.1-8B`) is complete.
2. **Gemma is absent** (no `google/*` or `*gemma*` in the cache). If Elyas resolves E1 open-Q4 by
   adding a Gemma arm as the extra non-Qwen lineage, that checkpoint (~5–18G) is the only new
   download. `Qwen2.5-32B` **base** is also absent (only Instruct) — irrelevant if E1 uses the
   Instruct 32B, matching the anchor.

Also cached (not E1): Qwen2.5-{0.5B,3B,14B}-Instruct; DeepSeek-R1-Distill-Qwen-7B (E1 excludes
R1-distill — deliberation channel); Mistral-7B-Instruct-v0.2; Llama-3.1-Tulu-3-8B{,-SFT,-DPO}
(all complete, 15G each); HarmBench-Llama-2-13b-cls. **S9 note** (not E1): the OLMo-2 post-training
ladder is incomplete — base + Instruct complete, but `OLMo-2-1124-7B-SFT` and `-DPO` are 9.4M STUBS
(weights not pulled); the Tulu-3 ladder is fully present, so S9's "open-weights post-training
ladder" could run on Tulu-3 now but would need the two OLMo SFT/DPO weights completed.

---

## Files (all under `$EXP/improvement_plan/iclr_exec/s7_rowcount/`)
- `patch_s7.py` — reads the locked script, writes baseline + fixed variants (never edits original)
- `extract_natural_errors_ORIG.py` — verbatim copy of the locked script
- `extract_baseline.py`, `extract_fixed.py` — redirect-only / redirect+one-line-fix
- `out_baseline/`, `out_fixed/` — full rerun outputs (summary.json, natural_errors.jsonl,
  natural_rollouts.jsonl, run.log)
- `diff_s7.py` — reproduction check vs published + baseline↔fixed diff + entity-fact byte-identity
- `fab_check.py` — unentailed fabricated-rule shift; `anchor_check.py` — strict Q1 anchor + totals;
  `subsetsum_check.py` — 4,857/4,539 subset search
- `REPORT.md` — this file
