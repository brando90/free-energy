# S7 + row-count reconciliation + E1 cache inventory — REPORT

Item: ICLR revision plan §3 SHOULD **S7** (solved() discourse-marker fix) + plan flag on
STRICT_METRICS row totals + E1 prep cache inventory. Zero GPU. Executed 2026-07-22.

All new files under `$EXP/improvement_plan/iclr_exec/s7_solved_fix_rowcount/`
(`$EXP` = `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`).
Nothing under `results/` or existing `improvement_plan/` subdirs was modified.

---

## PART A — solved() discourse-marker fix

### The bug
The stored `solved` flag on every gold rollout is produced by `src/common.py::solved()`
(and the identical `src/expa_global_expansion.py::solved()`):

```python
def solved(gen_text, target):
    sents = split_sentences(gen_text)
    return len(sents) > 0 and norm(sents[-1]) == norm(target)
```

It compares the **raw** final sentence to the target. It never strips leading discourse
markers, so a completion ending "So, Alex is a tumpus." fails the match (`norm` keeps the
"so ") even though the model reached the target. The audit's `validator.validate_continuation`
DOES strip markers (`sents = [strip_marker(s) for s in sents]`, then
`final_ok = _norm(sents[-1]) == _norm(target)`), so those rollouts re-validate as
`valid_rederivation`. `NATURAL_ERRORS_REPORT.md` §1 flags exactly this ("Nine 'failures'
re-validate as valid_rederivation … pipeline `solved()` did not strip discourse markers;
footnote-level").

### The one-line fix (find & change)
In `extract_natural_errors.py::main()` the failure test was:
```python
is_fail = (not r["solved"]) or v["class"] != "valid_rederivation"
```
`r["solved"]` is the stale, marker-blind stored flag. Fixed to use the marker-aware validator
field `v["final_ok"]` (= `norm(strip_marker(final_sentence)) == norm(target)`), which is exactly
"solved() with discourse markers stripped":
```python
is_fail = (not v["final_ok"]) or v["class"] != "valid_rederivation"
```
(Original script NOT edited in place. Copies live in the slug dir: `extract_baseline.py`
= verbatim analysis with OUT_DIR redirected; `extract_fixed.py` = redirect + this one line.
`patch_s7.py` produced them; `out_baseline/`, `out_fixed/` hold the two full rerun outputs.)

### Baseline reproduction (HARD RULE 5 — validated against published anchors)
`extract_baseline.py` reproduces the report to the last digit:
n=236 combined fresh invalid entity-facts (135 PrOntoQA + 101 EXPD); control false-positive
0/486; the **9** revalidate-valid anomalies (exact ids below); usable 15/28=0.536 vs inert
5/107=0.047; modal `false_cwa_unentailed` 115/135=0.852. (The raw `n_rule_fabricated`=428/1184
differs from the report's 369/1118 only because the report subtracts the 59/66 entailed-true
"derived rule" restatements — 428−59=369, 1184−66=1118 — i.e. an accounting split, not a bug.)

### Result of the fix — what reclassifies, what moves
Exactly **9 rollouts reclassify** failed→valid (all had stored `solved=False`; validator class
`valid_rederivation`). Control valid cohort 486→**495**, still **0** false positives. Fixed
anomaly list is empty.

The 9: legacy `1hop_…4testhops…example79`, `…example91`, `1hop_…5testhops…example1`;
expa `3hop…example2`, `3hop…example93`, `4hop…example97::in_context_example5`;
expd `attr_f090_d5_B`, `cat_f196_d1_usable`, `cat_f196_d1_inert`.

**None of the four report headline numbers move:**

| headline | baseline | fixed |
|---|---|---|
| n natural invalid entity-facts (comb / PrOntoQA / EXPD) | 236 / 135 / 101 | **236 / 135 / 101** |
| 93% coverage (126/135; modal 115/135=85%) | 115/135=0.852 | **115/135=0.852** |
| usability asymmetry (PrOntoQA usable vs inert) | 15/28=0.536 vs 5/107=0.047 | **15/28=0.536 vs 5/107=0.047** |
| control false-positive | 0/486 | 0/495 |

Proof it cannot move: a `valid_rederivation` rollout has every entity-fact derivable from the
true state and reaches the target, so it contributes **zero** fresh natural errors. The fresh-error
inventory is therefore invariant. Verified directly: the 236 `entity_fact` rows in
`natural_errors.jsonl` are **byte-identical** across baseline and fixed (SHA256
`f7113682d1c0e643` both), and `natural_rollouts.jsonl` differs by exactly those 9 removed rows,
0 added.

**Secondary/denominator counts that DO shift (footnote-level only):**
- total auditable failures 525→**516** (PrOntoQA 225→219, EXPD 300→297)
- valid control cohort 486→**495**
- unentailed fabricated-rule statements: combined 1,118→**1,112** (PrOntoQA 369→365, EXPD 749→747)
  — the 9 valid rollouts each restated a few novel rules, which the validator ignores but the
  step-walk tallies as rule objects.
- PrOntoQA rollout taxonomy: "other-error-kinds-only" 81→78, "truncation/derail-no-falsehood" 16→13
  (the 107 has-fresh-error rollouts and the 39/196 downstream-use rollout counts are unchanged).

The "3,715 gold rollouts / 525 failures / 0 FP on 486 valid" sentence in the report/defense text
should read **516 failures / 495 valid** post-fix; nothing else changes.

### One-paragraph appendix disclosure (drop-in)
> The pipeline's `solved()` predicate matches the final continuation sentence to the target
> without stripping leading discourse markers ("So,", "Therefore,"), whereas our formal validator
> strips them before matching. Nine gold rollouts (3 legacy, 3 EXPA, 3 EXPD) were consequently
> stored as unsolved yet re-validate as valid rederivations under the marker-aware validator.
> Reclassifying them (using the validator's marker-stripped final-answer check as the solved
> criterion) removes them from the failed cohort: the natural-error inventory is unchanged —
> every reported statistic derived from fresh natural errors is invariant, because a valid
> rederivation contains no fresh error by construction (the 236 extracted error rows are
> bit-identical before and after). Only failure-cohort denominators shift by nine
> (525→516 auditable failures; 225→219 PrOntoQA; the count of naturally fabricated rule statements
> drops 1,118→1,112), none of which affects the coverage (93%), modal-cell (85%), or
> usable-vs-inert absorption (54% vs 5%) results.

---

## PART B — strict-pass row-count reconciliation

### Definitive count
The strict pass (`stage0_strict/strict_metrics.py`, `strict_metrics.json` generated
2026-07-02T06:53:30) consumed **6,147 rows**, confirmed three independent ways:
1. Per-experiment `rows_used` in the report metadata block sum to 6,147:
   EXPA 1,800 + EXPC_POLARITY_CONTROL 84 + EXPC_POLARITY_CONTROL_FULL 1,524 + legacy 2,739
   (contradiction 390 + distractor 390 + falsehood 105 + neghop2 260 + neghop3 199 + neghop4 134
   + neghop5 91 + negstep 390 + paraphrase 390 + wrong 390 = 2,739).
2. The 21 per-family tables' pooled `n` sum to 6,147.
3. `strict_rows.jsonl` = exactly 6,147 lines.

Denominator convention = "all rows retained (generation_failed and unparsed included)"
(report header + script docstring). EXPB (900 rows) is **excluded by schema** and is in neither total.

### What each candidate total means
| figure | source | what it is | reproducible? |
|---|---|---|---|
| **6,147** | STRICT_METRICS tables / `strict_rows.jsonl` | all retained rows across the 21 experiment×family cells (strict-pass denominator) | YES (3 ways above) |
| 4,112 | truth=="false" in `strict_rows.jsonl` | audited-false plants — the M1/M3 SRR & stated-reliance denominator (the rest: 1,645 true plants [benign/true-interruption/paraphrase/mostly-true "wrong"] + 390 unparsed distractor-rule plants) | YES |
| 5,938 | class∉{unparsed} | parseable rows (209 unparsed-class rows dropped) | YES |
| 4,090 | valid_rederivation / closure_valid | closure-valid completions | YES |
| **4,857** | STATUS.md line 32 only | — matches NOTHING — | **NO** |

### Verdict on 4,857
STATUS.md's "4,857 rows" appears **only** in the running log (grep of `improvement_plan` + `src`
finds it in no artifact). It equals none of: all-rows 6,147, audited-false 4,112, parseable 5,938,
valid 4,090, or the M5 retro-distance subset 4,539 (=EXPA 1,800 + legacy 2,739). No family subset
of the current strict pass reconstructs it. It is a **stale/erroneous hand-count in the log** and
must not be cited. (STATUS same-entry text confirms EXPC_FULL was included and EXPB skipped, so it
is not a pre-EXPC_FULL figure either — that would be 4,623.)

Independent robustness note: Q1's headline 8.7pp strict compression (one-hop hop-sound valid
0.300 vs global 0.213) is a within-cell comparison on the two 450-row EXPA cells — I verified
135/450=0.300 and 96/450=0.213, plus SRR 113/450=0.251 and 1/450=0.002, directly from
`strict_rows.jsonl` — so it is unaffected by the total-count confusion.

### One-paragraph reconciliation (drop-in)
> The strict hop-sound / strict-repair pass was run over **6,147 stored generations** — the sum
> of the per-experiment row counts EXPA (1,800), EXPC_POLARITY_CONTROL (84),
> EXPC_POLARITY_CONTROL_FULL (1,524) and the ten legacy perturbation families (2,739); the
> released `strict_rows.jsonl` contains exactly 6,147 rows and the per-family tables sum to the
> same total, all under the "all rows retained" denominator (generation-failed and unparsed
> included). EXPB (900 rows) is excluded for schema reasons. Within this pass, 4,112 rows are
> audited-false plants (the denominator for the strict-repair and stated-reliance metrics) and
> 5,938 are parseable. The figure "4,857" that appears in an internal running log is a
> transcription error with no corresponding subset and is superseded by 6,147.

---

## PART C — E1 checkpoint cache inventory (skampere2)

Cache: `/lfs/skampere2/0/eobbad/.cache/huggingface/hub` (a second, near-full AFS cache
`/afs/cs.stanford.edu/u/eobbad/.cache/huggingface/hub`, 4.6G / 5G quota, holds only small
artifacts — not the E1 weights). **No downloads should be attempted; none were.**

E1's four named models — all present and complete (weights on disk):

| E1 model | cached repo (complete) | size | snapshot rev |
|---|---|---|---|
| Qwen2.5-1.5B (base+inst) | `Qwen--Qwen2.5-1.5B` / `-1.5B-Instruct` | 2.9G / 2.9G | 8faed761 / 989aa798 |
| Qwen2.5-32B | `Qwen--Qwen2.5-32B-Instruct` | 62G | 5ede1c97 |
| OLMo-2-7B (base+inst) | `allenai--OLMo-2-1124-7B` / `-7B-Instruct` | 28G / 14G | 7df9a825 / 470b1fba |
| Llama-3.1-8B (base+inst) | `NousResearch--Meta-Llama-3.1-8B` / `-8B-Instruct` | 15G / 30G | 1f47e50c / d10aef79 |

Anchor `Qwen--Qwen2.5-7B-Instruct` present & complete (15G, rev **a09a3545** — matches the rev in
NATURAL_ERRORS_REPORT). Also cached: Qwen2.5-{0.5B,3B,14B}-Instruct; DeepSeek-R1-Distill-Qwen-7B
(15G, but E1 excludes R1-distill — deliberation channel); Mistral-7B-Instruct-v0.2; Tulu-3-8B
{,-SFT,-DPO}; HarmBench cls.

**E1 would need to download: nothing for the four core models.** Two caveats:
1. **`meta-llama/Llama-3.1-8B-Instruct` is a 76K STUB** (gated repo; only refs/metadata, 0
   safetensors). E1 launch scripts must point Llama-Instruct at the complete
   **`NousResearch/Meta-Llama-3.1-8B-Instruct`** mirror (rev d10aef79) or configure an HF token to
   finish the gated download. The Llama **base** (`NousResearch/Meta-Llama-3.1-8B`, 15G) is complete.
2. **Gemma is absent** (no gemma-2 / gemma-3 in either cache). If Elyas resolves E1 open-question
   Q4 by adding a Gemma arm as the extra non-Qwen lineage, that checkpoint (~5–18G) would be the
   only new download. Qwen2.5-32B **base** is also absent (only Instruct cached) — irrelevant if
   E1 uses the Instruct 32B, as the anchor does.

Bonus for S9 (not E1): the OLMo-2 post-training ladder is partially cached — base + Instruct
complete, but `OLMo-2-1124-7B-SFT` and `-DPO` are 9.4M STUBS (weights not pulled); the Tulu-3-8B
{base? no — SFT/DPO} ladder is fully present. S9's "open-weights post-training ladder" would need
the two OLMo SFT/DPO weights completed.

---

## Files (all under `$EXP/improvement_plan/iclr_exec/s7_solved_fix_rowcount/`)
- `extract_natural_errors_ORIG.py` — verbatim copy of the locked script
- `extract_baseline.py`, `extract_fixed.py`, `patch_s7.py`, `diff_s7.py`
- `out_baseline/` (summary.json, natural_errors.jsonl, natural_rollouts.jsonl, run.log)
- `out_fixed/`   (same set, post-fix)
- `REPORT.md` (this file)
