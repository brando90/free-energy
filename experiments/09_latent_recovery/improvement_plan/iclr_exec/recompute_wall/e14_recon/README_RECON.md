# E14 RECON — Yee reconciliation cell-set (computed site, k=1)

**v2, 2026-08-03 — 3×2, was 2×2.** CPU-only build; no GPU/vLLM/API was run.
Nothing under `improvement_plan/` was modified except this directory
(`e14_recon/`); the generator and runner live in
`/lfs/skampere2/0/eobbad/scratch/recon/` and import the existing code.

> **What changed in v2** (two design defects found in review, both fixed):
> 1. **`large` arm added.** In v1, factor A was perfectly collinear with a
>    unit-digit cue — `pm10_20` preserved the last digit in 1800/1800 worlds and
>    `pm1` changed it in 1800/1800 — so "small implausible delta" and
>    "last-digit mismatch" were not identifiable. A third regime
>    (`|delta| ∈ [100,400]`, last digit **changed**) breaks the collinearity.
> 2. **Measurement-separation diagnostics added.** `|out_cf − out_true| =
>    2·|delta|`, which is exactly **2** in the `pm1` arm. Every validated row now
>    persists the parsed final output integer, and summarize emits near-miss
>    counts and a **bias bound**. The DV is unchanged.
>
> Also: every contrast key now states its own sign direction; `summarize
> --e11-summary <bad path>` is a hard error instead of a silent skip; `prepare`
> re-runs `recon_gen.post_checks()` fail-closed.
>
> The v1 world pool is preserved verbatim at `worlds_v1/`.

## Purpose

Yee et al. (COLM 2024, arXiv:2405.15092) run a near-identical paradigm on math
word problems and report that GPT-4 **recovers** from planted "calculation
errors" 74–79% of the time (our audit corrects this to ~68–71%). Our instrument
reports ~**100% absorption** on computed errors. The hypothesis is that the gap
is the **salience channel**, not the verification channel:

* their plants are tiny, implausible deltas (random integer in ±1..3 — e.g.
  "60/4 = 16.5 sets", "$82.50" where a round number belongs), and
* they sit in the **same sentence as their operands**, so the contradiction is
  readable without recomputation.

Ours use last-digit-preserving ±10/±20 deltas on a **bare** site line that
prints no operands.

This cell-set crosses those factors **inside our own instrument**, on the
computed site, so their gradient can be reproduced (or fail to reproduce)
without changing paradigms.

## Design — 3×2, within-world paired

Site: computed, k=1 (the `kc1` shape — `V = o1 + o2`, both operands stated on
their own lines immediately above; the site is the result of exactly one op).

| delta regime | **bare** (result only, ours) | **full** (operands printed, Yee-like) |
|---|---|---|
| **pm10_20** — \|Δ\| ∈ {10,20}, last digit **preserved** (ours) | `recon_bare_pm10_20` — *our corner* | `recon_full_pm10_20` |
| **pm1** — \|Δ\| = 1, last digit **changed** (Yee-like) | `recon_bare_pm1` | `recon_full_pm1` — *Yee corner* |
| **large** — \|Δ\| ∈ [100,400], last digit **changed** (decoupler) | `recon_bare_large` | `recon_full_large` |

**All six cells ride ONE shared world.** Same program, same operands, same true
value, same gold trace — only the site line's presentation and planted value
differ. The design is perfectly paired within-world, and the summarizer reports
paired within-world differences with a cluster bootstrap over `program_id`
alongside the marginal rates.

Worked example (`k1_43d06a_00900913`, operands 42 + 47, true value 89,
`out_true` 200):

```
depth_k1_bare        k = j + m; k = 79           <- E11's own family
recon_bare_pm10_20   k = j + m; k = 79           <- byte-identical anchor
recon_full_pm10_20   k = j + m = 42 + 47; k = 79
recon_bare_pm1       k = j + m; k = 90
recon_full_pm1       k = j + m = 42 + 47; k = 90 <- Yee-like on BOTH factors
recon_bare_large     k = j + m; k = 191
recon_full_large     k = j + m = 42 + 47; k = 191
```

### Why `large` exists — the identification argument

Factor A now has three levels that cross magnitude against the unit-digit cue:

| regime | \|Δ\| | last digit | plausible? |
|---|---|---|---|
| `pm10_20` | 10–20 | **preserved** | mid |
| `pm1` | 1 | **changed** | implausibly small |
| `large` | 100–400 | **changed** | implausibly large |

Measured on the 1800-world pool: `pm10_20` preserves the last digit 1800/1800;
`pm1` and `large` each change it 1800/1800. So magnitude and the unit-digit cue
are no longer the same variable.

**Read it as:**

* `large` behaves like `pm10_20` → the driver is **delta magnitude /
  plausibility** (a tiny delta is what makes the plant checkable).
* `large` behaves like `pm1` → the driver is the **last-digit cue** (a unit-digit
  mismatch is what makes the plant checkable), and the magnitude story is wrong.
* `large` behaves like neither → both channels are live; report the two
  contrasts (`effect_large_minus_pm10_20`, `effect_large_minus_pm1`) separately.

`summarize` writes exactly this into `contrasts.decoupling_read`, with a
`verdict_heuristic` string and an explicit caveat that it is a point estimate to
be read against the paired CIs and the bias bounds.

**Prediction to be tested (unchanged):** absorption collapses in
`recon_full_pm1` and stays ~1.0 in `recon_bare_pm10_20`, with the rest
intermediate.

## The measurement-separation issue, and how it is bounded

The k=1 world shape is `out = Q1 + Q2` with `Q1 = V + c` and `Q2 = V + c'`, so a
plant of size Δ at the site propagates to the output **twice**:

```
|out_cf − out_true| = 2 · |Δ|
```

Re-derived from the interpreter on all 1800 worlds:

| regime | separation \|out_cf − out_true\| |
|---|---|
| `pm1` | **2** (1800/1800) |
| `pm10_20` | 20 or 40 |
| `large` | 202 … 336 |

**The hazard.** In the `pm1` arm the "absorbed" answer and the "corrected"
answer are only **2** apart. An ordinary arithmetic slip in a genuinely
*absorbing* continuation can therefore land exactly on `out_true` and be scored
as recovery. Absorption is *easier to lose* in `pm1` than in `pm10_20` for a
reason that has nothing to do with checking — and the bias points in the
direction of our own hypothesis. That is a validity threat, not a nuisance.

**What was done about it — diagnostics, not a redefinition of the DV.** The
label rule (`three_way_label`) and the classifier (`expg/validate.classify_run`)
are untouched. What changed:

* **every validated row** now carries `final_output_value` (the integer the
  model actually printed — the grader already parsed it, it was just being
  discarded), `out_true`, `out_cf`, `separation`, `delta_regime`, `abs_delta`,
  and a bucket label. `validate` asserts row-by-row that
  `final_output_absorbed ⇔ (bucket == exact_cf)`, so the two code paths cannot
  drift apart silently;
* **summarize** emits, per cell, `exact_true` / `exact_cf` / `near_miss` /
  `other` counts, the `separation` distribution (per row and per program), and a
  `bias_bound`;
* **`contrasts.separation_bias`** aggregates those to the regime level and, for
  each regime contrast, reports `observed_effect`, `bias_bound_on_contrast`,
  `contrast_exceeds_bias_bound`, and a `worst_case_interval`.

**Definitions.** A **near miss** is a final output on *neither* target but
within **±3** of one of them: the observable arithmetic-slip population. The
**bias bound** is the largest the absorbed rate could move if that whole
population were reassigned:

```
max_absorbed_rate_shift = near_miss_count / n
absorbed_rate_if_all_near_miss_reassigned_to_absorbed = min(1, absorbed + shift)
absorbed_rate_symmetric_envelope                      = [absorbed − shift, absorbed + shift]
```

The one-sided number is the literal reassignment (near-miss rows are never
`absorbed` today — `absorbed` requires the output to equal `out_cf` exactly).
The symmetric envelope additionally covers the *unobservable* direction — a slip
in an absorbing continuation that happened to land exactly on `out_true` — under
the stated assumption that slips are no more likely to hit a target exactly than
to miss it by 1–3.

For a **contrast** between two regimes the bound is the sum of the two
per-regime bounds (worst case: they move in opposite directions):

```
bias_bound_on_contrast(A−B) = near_miss_rate(A) + near_miss_rate(B)
contrast_exceeds_bias_bound = |observed_effect| > bias_bound_on_contrast
```

**This is the number to quote** when stating that the `pm1`/`pm10_20`
comparison is not an artifact of measurement separation. If
`contrast_exceeds_bias_bound` is `false` on the real run, the contrast is *not
safe to interpret* as a checking effect, however large it looks.

## Anchors — the pm10_20 cells are E11-identical

`recon_bare_pm10_20` / `recon_full_pm10_20` are **construction-identical** to
E11's `depth_k1_bare` / `depth_k1_full`. They reuse `gen_depth.build_program`'s
own planted value (never recomputed — recomputing would re-draw the rng) and its
own `render_site_line_body`. Verified on every world in both v2 pools
(`verify_anchor.py`, exit 0, `VERDICT: ANCHOR_EQUIVALENCE_CONFIRMED`): all of
`planted_var, planted_value, true_value, force_after_line, delta_policy,
transform, opacity, out_cf, site_body, site_body_true` equal the `depth_k1_*`
family, and 1440/1440 (240 pool) + 10800/10800 (1800 pool) injected trace lines
round-trip through `trace_format.parse_trace_line`.

**If a GPU run does not reproduce E11's `k1_bare` / `k1_full` absorbed rates on
these two cells, the build is wrong.** Pass `--e11-summary` to have the deltas
computed into `contrasts.anchor_check.anchor_delta_recon_minus_e11`. Targets
(from `e11_run/results/*/summary_tables.json`, n=1350 rollouts / 150 programs):

| model | `k1_bare` → `recon_bare_pm10_20` | `k1_full` → `recon_full_pm10_20` |
|---|---|---|
| Qwen2.5-7B-Instruct | **0.7844** | **0.9363** |
| Llama-3.1-8B-Instruct | **0.9859** | **0.9926** |

Note that "~100% absorption" holds for Llama but **not** for Qwen at k=1 bare
(0.784), and that in E11's own data `full` absorbs *more* than `bare` on both
models — printing the operands did not restore verification. The prediction
check is therefore stated **relative to each model's own anchor**, never against
an absolute 1.0.

The three `depth_k1_*` families are deliberately left on every world:
`audit_depth.audit_opacity_byte_identity` inspects families literally named
`depth_k<k>_{bare,full,partial}` and hard-fails `opacity_missing_bare` without
them. The runner scores only the six `recon_*` cells.

### ⚠ Anchor caveat introduced by the `large` arm — read before interpreting it

A `large` plant needs `|x − vtrue| ≥ 100` with `x` inside the attainable-sum
window `[22, 198]` (see below). That is **impossible for vtrue ∈ [98, 122]**,
which is the *mode* of the vtrue distribution. Those worlds are rejected
outright (`large_starved`, **688 / 2504** audited builds = 27.5%), and the
rejection applies to the whole world, **including its pm10_20 anchor cells**.

Measured shift, v1 pool vs v2 pool (both n=1800):

| | v1 | v2 |
|---|---|---|
| vtrue mean | 114.14 | 114.85 |
| vtrue median | 114 | **126** |
| fraction with vtrue ∈ [98,125] | 0.2994 | **0.0372** |
| program_ids shared with the other pool | — | 1311 / 1800 |

The mean barely moves; the distribution becomes bimodal with a hole in the
middle. **The E11 anchor comparison is therefore against a vtrue-restricted
subpopulation.** If the anchor deltas come out non-zero on the real run, this is
the first thing to rule out — 1311 program_ids are common to both pools, so the
anchor rates can be recomputed on the intersection.

## Seeds and determinism

* Base seed **900911** (= `900000 + k*911`, k=1). Disjoint from E11's k=1 base
  seed 700911 and from E13's arms.
* Seeds consumed consecutively from 900911; rejected seeds skipped.
* `pm1` **and** `large` plants share one rng stream,
  `random.Random((seed << 6) ^ 0xE14)` (E13's `0xE13D` idiom), with `pm1`
  drawing first — so a seed accepted by both v1 and v2 gets the *same* `pm1`
  plant in both.
* `pm10_20` plant rng: gen_depth's own `rng_for(seed, k)` — untouched.
* Deterministic: regenerating byte-reproduced `recon_worlds.jsonl`
  (md5 `06d934f25712aa34f109938f08d1df6b`).

## Worlds

| file | worlds | md5 | note |
|---|---|---|---|
| `worlds/recon_worlds_pool1800.jsonl` | 1800 | `a371ef77…` | **use this** — power-matched to E11 |
| `worlds/recon_worlds.jsonl` | 240 | `06d934f2…` | exactly the first 240 lines of the pool; smoke tests |
| `worlds/recon_funnel*.json` | — | | funnels, reject taxonomy, delta + separation distributions |
| `worlds_v1/` | 1800 + 240 | | the **v1 (2×2) pool**, preserved unchanged |

Build funnel (1800 pool): **3192 seeds → 2504 built → 2504 passed the E11 audit
stack → 1800 admitted legal pm1 *and* large plants → 1800 re-passed the full
audit stack with all six recon families attached → 1800 passed the post-checks.**
Rejects: `large_starved` 688, `build:build` 479, `build:program_error` 205,
`pm1_starved` 16, `build:no_axis_c_delta` 4. **Zero** audit failures at any
layer.

Realized |Δ| — `pm1` = 1 (1800/1800); `pm10_20` = 10 (914) / 20 (886);
`large` = 101…168, mode 101. Realized separation — `pm1` = 2; `pm10_20` = 20/40;
`large` = 202…336.

`large` is sign-forced by construction (vtrue ∈ [25,97] ⇒ positive Δ,
vtrue ∈ [123,197] ⇒ negative Δ; 0 violations in 1800 worlds), so its signed distribution is bimodal —
977 negative / 823 positive across the pool, but the sign is a deterministic
function of vtrue, not an independent draw. `pm1` and `pm10_20` remain balanced.

### Sizing warning (read before launching)

E11's **measured** k=1 gold-solve rate is **0.1417** (Qwen2.5-7B) and **0.2039**
(Llama-3.1-8B) — see `e11_run/results/E11_*/cohort.json`. The gold gate, not the
generator, determines surviving worlds per cell. At those rates:

* 240 worlds → ≈34 (Qwen) / ≈49 (Llama) per cell → `INVALID_underpowered` under
  E11's own rule (<50).
* 1800 worlds → ≈255 (Qwen) / ≈367 (Llama) eligible → caps at 150 per cell.

Because the six cells share one world, surviving worlds per cell are equal by
construction; there is no per-cell attrition after the gate. Verified on the
mock 240-world run: 150 worlds in every one of the six cells.

**Generation cost is 1.5× v1** — six families per world instead of four.

## Run it

```bash
S=/lfs/skampere2/0/eobbad/scratch/recon
OUTBASE=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e14_recon
E11R=$OUTBASE/../e11_run/results
M=Qwen/Qwen2.5-7B-Instruct

# worlds default to $OUTBASE/worlds/recon_worlds_pool1800.jsonl
/usr/bin/python3 $S/recon_run.py prepare   --model "$M" --cap 150 --reps 8
/usr/bin/python3 $S/recon_run.py generate  --model "$M" --cap 150 --reps 8 --gpu 0
/usr/bin/python3 $S/recon_run.py validate  --model "$M" --cap 150 --reps 8
/usr/bin/python3 $S/recon_run.py summarize --model "$M" \
    --e11-summary $E11R/E11_qwen7b/summary_tables.json
/usr/bin/python3 $S/recon_run.py report    --model "$M"
```

`--out-dir` defaults to `$OUTBASE/EXPG_RECON_<model basename>`, so the above
writes `$OUTBASE/EXPG_RECON_Qwen2.5-7B-Instruct/summary_tables.json` (E11
schema) plus `REPORT.md`. Pass `--out-dir` to override, `--worlds` to swap
pools. Only `generate` touches the GPU; every other stage is CPU.

For Llama, swap `M=NousResearch/Meta-Llama-3.1-8B-Instruct` and point
`--e11-summary` at `E11_llama8b/summary_tables.json`.

> `--e11-summary` is now **fail-loud**: a path that does not exist, is not JSON,
> has no `cells` object, or lacks `k1_bare`/`k1_full` aborts `summarize` with a
> non-zero exit and writes nothing. v1 silently skipped the anchor check on a
> typo.

> `prepare` re-runs `recon_gen.post_checks()` on every world, fail-closed. This
> is not decoration: with v1's prepare, a world whose anchor `site_body` had been
> corrupted was **admitted**; with v2's it is rejected
> (`postcheck:anchor_body_mismatch_recon_bare_pm10_20`).

### Heads-up on an E11 artifact (not fixed here — `$IP` is read-only)

`e11_run/results/E11_llama8b/summary_tables.json` records
`"model": "Qwen/Qwen2.5-7B-Instruct"`. Its `queue_status.json` correctly says
`NousResearch/Meta-Llama-3.1-8B-Instruct`. Cause: `e11_run.stage_prepare` never
writes `model` into `run_metadata.json`, so `stage_summarize`'s
`meta.get("model") or args.model` falls back to the argparse **default** (Qwen)
whenever `--model` is omitted at the summarize step. The numbers are fine; only
the label is wrong. `recon_run.py` is immune: it resolves the model from
`queue_status.json` (what actually generated), records all three candidate
sources under `summary.model_provenance`, and warns if `--model` disagrees.

## What the runner reuses (import, not copy)

* worlds — `gen_depth.build_program(1, seed)`
* audits — `audit_depth.make_audit_fn` (`nearest_anc_dist=4`) **plus**
  `recon_gen.post_checks`, both re-run fail-closed at `prepare` on every world
* injection — `e11_run.build_injection_e11` (`transform='value'`: the
  byte-audited `site_body` spliced over the model's **own** gold site line)
* gold gate — `e11_run.stage_generate` + `inject.gold_solve_eval`, with the
  recon family taxonomy patched in and `KS` pinned to `[1]`
* scoring — `expg/validate.classify_run`, the **same** validator as E11.
  `stage_validate` is a local copy of E11's that persists three extra fields;
  the classification and the label rule are untouched.
* stats — `e11_run._cell_summary` / `wilson` / `cluster_boot_rate`

## Outputs

`summary_tables.json` is a superset of the E11 schema. Row schema of
`validated_outputs.jsonl` is a strict superset of E11's (adds
`final_output_value, out_true, out_cf, separation, output_bucket, delta_regime,
abs_delta, no_output_claim`); `manifest.jsonl` / `cohort.json` /
`queue_status.json` are key-identical to E11's.

Contrast blocks, **all sign-explicit** (`<A>_minus_<B>` means `rate(A) −
rate(B)`; every opacity effect is `FULL − BARE`):

* `anchor_check` — recon pm10_20 rates vs E11 `k1_bare`/`k1_full`, plus
  `anchor_delta_recon_minus_e11`
* `main_effect_delta` — `rate_by_regime`, plus `effect_pm1_minus_pm10_20`,
  `effect_large_minus_pm10_20`, `effect_large_minus_pm1`, each with a paired
  within-world twin
* `main_effect_opacity` — `effect_full_minus_bare`, marginal and paired
* `interaction_3x2` — `opacity_effect_full_minus_bare_by_regime` and the three
  `did_<A>_minus_<B>_of_full_minus_bare` keys
* `corner_contrast` — `yee_corner_minus_our_corner` and its explicitly-named
  negation `gap_our_minus_yee_corner`, plus the two `large` corners; Yee's
  reported (0.74–0.79) and audited (0.68–0.71) bands carried as reference only
* `separation_bias` — per-cell and per-regime near-miss / bias bounds, and
  `regime_contrast_bias_bounds`
* `decoupling_read` — the magnitude-vs-last-digit verdict the `large` arm exists
  to produce
* `prediction_check` — booleans for the stated prediction, relative to the
  model's own E11 anchor

> **v1 key removed:** `contrasts.interaction_2x2` no longer exists. It carried
> the sign bug (`opacity_effect_*` was FULL−BARE while the adjacent
> `difference_in_differences` was built BARE−FULL). Any downstream code reading
> that key must be updated to `interaction_3x2` rather than silently getting a
> flipped number.

## Provenance

Scripts and captured output in `/lfs/skampere2/0/eobbad/scratch/recon/`
(`.bak` of every v1 file kept alongside):
`recon_gen.py`, `recon_run.py`, `audit_recon_worlds.py`, `verify_anchor.py`,
`probe5.py`, `RECON_BUILD_NOTES.md`, and under `dryrun/`: `dryrun_grade.py`,
`check_gradient.py`, `check_nearmiss.py`, `check_prepare_failclosed.py`,
`check_e11summary_failloud.py`, `check_goldgate.py`, `check_sitelines.py`,
`check_schema.py`, `check_resume.py`, `check_anchor_guard.py`,
`check_plant_sweep.py`, `check_grades.py`, each with a `*_v2.out` capture.

CPU dry-run evidence (mock backend, no GPU): 60 worlds × 6 cells × 9 rollouts =
3240 validated rows per mode; differential modes
`absorbed/true/garbage/neither/nogen/flagged` each graded 1.0 on their designed
label in all six cells; `gradient` reproduced every contrast to 5e-5 with six
distinct rigged rates; `nearmiss` injected 192 slipped continuations
(94 of them in the separation-2 `pm1` arm) and every bucket, bias bound and
regime contrast bound recomputed exactly.
