# S2 + W3 — EXPB joint-outcome analysis & strict-schema adaptation

Item: ICLR revision plan **S2** (EXPB strict-schema adaptation) + reviewer **W3**
(the local-certificate result is a parse artifact; analyze all outcomes jointly).
Zero GPU. All numbers computed cluster-side from the authoritative EXPB row files.

- Slug dir (cluster): `improvement_plan/iclr_exec/s2w3_expb_joint/`
- Inputs (read-only): `results/EXPB_LOCAL_CERT_FLIP/{manifest,validated_outputs}.jsonl`
  (900 rows each), plus `summary_tables.json` / `EXPB_REPORT.md` for anchors.
- Tooling reused verbatim: `improvement_plan/stage0_strict/strict_metrics.py`
  (`analyze`, `compose`, `summarize_family`, `failed_row`) + `src/validator.py`.
- Model: Qwen2.5-7B-Instruct only (EXPB is single-model). Greedy, max_new_tokens=192.

## Headline (appendix-ready)

The local-certificate intervention **shifts continuation *style* wholesale**; it does
not cleanly reduce error absorption. Adding the direct contradiction pushes 31.7% of
continuations into the **unparsed** bucket (up from 4.0%), draining the parroted and
injection-dependent buckets at the same time. Under an honest joint-outcome accounting
over all 900 rows, **only the verbal-doubt contrast survives worst-case imputation of
the unparsed rows** (+0.370, 111-vs-0 discordant pairs, McNemar p≈8e-34). The
apparent drop in injection-dependence (0.153→0.027) **does not survive**: if the extra
unparsed rows were absorbing, the effect reverses. And even the surviving doubt/repair
signal is doubt-lexicon-driven (W5 caveat) and largely a *restatement of the supplied
certificate* rather than autonomous proof search — hop-sound re-derivation stays flat
at ~0.10–0.12 across all three arms.

---

## Schema adaptation (S2 — "adaptable later" resolved)

`stage0_strict/strict_metrics.py::load_expa` skipped EXPB because it requires the
manifest key **`prefix_steps`**, which EXPB does not have (it stores context as
`proof_prefix_steps` and `assistant_prefill_steps`). The prefill is constructed as

```
assistant_prefill_steps = proof_prefix_steps  (+ certificate C for cert arms)  + [injected F]
```

with the injected falsehood **F always the last prefill step** (verified for **900/900**
rows). The faithful strict-validator prefix is therefore

```
prefix_steps := assistant_prefill_steps[:-1]      # drop F (passed to analyze() separately)
```

which is **condition-aware**: for `LOCAL_CERTIFICATE` it seeds C = complement of F
("Rex is not a vumpus") into the validator's prefix state, and for
`IRRELEVANT_CERTIFICATE_CONTROL` it seeds the unrelated true statement. No other
change to the strict machinery. This adapter feeds EXPB into the *same* `analyze()`
that produced the published STRICT_METRICS_REPORT tables, so EXPB strict numbers are
directly comparable in definition.

**Validation (HARD RULE 5).** The adapted replay reproduces the stored EXPB closure
class for **900/900** rows (0 mismatches). Composition counts reproduce
`EXPB_REPORT.md` exactly (valid 113/140/150, poisoned 46/8/72, unparsed 12/11/95),
doubt counts reproduce exactly (3/114/7), and the imputation bounds reproduce
`stage0/STAGE0_REPORT.md` exactly (LOCAL valid [0.4667, 0.7833], GLOBAL [0.3767,
0.4167], IRRELEVANT [0.5000, 0.5367]). The script asserts every one of these anchors
and raises on drift.

---

## (a) Full joint-outcome composition — proportions of ALL 900 rows

Six mutually-exclusive outcomes partition each arm (n=300); they sum to 300 exactly.
`verbalized_doubt` is an orthogonal overlay (measured on raw text pre-parse), reported
separately in (c).

| outcome | no-cert (GLOBAL) | local-cert (LOCAL) | irrelevant-cert |
|---|---|---|---|
| closure-valid       | 0.377 (113) | 0.467 (140) | 0.500 (150) |
| injection-dependent | 0.153 (46)  | **0.027 (8)** | 0.240 (72) |
| parroted            | 0.367 (110) | 0.117 (35)  | 0.183 (55) |
| derailed            | 0.063 (19)  | 0.073 (22)  | 0.040 (12) |
| **UNPARSED**        | 0.040 (12)  | **0.317 (95)** | 0.037 (11) |
| generation-failed   | 0.000 (0)   | 0.000 (0)   | 0.000 (0)   |

Reading: the local certificate does not simply convert "injection-dependent → clean".
It **redistributes the whole continuation-style distribution** — the injection-dependent
(−0.127) *and* parroted (−0.250) buckets both shrink, and almost all of that mass moves
into **unparsed** (+0.277), not into closure-valid (+0.090). The irrelevant certificate,
by contrast, leaves the style distribution close to baseline (unparsed 0.037) yet reaches
the **highest** closure-valid rate (0.500) — so a certificate that carries **no**
local contradiction produces *more* parsed valid completions than the local one. That
pattern is the reviewer's W3 point, quantified.

Figure: stacked composition bar chart at `expb_composition.png` (draft, for our review
only). CSV at `expb_composition.csv`.

---

## (b) Monotone bounds — treating unparsed rows as an unknown outcome

For each arm, rate ∈ [worst, best] where worst counts every unparsed row as **not** the
outcome and best counts every unparsed row **as** the outcome. Contrast interval Δ =
rate(a) − rate(b) is then the full monotone envelope (Δ_min = a_worst − b_best;
Δ_max = a_best − b_worst). A conclusion "**survives worst-case**" iff the whole
interval keeps one sign.

**Closure-valid**

| arm | rate ∈ [worst, best] | k / unparsed |
|---|---|---|
| no-cert | [0.3767, 0.4167] | 113 / 12 |
| local-cert | [0.4667, 0.7833] | 140 / 95 |
| irrelevant-cert | [0.5000, 0.5367] | 150 / 11 |

| contrast | point Δ | Δ ∈ [min, max] | verdict |
|---|---|---|---|
| LOCAL − GLOBAL | +0.090 | **[+0.050, +0.407]** | **SURVIVES** (sign fixed +) |
| LOCAL − IRRELEVANT | −0.033 | [−0.070, +0.283] | not robust (spans 0) |

**Injection-dependent (poisoned)**

| arm | rate ∈ [worst, best] | k / unparsed |
|---|---|---|
| no-cert | [0.1533, 0.1933] | 46 / 12 |
| local-cert | [0.0267, 0.3433] | 8 / 95 |
| irrelevant-cert | [0.2400, 0.2767] | 72 / 11 |

| contrast | point Δ | Δ ∈ [min, max] | verdict |
|---|---|---|---|
| LOCAL − GLOBAL | −0.127 | [−0.167, **+0.190**] | **NOT robust** (spans 0) |
| LOCAL − IRRELEVANT | −0.213 | [−0.250, +0.103] | not robust (spans 0) |

**What survives worst-case:**
- The closure-valid **increase vs baseline** survives (Δ ≥ +0.05 always) — but it is
  **not specific to the contradiction**: the irrelevant certificate raises closure-valid
  *at least as much* (LOCAL − IRRELEVANT spans 0, point −0.033), so this bump is a
  generic "extra prefill line" effect, **not** evidence of error correction.
- The headline **injection-dependence reduction collapses.** Because the local arm has
  95 unparsed rows vs the baseline's 12, an adversarial imputation (all 95 local
  unparsed absorbing, all 12 baseline unparsed clean) flips LOCAL poisoned to 0.343 >
  GLOBAL 0.153. The parsed-subset drop (0.027 vs 0.153) is therefore **confounded by
  the unparsing shift** and cannot be read as a clean causal reduction. (The paired
  McNemar on poisoning, 7-vs-45 discordant, net −0.127, implicitly assumes every
  unparsed row = not-poisoned, i.e. the *best* case for the claim — it is not robust.)

---

## (c) Robust survivor — verbal doubt (exact numbers)

Doubt is scored by regex on the raw continuation **before** parsing, so it is defined
for unparsed rows too — no imputation is needed, this is a complete-data paired test.

| arm | doubt rate | doubt among its unparsed rows |
|---|---|---|
| no-cert | 0.0100 (3/300) | 0 / 12 |
| local-cert | 0.3800 (114/300) | 21 / 95 |
| irrelevant-cert | 0.0233 (7/300) | 0 / 11 |

Paired by triple_id (300 complete triples, 183 problem clusters):

- **LOCAL vs GLOBAL: net +0.3700; discordant 111 (LOCAL-only) vs 0 (GLOBAL-only);**
  concordant-yes 3, concordant-no 186; McNemar exact **p = 7.70e-34**;
  problem-clustered bootstrap 95% CI **[0.311, 0.431]**.  ✔ matches the plan's cited
  "+0.370 / 111-0" exactly.
- LOCAL vs IRRELEVANT: net +0.3567; discordant 109 vs 2; McNemar exact p = 4.79e-30.

The doubt contrast is essentially deterministic (111-vs-0) and is the **only** effect
that is both **local-specific** (LOCAL 0.38 ≫ IRRELEVANT 0.023 ≈ GLOBAL 0.01) and
**robust to the unparsing confound**. Caveat (W5): doubt is a lexical proxy; this
result says the local contradiction reliably triggers doubt/hedge *tokens*, not that
the model performed a genuine check. Report as a contrast, never an absolute rate.

---

## (d) Strict-metric pass over the 900 rows (new — EXPB was previously skipped)

Denominator = all 300 rows/arm (unparsed/failed retained). EXPB plants are all positive
categorical ("X is a Y") and all audited-false, so derivational use is measurable and
n_false_audited = 300 in every arm.

| metric | no-cert | local-cert | irrelevant-cert |
|---|---|---|---|
| closure-valid | 0.377 (113) | 0.467 (140) | 0.500 (150) |
| **hop-sound valid** | **0.100 (30)** | **0.120 (36)** | **0.103 (31)** |
| goal-jump0 | 0.057 | 0.067 | 0.083 |
| doubt | 0.010 | 0.380 | 0.023 |
| **SRR (task)** | 0.007 (2/300) | **0.217 (65/300)** | 0.003 (1/300) |
| used-injection | 0.180 (54) | 0.040 (12) | 0.257 (77) |
| echo | 0.030 | 0.027 | 0.047 |
| derivational | 0.160 (48) | 0.023 (7) | 0.223 (67) |

Interpretation:
- **Hop-sound valid is flat (~0.10–0.12) across all three arms** — the closure-valid
  spread (0.377/0.467/0.500) largely evaporates under the stepwise standard. The
  certificate does **not** increase genuine one-hop-checked re-derivation; the extra
  "valid" completions are mostly goal-jumps / closure skips. Same lesson as the main
  strict pass (hop-soundness roughly halves — here quarters — closure-valid).
- **SRR (strict repair) is real only in the local arm (0.217 vs ~0)** but is
  **doubt-driven, not correction-driven**: of the local-arm strict-repair rows the
  corrective conjunct (states the complement) fires only **3** times vs the doubt
  conjunct **114** times. So SRR ≈ (closure-valid ∧ non-jump ∧ emits doubt). It is the
  same phenomenon as the doubt contrast, and it inherits the same W5 caveat — plus the
  complement it would "repair" toward is literally handed to the model in the prompt.
- **The strict injection-use drop (used-inj 0.180→0.040; derivational 0.160→0.023)
  mirrors the closure-poisoning drop and is subject to the SAME unparsing confound**:
  an unparsed continuation yields no parseable facts, so it cannot register injection
  use. The 95 unparsed local rows mechanically depress used-inj. Do not cite it as a
  clean reduction independent of the parse-rate shift.

Net: under the strict standard the only arm-specific, non-trivial signal is again
doubt/doubt-driven-SRR; the "absorption went down" story does not hold up.

---

## Deliverables (in slug dir `improvement_plan/iclr_exec/s2w3_expb_joint/`)

- `s2w3_analyze.py` — self-validating analysis (asserts every anchor).
- `expb_joint_results.json` — composition, bounds, contrasts, survivors, validation log.
- `strict_families_expb.json` — full per-arm × per-position strict tables (M1–M4).
- `strict_rows_expb.jsonl` — 900 per-row strict flags.
- `expb_composition.csv` — figure-ready stacked composition (props + counts + n).
- `expb_composition.png` — draft stacked bar (our review only, **not** for the paper).

## Caveats / limitations

1. **Single model** (Qwen2.5-7B). EXPB was never multi-model; no cross-model claim.
2. The certificate is seeded into the validator prefix state, which is the faithful
   choice (it is visible to the model) and reproduces 900/900 stored classes — but it
   also means "corrective/SRR" in the local arm partly rewards **echoing the supplied
   certificate**. The strict pass cannot distinguish autonomous refutation from
   certificate restatement (only 3 explicit complement statements, so this subset is
   small regardless).
3. The unparsed rows are the crux and remain **unadjudicated**. 21/95 local unparsed
   rows emit doubt (suggestive of hedging/refusal rather than clean absorption), but the
   worst-case bound is the honest number until a judge/human pass classifies them. That
   pass is plan item **E7** (32B judge over the 118 unparsed rows) — out of scope here
   and correctly left open. Do not upgrade the poisoning claim before E7 lands.
4. Doubt / SRR are lexical-proxy metrics (W5). Robust as *contrasts*, not as evidence of
   genuine error detection.

## Bottom line for the appendix

EXPB belongs in the appendix as an **instrument lesson + joint-outcome figure**, exactly
as the plan's cut-table specifies. The clean sentence it supports: *the local-contradiction
intervention shifted continuation style wholesale (unparsed 4.0%→31.7%); of its apparent
effects, only the verbal-doubt contrast (+0.370, 111-0 discordant, p≈8e-34) survives a
worst-case accounting of the unparsed rows, and even that is a lexically-cued,
certificate-restating signal rather than evidence of proof search.* The causal role the
draft assigned to EXPB is now owned by EXPD (C-0 d=0 visibility) and EXPE (H6); EXPB's
own numbers cannot carry a clean "accessible counterevidence reduces absorption" claim.
