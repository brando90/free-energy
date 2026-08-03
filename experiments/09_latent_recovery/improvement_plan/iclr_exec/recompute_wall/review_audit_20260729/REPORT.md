# Adversarial validity review — recompute-wall paper
**Date:** 2026-07-29 · **Reviewer:** Claude (three-pass adversarial review)
**Repo:** /lfs/skampere2/0/eobbad/free-energy @ c38b4c3 · **Paper:** paper_latex/papers/recompute_wall (main.pdf 2026-07-29 10:59Z)
**Scratch (all fixture scripts + captured outputs):** /lfs/skampere2/0/eobbad/scratch/review_2026-07-29/
*(~/scratch was unusable — AFS quota exceeded — so scratch lives on /lfs as noted in LOG.md.)*

**Method note.** Review ran as 8 mapping agents + 3 independent re-derivation agents + fixture attacks, then
inline verification. A billing outage killed some agents mid-run; their executed fixture outputs survived in
scratch and every load-bearing claim below was re-verified inline (verify_main/verify_main2.out). Every
executed check cites its script. Claims labeled CONFIRMED have an executed repro; SUSPECTED are read-only.

---

## 1. Kill shots

**K1 — fig3/fig4 haiku ships the superseded R=3 pilot while the appendix asserts the registered R=8 rerun
is what is reported; the two runs disagree beyond CI on one cell. CONFIRMED.**
verified_numbers.json fig3/fig4 haiku blocks are byte-identical to E11_haiku (R=3+greedy, n=240/cell).
A_appendix.tex:9–14 says frontier cells are "reported at that registered depth" (R=8, n=540) and that the R=3
runs "reproduced every cell within its confidence interval." E11_haiku_r8 exists (finished 17:21Z, ~3.5 h
*after* the figures were built at 13:50Z) and gives k0 = 0.0185 vs 0.0125 (so the intro's "1.3 %" and the
abstract's "1 percent" become ≈2 %), fig4 full = 0.9944 vs 1.0000, and fig4 partial = 0.9759 [0.9593,0.9859]
vs 0.9958 [0.9768,0.9993] — outside each other's CIs in both directions, falsifying the consistency sentence.
Repro: verify_main/verify_main2.out §V2; rederive_fig34/rederive_fig34.py PART 3.
Fix is analysis-level: repoint registry+figures at E11_haiku_r8 (data on disk) or rewrite the appendix.

**K2 — Section 7's two headline numbers do not mean what the prose says. CONFIRMED.**
(a) "in 85 percent … no re-derivation appears": arm_a.py:224/275 silently drops every reasoning-ON row whose
summary came back empty — 38–45 % of ON rows (deep: 108/240) — and the exclusion is outcome-correlated
(excluded rows absorb at 0.972 vs 0.795 included). Counting empty channels as never-rederive gives 91.7 %.
The paper discloses neither the exclusion nor the subsample. Repro: map_arm/arm_a_summary_exclusion.{py,out}.
(b) "the final answer still builds on the planted value in 35 to 68 percent": 35 % is ARM A *strict* deep
(7/20); 68 % is ARM B *loose* deep (21/31). The consistent alternatives are A-loose 54.2 %, B-strict 22.2 %
(below the quoted floor). Worse, the loose signal is confounded: re-executing the cf world shows 11/19 of the
deep loose-only rows (28/30 at onehop) fire on values the *plant world itself* produces downstream, on line
numbers, or on prefix text — zero re-derivation required. Cleaned deep endpoint ≈47 %.
Repro: map_arm/loose_fp_fixture2.{py,out}; attack_arm/scan_attack.out PART 4.

**K3 — Your own E13 matched controls (finished 16:37Z today, after the PDF) fire their pre-registered
"reframe" branches against Sections 4 and 5. CONFIRMED (numbers), decision pending (yours).**
line_k0 (zero ops, k1 geometry) absorbs at 0.9821 (llama) / 0.7867 (qwen) — ≈ their k1 rates — so the
open-model k0→k1 "sharp jump" in fig3 is construction/format-driven, exactly the branch E13_MATCHED_CONTROLS.md:12
labels "forces a major reframe." Haiku keeps a real step (line_k0 0.1375 vs k1 ≈ 1.0) but its floor is 4×
the shipped anchor (0.0333). The truth-printed full arm (k1_full_sum) collapses absorption to 0.5708 / 0.4722
/ 0.3796, so Sec 5's "deference, not inability" survives only in weakened, two-sided form; and the E13a grid
shows haiku absorbing the false line at 0.4458 when the true value precedes it as a note — fig2's readable
series sits in the easiest corner of the format/order grid. The shipped paper cites none of this.
Repro: verify_main2.out §V4 (all cells re-read from e13/results/*/summary_tables.json).

**K4 — Headline prose is contradicted by the paper's own registry/artifacts in six places. CONFIRMED.**
Abstract + intro + conclusion: "computed falsehoods are absorbed in 95 to 100 percent … by every model
tested" — registry computed rates are 0.50 (qwen1p5b, n=30), 0.7273 (olmo7b, n=66), 0.88 (qwen7b, n=150).
03_wall.tex:6: "absorption is 1.00 for the four frontier Anthropic models" — sonnet-4-5 is 0.9722; same
line: five-op cell "0.95 or above throughout" — olmo7b deep_kc5 = 0.8667 (13/15), qwen1p5b n=1.
05_opacity.tex:8: haiku "stays at 1.00 across bare, full, partial at every depth" — registry partial 0.9958
(r8: 0.9759). 06_fixes.tex:10: mechanical scan "fires in under a tenth … regardless of condition" — A0
k1-bare = 0.1167. 07_reasoning.tex:8: "0.95 … 0.93" — artifacts give B1 0.975/0.954 (pooled 0.965) and B2
0.925; :6 "0.06 … matching Section 3" — Section 3's gpt-5.1 readable is 0.0857 from a different run/endpoint.
Repro: verify_main.out §V1 (verbatim quotes), verify_main2.out §V5/V7/V8.

**K5 — fig3/fig4 inference is at the wrong unit, and the contrast claims that motivated registration-level
cluster inference become marginal under it. CONFIRMED.**
Every cell pools 9 rollouts/program (4 for haiku) and ships rollout-level Wilson bands; the artifacts store
program-cluster bootstrap CIs alongside (E11 registration §5: "All CIs are cluster-bootstrap (cluster =
program family); Wilson for single rates" — single-rate Wilson is defensible, contrast claims are not).
Under the stored cluster_boot95: qwen k0 [0.6652,0.7548] vs k1 [0.7296,0.8319] **overlap** (Sec 4 presents
the ramp as real); llama k5-vs-k3 drop not significant ([-0.098,+0.006]); qwen full-minus-bare survives
(+0.152 [0.106,0.201]). CI widths are 1.8–2.5× Wilson on non-saturated cells.
Repro: verify_main2.out §V3; rederive_fig34.py PART 4 + rederive_fig34_round2.py §E.

---

## 2. Data path summary (Pass 1)

1. **Worlds.** expg/gen_programs.py builds "regime2" programs (L=12, planted site line 6) for fig1/fig2;
   e11_generator/gen_depth.py builds E11/E12 worlds (L=20, 14 ops, site line 10, k ∈ {0,1,2,3,5}, arms
   bare/full/partial). Seeded, audited fail-closed (audit_depth.py; 104/104 tests re-executed).
2. **Gold gate.** Each model first solves the unperturbed program; inject.gold_solve_eval keeps only exactly-
   solved traces → per-model cohorts (this is why every fig2/fig3 n differs by model).
3. **Injection.** inject.build_injection (fig2) / build_injection_e11 (fig3/4) splices the planted false
   value over the model's *own* gold line; rejects are fail-closed and logged.
4. **Generation.** Open models: vLLM token-id prefill, greedy + R sampled @T=0.7 (ran on skampere1, synced).
   Claude: Anthropic prefill (haiku/sonnet-4-5) or user-turn bridge (opus/sonnet-5). GPT: bridge; gpt-3.5:
   completions. gpt-5.1/opus/sonnet-5 run at provider-default temperature (param rejected) — roster asymmetry.
5. **Grading.** One shared validator (expg/validate.py classify_run) re-executes the program in the true and
   counterfactual worlds and compares the continuation's parsed integer claims — same interpreter generates
   references and grades (interp == CPython verified on all generated grammar).
6. **DV forks.** fig2 GPT = absorbed AND NOT doubt-lexicon; fig2 Claude/open = plain absorbed (fork verified
   numerically inert today: absorbed∧flagged = 0 in every fig2 cell). fig3/4 = flag-first three_way_label
   plus a fourth, undisclosed **unresolved** bucket kept in denominators (up to 21 % of a cell).
7. **Aggregation.** Wilson + stored (unused) cluster bootstrap → summary_tables.json / topup_analysis.json /
   bridge_analysis.json → hand-copied registry results/verified_numbers.json → fig2/3/4. Sections 2, 6, 7 and
   the appendix numbers bypass the registry entirely (trace to reports).
   No `% source:` comments exist in the tex; the registry `_sources` block is the only trace, and its fig1
   paths point at a directory that lives elsewhere (experiments/09_latent_recovery/results/).

---

## 3. Findings

### P1 — a reported number is wrong or does not mean what the paper says (analysis re-run required)

**F1. Haiku R=3/R=8 provenance inversion.** See K1. `paper_latex/.../A_appendix.tex:9` +
`results/verified_numbers.json` fig3/fig4 haiku. CONFIRMED. Blast: fig3 haiku curve, fig4 haiku bars,
abstract/intro/caption "1 %"→"1.3 %", appendix consistency sentence. Direction: k0 understated 0.0125→0.0185;
k1-partial overstated 0.9958→0.9759. Conclusions (haiku ≈ ceiling) unchanged.

**F2. Sec 7 taxonomy computed on an outcome-correlated 55 % subsample.** See K2a. `reasoning_arm/arm_a.py:224,275`.
CONFIRMED. Blast: "85 percent", every conditional-deference stat downstream of `rederive_bucket`.
Direction: number rises to 0.917 under count-empty-as-never; the qualitative claim gets *stronger*, the
printed denominator is wrong either way.

**F3. "35 to 68 percent" mixes strict/loose across arms and the loose signal is confounded.** See K2b.
`reasoning_arm/rederive_scan.py:69–71`, `07_reasoning.tex:10`. CONFIRMED. Blast: "the sharpest evidence in
the paper that the failure is deference." Honest ranges: strict 22–35 % (n=9/20), cleaned-loose ≈47–54 %.

**F4. Prose-vs-registry drift (six instances).** See K4. CONFIRMED. Blast: abstract, intro, conclusion,
03_wall:6/8, 05_opacity:8, 06_fixes:10, 07_reasoning:6/8. All fixable in text; none require new data.

**F5. Nested-rollout Wilson CIs; contrasts marginal under stored cluster bootstrap.** See K5.
`e11_run/e11_run.py:587`, registry wlo/whi fields. CONFIRMED. Blast: every fig3/fig4 error band; Sec 4 qwen
ramp sentence; "never falls again" (also see F6). Rates and n unaffected.

### P2 — robustness threat or surviving alternative explanation

**F6. fig3 k=5 cells measured on an order-of-magnitude more selected cohort, against the registered rule.**
Gold-solvability gate keeps 3.11 % of llama programs (56) and 1.22 % of qwen (22) at k=5 vs 9–24 % at k≤3 —
selection intensity depends directly on the fig3 x-axis. qwen k5 carries the pipeline's own
`INVALID_underpowered` stamp (program_n=22 < 50; E11_DEPTH_OPACITY.md §6.2 registers "reported, not scored";
§ line 65 registers truncation "at the deepest viable k"), yet fig3 plots it as a settled point and
04_depth.tex:10's "never falls again at greater depth" leans on it — while llama k5 *does* fall below k2/k3
with disjoint Wilson CIs (0.9464 vs 0.9970). CONFIRMED: verify_main2.out §V3; rederive_fig34.py PART 5;
queue_status.json gold_cohort. Mitigation found: identical program sets across opacity arms, so fig4 within-k
contrasts are balanced. Direction at k5: unknowable from record (no data for unsolved programs).

**F7. k=0 anchor is a different construct, and E13 shows the open-model step is construction-driven.**
The k=0 arm differs from k≥1 in four bundled ways (off-format "note:" plant vs in-format completed line;
model must rewrite the site line vs handed it; root var contradicted adjacently vs site value contradicted
nowhere; digit-swap delta vs last-digit-preserving ±10/20). E13's line_k0 (in-format, zero ops) absorbs at
0.9821/0.7867 (llama/qwen) ≈ their k1 rates → their fig3 step is format, not verification cost. Haiku:
line_k0 0.1375 vs anchor 0.0333 — step survives, floor 4×. `e11_run.py:150`, `gen_depth.py:539–566`,
e13/results/*. CONFIRMED (V4). Blast: Sec 4 open-model narrative, abstract "step function" scope (frontier
claim survives), fig3 k0 points' interpretation.

**F8. Sec 5 "deference, not inability" and fig2's readable series, against E13b/E13a.**
Truth-printed full arm collapses absorption to 0.5708/0.4722/0.3796 (haiku/llama/qwen) vs ≈1.0 shipped full
arm — the shipped full arm never prints the true sum, so "operands shown" ≠ "truth visible." Grid: haiku
absorbs the false line at 0.4458 when the true value precedes it as a note (0.0083 in the shipped corner);
qwen 0.9326 — readable-catching is substantially format/order-following. CONFIRMED (V4). Blast: Sec 5
verbatim claims; fig2 "readable" construct label; Discussion framing. Partial defense: E13 dose cells show
frontier readable-catching robust to delta style (haiku dose_off1 = 0.0000, dose_off10_20 = 0.0042), so the
fig2 frontier wall itself stands.

**F9. Undisclosed fourth outcome ("unresolved") with condition-correlated mass shapes the qwen ramp.**
02_setup.tex:16 claims exactly three arithmetic-decided outcomes; the pipeline has four, and unresolved
(no gradeable output, kept in denominator as not-absorbed) is 8.3 % of qwen k0, **21.0 % of qwen k1**, ~5–7 %
at k2/k3. Excluding it, qwen goes 0.777→0.9925 at k0→k1 — a step, not a ramp; llama likewise 0.663→1.000.
So the "Qwen ramps more gradually" contrast (04_depth.tex:10) is substantially an output-emission artifact.
CONFIRMED: inline recount (this review; verify_main dir), labels from validated_outputs.jsonl.
Direction: absorbed understated wherever unresolved is high — conservative for the wall claim, but the
cross-model shape story changes under the alternative convention.

**F10. fig2 "computed" series is a registration deviation.** E10_WIDEN.md:30 registers recompute-side
absorption POOLED over opfree_kr1/kr8 + onehop_kc1 + deep_kc5; the paper ships onehop_kc1 only, with opfree
reclassified post-hoc as a "copy" cell (disclosed as a definition, never as a deviation). Under the
registered pooling, prefill-haiku computed ≈0.65, not 1.00. The onehop-only choice is scientifically
defensible (opfree is contaminated by bridge-mode copy inflation — bridge inflates haiku opfree ~10×,
25/70 vs 3/70, CONFIRMED from validated files) but must be disclosed as a deviation. SUSPECTED→CONFIRMED
(doc text V6; rates from topup/bridge analysis JSONs).

**F11. Registered n<50 INVALID rule breached by the roster.** E10_WIDEN.md:66: "n<50 … INVALID-underpowered
(reported, not scored, counts as 'pair not established')." fig2 plots qwen1p5b computed at n=30 as a scored
point and every "thirteen models"/"every model" sentence counts it; olmo (66) and the four GPT rows (58–70)
sit in the registered CI-widened band with only qwen1.5b/olmo caveated in text. CONFIRMED (V6, V7).

**F12. fig2 decoding asymmetry.** opus-4-8, sonnet-5, gpt-5.1 (and fable) ran at provider-default
temperature with a single sample and no seed (API rejects the param; api_gen.py:44–66); the other nine
models are greedy/T=0. Their readable-side near-zeros are one-draw estimates from a different sampling
distribution. CONFIRMED (code + run records). Blast: fig2 per-model comparability; not the computed-side
1.00s (temp-robust at ceiling). Needs a caption/appendix disclosure table at minimum.

**F13. E12 registration deviations + provenance hole.** Registered shared-cohort within-world pairing
(E12_PROMPTED.md:44) vs shipped per-arm gold regeneration with per-arm kept-60 gates (four-way intersection
at qwen k5 = 5 programs); registered verify-instruction bytes replaced with different A2/A3 wording, no
addendum; E12_open (Sec 6 llama/qwen rows) runner source absent from skampere2 and gold generations never
logged; **no prompt text persisted in any raw_generations file in the whole program** (arm audit:
any_prompt_field=false), so instruction bytes are re-derivable only from code. CONFIRMED (prior-audit
arm_overlap_audit re-read + V6). Mitigation: the prior audit's intersection reanalysis moves no rate
materially — but none of this is in 06_fixes.tex.

**F14. fig2 readable-vs-computed compares disjoint programs and different delta policies.**
kr1 (readable) and kc1 (computed) are different program sets (overlap = 0, by design: copy site vs binop
site) and different deltas (digit_swap, |Δ| median 27 vs last-digit-preserving ±10/20, median 10). The
within-model contrast bundles perturbation style with checkability. E13 dose cells bound the delta part at
the frontier (see F8) but not for open models (llama dose_off1 0.688 vs adjacent 0.661 — fine; qwen
dose_off10_20 0.759 vs adjacent 0.690 — modest inflation). CONFIRMED: rederive_fig2 item-set audit +
verify_main2 V4. Blast: interpretation of fig2's within-model gap for open models; frontier gap robust.

### P3 — latent or bounded; current numbers unaffected

- **Haiku perturbed continuations not archived** (E11_haiku*/raw_generations = 900 gold rows only) — all
  haiku fig3/fig4 grades unauditable from raw text. CONFIRMED. No error found in re-aggregation.
- **Sonnet-5 n=178:** two arithmetically-correct golds dropped on a claim-shaped scratch preamble
  ("parse"); pre-injection, condition-neutral; bound if retained [0.9889,1.0]. CONFIRMED (rederive_pool/forensics.py).
- **gpt-3.5 n=68/58:** nine 429 RateLimitErrors, never retried (resume treats failed run_ids as done —
  permanent silent loss mechanism), spread ~uniformly over 5 cells. CONFIRMED (attack_e10/audit_e10_followup.out §5).
- **Doubt-lexicon scanned on full text incl. post-output chatter while claim scan stops at first output**
  — flag-first would reclassify absorbed→flagged (E11: ≤22 llama rows, 0 qwen; fig2: absorbed∧flagged=0
  everywhere, so inert today). CONFIRMED (map_expg fixture E3; fx2 post_output_doubt).
- **Grader edge semantics** (all CONFIRMED via attack_expg/fx2_hostile_grader.out): truncation/refusal/
  no-output → non-absorbed but kept in denominator; format-variant correct answers ("100.", "1e2-style",
  bare "output = N") fail to parse → non-absorbed (lower-bounds absorption; realized no_output=0 in all
  fig2 API cells); output claim matched on value only, not variable (out_wrong_var_cf absorbs — latent);
  statement text never checked (wrong_stmt_right_values counts trace-valid — latent);
  post-print self-correction not credited (dup_output_cf_then_true stays absorbed — scan-stop convention).
- **arm_b think_text stored truncated at 4000 chars while scan ran on full text** — 29 cf rows not
  reproducible from record; rescan flips 4 buckets. CONFIRMED (attack_arm/scan_attack.out PART 5).
- **arm_b stop strings fire inside <think>** (8/15 truncations mid-deliberation ≤3.7k chars). CONFIRMED.
- **Appendix protocol-constants paragraph describes the wrong substrate for fig1/fig2** (claims L=20/site 10
  "fixed across cells"; regime2 is L=12/site 6) and the sampling paragraph's "R=6 … 700 rollouts over 100
  programs" belongs to E12_open, not the fig3 arms (1350/150/R=8+greedy). CONFIRMED (V1 + artifacts).
- **Registry coverage:** verified_numbers.json covers only fig2/3/4; every Sec 2/6/7 and appendix number
  bypasses it; `_sources` fig1 paths have the wrong root. CONFIRMED.
- **e12_api.py:401–404** `except Exception: continue` would silently drop a cont row on resume (unexercised).
  **e11_api.py:151** drops injection rejects with no audit row (recount proves zero occurred). SUSPECTED-latent.
- **Budget governance docs contradict each other** (BUDGET.md $500 ceiling vs BUDGET_NOTE.md $100/$150; E13
  ledger figures disagree). CONFIRMED. No paper number affected.
- **E13_inc/A4** (inconsistency-instruction arm: k0 0.0208, k1 1.0, k5 1.0) exists and is unreported —
  it *supports* Sec 6; free win. CONFIRMED (V4).

### Killed findings (investigated and refuted)

- **661 "force_after_line anomalies" in e10_widen manifests** (attack fx3 §F): all 661 are the entire
  adjacent_contradiction cell with force_after_line = site_line−1 uniformly — a design constant of the
  readable family, not corruption. Closure check: re-executed interp on all 661 rows — out_true ≠ out_cf on
  every row, recomputed out_true matches stored on every row. REFUTED (this review, inline).
- **DV-definition fork across fig2 columns** (GPT flag-subtracted vs Claude/open plain): recomputed both
  conventions over every archived validated row — absorbed∧flagged = 0 in all fig2 cells; zero numeric
  effect today. Latent-only (P3 if reruns ever produce flagged+absorbed rows).
- **Stale-code risk from untracked expg/**: compiled __pycache__ bytecode matches a fresh py_compile of the
  current source for every module (fx5_pyc_provenance.out), and full re-validation of all 4092 archived rows
  reproduces every archived class — the artifacts were produced by the code on disk. REFUTED.
- **Truncation confound (EXPG_LLM_MAX_MODEL_LEN)**: effective caps traced per run (8192 where set); zero
  length-capped rows in any scored open cell (one 'length' row total, in a control cell); all API
  stop_reasons end_turn/stop except the nine gpt-3.5 429s. No realized truncation anywhere. REFUTED as a
  confound for these runs (latent if worlds grow).

---

## 4. Number-by-number verdict table

Re-derivations used independent code (own interpreter, own parser, own Wilson) from per-sample raw records
unless noted. "MATCH" = agreement to 4 decimals including n and CI.

| Number | Paper/registry | Re-derived | Verdict | Note |
|---|---|---|---|---|
| fig2: all 13 models × readable/computed (26 cells) | verified_numbers fig2 | identical; **0 row-level disagreements** (two independent regrades: rederive_fig2, rederive_pool; open cells also fx1 full re-validation) | **MATCH** | raw→grade→rate reproduced end-to-end |
| fig2 odd n's: 61/30 (qwen1p5b), 66 (olmo), 68/58 (gpt-3.5), 178 (sonnet-5), 190/180 (Claude pools) | — | every exclusion accounted to program-id + logged reason | **MATCH** | see F11 (n=30 plotted vs registered rule), P3 items |
| fig3 llama/qwen: all 10 bare cells | fig3 block | identical; independent raw-text regrade row-agreement 99.93–100 % | **MATCH** | k5 cohort caveat = F6 |
| fig3 haiku: 5 cells | 0.0125, 1.0×4 (n=240) | re-aggregation matches R=3 run; **R=8 registered run: k0 0.0185** | **MATCH vs R3 / MISMATCH vs stated provenance** | K1 |
| fig4: 9 cells + inset lines | fig4 block | identical (inset verbatim in manifest, 1 hit/arm) | **MATCH** | haiku full/partial differ in R=8: 0.9944/0.9759 |
| fig3/fig4 CI bands (wlo/whi) | Wilson over rollouts | cluster-boot 1.8–2.5× wider; qwen k0/k1 overlap | **MISMATCH (unit)** | K5 |
| Sec 3 prose: "1.00 four frontier Anthropic" | 1.00 | sonnet-4-5 = 0.9722 | **MISMATCH** | K4 |
| Sec 3 prose: five-op "0.95+ throughout" | ≥0.95 | olmo 0.8667 (13/15), qwen1p5b n=1 | **MISMATCH** | K4 |
| Abstract/intro/concl "95–100 % by every model" | 95–100 | 0.50 / 0.727 / 0.88 for 3 of 13 | **MISMATCH** | K4 |
| Sec 6: haiku 1.00 all arms, ~160 tok; llama 0.99; qwen 0.79–0.81; k0 drift 0.75→0.86 | e12_fixes_r8 + E12_open | all reproduce | **MATCH** | provenance caveat F13 |
| Sec 6: scan "under a tenth regardless of condition" | <0.10 | A0 k1 = 0.1167 | **MISMATCH** | K4 |
| Sec 7: B1/B2 "0.95 / 0.93" | 0.95/0.93 | 0.975 & 0.954 (pooled 0.965) / 0.925 | **MISMATCH (rounding-by-narrative)** | K4 |
| Sec 7: "0.06 … matching Section 3" | 0.06 | arm run 0.061; fig2 row 0.0857 (different endpoint) | **MISMATCH (conflation)** | K4 |
| Sec 7: "85 % never re-derive" | 0.848 | on 55 % outcome-correlated subsample; 0.917 alt | **MISMATCH (meaning)** | K2 |
| Sec 7: "35 to 68 %" | 35–68 | strict 22–35; cleaned loose ≈47–54 | **MISMATCH (selection)** | K2 |
| Sec 2 gold-failure taxonomy (853/5 etc.), solve rates | 02_setup:14 | cohort.json matches exactly | **MATCH** | |
| fig1 worked examples | kr1_507786_00100000 / kc1_…00300000 | exist; continuations contain quoted outputs | **MATCH** | dir root differs from `_sources` |
| Fable-5 "unmeasured, 360/360 refusals, $2.93" | appendix | raw rows confirm 360/360; $ matches | **MATCH** | category "cyber"/probe outcomes report-only |
| Appendix sampling paragraph (R=8 reported; R=6/700/100 open) | — | contradicts every artifact | **MISMATCH** | K1 |

**Staleness:** analysis JSONs postdate their validated inputs; registry (07-28) matches current artifacts;
recomputation of current data reproduces every artifact value; pyc bytecode == current untracked source.
The single stale object is the *paper text* relative to E11_haiku_r8 and E13 (both newer than the PDF).

---

## 5. What I could not verify

- **Haiku grading from raw text** — perturbed continuations not archived (P3 above). Would need a re-run or
  acceptance of graded rows on faith.
- **ARM B / E11-open / E12_open runtime provenance** — executed on skampere1 (E12_open runner source absent
  even there per the prior audit's search of this node). Byte-identity of the runtime expg copies is
  asserted by sync receipts, not provable from skampere2. You told me to ignore skampere1; a 10-minute diff
  there would close this.
- **Exact API payload bytes** — prompts are rebuilt deterministically but not persisted anywhere in the
  program (arm audit: any_prompt_field=false). parity_check.json supports byte-parity; SUSPECTED-clean.
- **Fable stop category "cyber" + 4 mitigation-probe outcomes** — report-asserted, not persisted; re-check
  needs paid API calls.
- **Whether the 2 gpt-3.5 429 rows would absorb if retried** — needs API calls; bounded and immaterial.
- **Direction of the k5 selection bias (F6)** — unknowable from the record (no continuations for unsolved
  programs).
- **Mutation-testing of audit_depth.py's tests** — the agent assigned died before running it; the 104
  positive+negative assertions were re-executed and pass, but I did not demonstrate they fail under
  deliberately broken audit logic. Decorative-test risk unquantified.
- **Temperature-rejection claim** for opus/sonnet-5/gpt-5.1 (the stated reason for F12's asymmetry) —
  asserted in api_gen.py comments; verifying costs one API call per model.

**Questions for you:**
1. Which haiku run is canonical — R=3 pilot (shipped) or R=8 registered rerun? Everything in K1 follows.
2. Is E13 in-scope for this submission? Its results force rewrites of Secs 4/5 either way (K3) — using it is
   strictly better than a reviewer finding it.
3. Was the onehop-only "computed" definition (F10) decided before or after seeing the pooled numbers?
   The answer determines whether it's a disclosed-deviation footnote or a bigger problem.

---

## 6. Final call

| Experiment | Verdict | Reason |
|---|---|---|
| **e10 (fig2, Sec 3)** | **VALID-WITH-CAVEAT** | Every cell reproduces end-to-end from raw records with zero row disagreements; caveats are prose drift (K4), registered-rule deviations (F10, F11), decoding asymmetry (F12), and construct scope (F14) — all text/disclosure fixes, no data re-run. |
| **e11 (fig3/fig4, Secs 4–5)** | **NEEDS-PARTIAL-RERUN** (analysis-level; no new generation) | Haiku cells must be repointed to the registered R=8 run or the appendix rewritten (K1); contrast claims must move to the stored cluster CIs (K5); qwen k5 must be flagged/dropped per your own registered rule (F6); the unresolved bucket disclosed (F9); Sec 4/5 claims re-scoped against E13 (K3, F7, F8). |
| **e12 (Sec 6)** | **VALID-WITH-CAVEAT** | Numbers reproduce from e12_fixes_r8/E12_open; "under a tenth" is false at baseline; registration deviations and the E12_open provenance hole need disclosure (F13); an optional logged re-run of E12_open would close provenance cheaply. |
| **reasoning_arm (Sec 7)** | **NEEDS-PARTIAL-RERUN** (analysis-level) | Both headline taxonomy numbers misdescribe their denominators/signals (K2); recompute from stored records with declared conventions and rewrite; 29 capped-think rows are unreproducible from the record and should be excluded or re-scanned from re-generated traces. |

**Bottom line.** I found no fabrication and no grading bug that moves a plotted rate: the fig2/3/4 pipeline
reproduces exactly from raw artifacts under two independent reimplementations, and the frontier wall result
(readable ≈0, computed ≈1, switch at one operation) survives every attack I ran, including E13's own dose
and format controls. What does not survive unedited: the haiku run provenance, the fig3/fig4 inference unit,
the open-model depth-step interpretation, Section 7's two headline percentages, and six prose claims that
your own registry contradicts. All are fixable from data already on disk; nothing requires new GPU or API
spend except the optional provenance closures listed above.
