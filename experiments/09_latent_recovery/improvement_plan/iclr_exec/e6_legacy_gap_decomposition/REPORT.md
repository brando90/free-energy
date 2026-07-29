# E6 — Legacy-gap decomposition by measured refutation distance (reviewer Q1)

Generated: 2026-07-22. Zero GPU; analysis of stored rows only.
Location: `$EXP/improvement_plan/iclr_exec/e6_legacy_gap_decomposition/` (script `e6_decompose.py`, numbers `q1_decomposition.json`, this report).
Local mirror: `scratchpad/latent_recovery_review/exec_reports/e6_legacy_gap_decomposition_REPORT.md`.

## What this is

The reviewer's Q1: *"How much of the difference between one-hop and global falsehoods remains when comparing only positive categorical claims for which injection-dependence is mechanically measurable in the same way?"* This item decomposes the legacy EXPA one-hop vs global contrast (Qwen2.5-7B-Instruct, the draft's headline) by **measured** refutation distance d, recomputing all four DVs — permissive (closure-valid, doubt) and strict (hop-sound valid, SRR-task) — from one identical row-level pipeline for both cells, then re-states the contrast with the d=0-contaminated rows excluded, with cluster-bootstrap 95% CIs (problem_id clusters).

## Data & join integrity

- Inputs (read-only): `improvement_plan/stage0_distance/row_level_distances.jsonl` (measured d, problem_id; RETRO audit) joined on `run_id` to `improvement_plan/stage0_strict/strict_rows.jsonl` (strict + permissive DVs).
- 900/900 target rows joined (450 one_hop_falsehood + 450 global_falsehood); run_id unique within the EXPA subset (1800/1800); 0 family, 0 position, 0 doubt-flag disagreements between the two files; all 900 rows audited-false plants.
- Both cells cover the **same 150 problems** (3 positions x 150), so contrasts use a **paired** cluster bootstrap: resample the 150 problem_ids with replacement, recompute both arms' rates within each replicate (10,000 reps, seed 20260722, percentile CIs).

## (a) Reproduction anchors — 14/14 exact matches

Pooled recomputed vs published (`STRICT_METRICS_REPORT.md` pooled tables; RETRO addendum):

| anchor | recomputed | published |
|---|---|---|
| one-hop closure-valid | 0.6711 (302/450) | 0.671 (302/450) |
| one-hop hop-sound | 0.3000 (135/450) | 0.300 (135/450) |
| one-hop doubt | 0.3356 | 0.336 |
| one-hop SRR-task | 0.2511 (113/450) | 0.251 (113/450) |
| global closure-valid | 0.3933 (177/450) | 0.393 (177/450) |
| global hop-sound | 0.2133 (96/450) | 0.213 (96/450) |
| global doubt | 0.0133 | 0.013 |
| global SRR-task | 0.0022 (1/450) | 0.002 (1/450) |
| RETRO addendum d=0: n=67, doubt 0.284, closure 0.552 | all exact | RETRO_DISTANCE_REPORT addendum |
| RETRO addendum d=1: n=345, doubt 0.368, closure 0.690 | all exact | RETRO_DISTANCE_REPORT addendum |

The script hard-fails before bootstrapping if any anchor mismatches; none did.

## (b) Re-binned by measured d (pooled; cluster-bootstrap 95% CI)

d bins 0/1/2+ decompose the designed-d=1 one-hop cell (76.7% truly d=1, 14.9% d=0, 8.4% d>=2 per RETRO audit); "unreachable" is the global cell (450/450 measure d=inf, 0 contamination).

| measured d | n (problems) | closure-valid | doubt | hop-sound valid | SRR-task |
|---|---|---|---|---|---|
| 0 | 67 (58) | 0.552 [0.427, 0.676] | 0.284 [0.179, 0.397] | 0.254 [0.147, 0.370] | 0.224 [0.127, 0.328] |
| 1 | 345 (148) | 0.690 [0.637, 0.742] | 0.368 [0.315, 0.420] | 0.333 [0.277, 0.390] | 0.261 [0.213, 0.311] |
| 2+ | 38 (37) | 0.711 [0.564, 0.846] | 0.132 [0.027, 0.243] | 0.079 [0.000, 0.175] | 0.211 [0.081, 0.351] |
| unreachable (global) | 450 (150) | 0.393 [0.342, 0.447] | 0.013 [0.004, 0.024] | 0.213 [0.173, 0.256] | 0.002 [0.000, 0.007] |

Notes: d=0 rows have *lower* doubt than d=1 (0.284 vs 0.368), reproducing the RETRO non-monotonicity; the d=2+ bin is small (n=38) and its doubt/hop-sound already collapse toward the global cell.

## (c) THE Q1-ANSWER TABLE — one-hop vs global gap under identical measurement

Gap = one-hop minus global; paired cluster-bootstrap 95% CI in brackets. All CIs exclude zero.

| one-hop subset | n | closure-valid gap (permissive) | doubt gap (permissive) | hop-sound gap (strict) | SRR-task gap (strict) |
|---|---|---|---|---|---|
| Full cell (as published, incl. 14.9% d=0) | 450 | +0.278 [+0.211, +0.344] | +0.322 [+0.273, +0.373] | +0.087 [+0.033, +0.140] | +0.249 [+0.209, +0.291] |
| **Excluding d=0-contaminated rows (d>=1)** | 383 | **+0.299 [+0.231, +0.366]** | **+0.331 [+0.279, +0.385]** | **+0.095 [+0.037, +0.152]** | **+0.254 [+0.209, +0.300]** |
| Strictly d=1 only (purest designed contrast) | 345 | +0.297 [+0.226, +0.365] | +0.355 [+0.300, +0.410] | +0.120 [+0.059, +0.181] | +0.259 [+0.211, +0.307] |

**Plan expectation verified, not assumed:** excluding d=0 rows KEEPS or WIDENS the gap on every DV (doubt +32.2pp -> +33.1pp; closure +27.8pp -> +29.9pp; hop-sound +8.7pp -> +9.5pp; SRR +24.9pp -> +25.4pp), because d=0 rows are *low*-doubt (0.284 vs 0.368) and low-closure (0.552 vs 0.690). The "manufactured signal" account therefore cannot rest on d=0 contamination; it rests on the EXPE lexical-priming leg.

Per-position supplement (excl-d=0, point estimates) is in `q1_decomposition.json` (`excl_d0_by_position`): the closure gap is largest late (+36.4pp) and smallest early (+22.3pp); the doubt gap is largest early (+47.3pp); the hop-sound gap is near zero early (+2.6pp) and ~+12-14pp mid/late.

## Caveats (must accompany any use of the table)

1. **Identical measurement != matched claims.** This decomposition equalizes the DV pipeline and removes d mis-binning; it does NOT remove the polarity/syntax/lexical confound (one-hop plants are negations, global plants positive categoricals). That leg is answered by EXPD (H2' TOST polarity equivalence PASS; d-gradient dead under matching) and EXPE (lexical cueing), not here.
2. **Structural SRR asymmetry.** The global cell's SRR ~0 is partly by construction: the complement of a CWA-only falsehood is itself underivable, so an explicit corrective conjunct breaks closure-validity (STRICT_METRICS_REPORT, reading note 3). The SRR gap should not be read as pure behavior.
3. **Doubt is a demoted, descriptive DV** (PLAN2 registry; lexically cued per EXPE; backend-sensitive). Report as contrasts only, per house rule — the table complies.
4. d=2+ bin is small (n=38) and heterogeneous (d=2..4); its CIs are wide.
5. Single model (Qwen2.5-7B-Instruct, EXPA legacy sweep); W6 replication is item E1.
6. Measured d is BFS rule-distance from the prefix state (`shortest_rule_distance`, depth-64), i.e., distance-to-refutation in the code-native fragment — not hop count in the model's own proof.

## Five-sentence answer to reviewer Q1 (for the response letter)

1. We re-audited every planted claim in the legacy one-hop and global cells with a single mechanical distance auditor and recomputed all outcome measures — permissive (closure-valid completion, verbalized doubt) and strict (hop-sound validity, strict repair) — through one identical row-level pipeline for both cells.
2. The audit confirmed the contamination concern behind Q1 — 14.9% of designed "one-hop" rows actually have their contradiction already stated in the prefix (d=0) — yet excluding those rows keeps or slightly widens every gap (closure-valid +27.8pp to +29.9pp, doubt +32.2pp to +33.1pp, hop-sound +8.7pp to +9.5pp, strict repair +24.9pp to +25.4pp; paired cluster-bootstrap 95% CIs all exclude zero), so the published contrast was not manufactured by mis-binned rows.
3. Under the strict stepwise standard, roughly one third of the permissive validity gap survives identical measurement (+9.5pp hop-sound, [+3.7, +15.2]), alongside a large strict-repair asymmetry (+25.4pp) that is partly structural, because the complement of a globally-false claim is underivable in our worlds and explicit correction is therefore closed to that cell.
4. However, our pre-registered deconfounded experiment shows this surviving gap does not reflect graded proof search: with fully matched positive categorical claims, rejection is a step function of visibility (stated-complement 0.247 at d=0 vs 0.031 at d=1, flat for d>=1), and absorption tracks derivational usability rather than counterevidence distance.
5. The residual doubt asymmetry is largely lexically cued (frequency-matched non-refuting overlap elicits the same "rejection" markers as genuinely refuting evidence), so the revised paper presents the legacy one-hop/global contrast as the confounded observation that motivated the pre-registered tests, with this decomposition quantifying exactly how much survives identical measurement.

## Provenance

- Script: `e6_decompose.py` (this dir); deterministic, seed 20260722, 10,000 bootstrap reps.
- Numbers: `q1_decomposition.json` (this dir).
- No files outside `iclr_exec/e6_legacy_gap_decomposition/` were written; results/ and existing improvement_plan subdirs untouched.
