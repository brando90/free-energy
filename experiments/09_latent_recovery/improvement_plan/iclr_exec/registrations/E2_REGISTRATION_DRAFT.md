# E2 — Mini pre-registration DRAFT: EXPG-FULL second regime (interpreter-validated program traces)

**Status: DRAFT — NOT REGISTERED. NOT timestamped. No Stage-2 generation may begin under this document.**
**Lead:** Elyas Obbad · **Drafted:** 2026-07-22 (agent draft for Elyas/Brando sign-off) · **Template:** PLAN2.md v1.1 house style.
**One-line claim under test:** in a second, better-instrumented regime (restricted-Python execution traces, external interpreter ground truth), silent absorption of a planted false value is governed by **check type** — plants whose falsity is exposed by *re-readable* stated evidence (adjacent self-conflict / note-style re-readable plants) are resisted more than plants presented as *computed* results (which would require recomputation to falsify).

Governing artifacts: `improvement_plan/reports/regime2.md` (EXPG design v2 — cells, generator constraints, metrics map, skeptic-attack table; adopted wholesale except where amended below), `results/EXPG_PROGTRACE_PILOT/REPORT.md` + STATUS.md 2026-07-02 EXPG entry (pilot outcomes that FORCE the DV/primary choice below), PLAN2 v1.1 (which parked EXPG and mandates: "its single contrast enters a NEW timestamped registration with the DV chosen by the §5.1-style evidence procedure on its pilot report" — this is that registration), ICLR_REVISION_PLAN.md §3 row E2.

---

## 1. Lineage (honest provenance — why the primary is what it is)

1. regime2.md originally proposed H7: the **k_r re-read-distance gradient** (judge-doubt + absorption, k_r=1 vs k_r=8) as EXPG's confirmatory pair.
2. **The pilot killed that choice** (instrument sound: 60/60 fixtures, solve 0.92, parse 1.00): k_r shows **NO gradient in any channel for Qwen2.5-7B — opfree k_r=1 = k_r=8 = 1.00 absorption**. Wrong values presented as computed results are absorbed wholesale regardless of how close the true value sits. Verbal doubt floors at 0 everywhere (both detectors), replicating the PrOntoQA doubt floor.
3. The **live contrast in the pilot is family-level**: adjacent self-conflict resisted at 26% (absorption 0.74) vs computed-value plants at 1.00 — a production-vs-verification asymmetry, consistent with the EXPH2 frontier result (haiku/sonnet-4-5 reject re-readable plants 0.04–0.22 absorption but absorb 100% of recomputable plants).
4. **Therefore:** the family-level contrast is THE single registered primary (§5). The k_r gradient is registered as **secondary/descriptive only**, with the reason printed: *it is known-flat for Qwen from the pilot; registering a known-dead test as confirmatory would be theater* (it retains descriptive interest because the frontier shows a k_r gradient, 0.043→0.217 — a cross-tier contrast, not a Qwen hypothesis).
5. H7-as-originally-conceived would have failed; its removal from the PLAN2 family (v1.1) is hereby carried forward, visibly.

## 2. Results already in hand (excluded from this family)

EXPG pilot (all cells, n=35-class); EXPH2 regime-2 frontier cells (descriptive, note-cell unparse ≤0.386 caveat); all EXPD/EXPE/EXPH results; arithmetic (n=99) and GSM8K (n=57) locked anchors. No slot below is scored on any of them.

## 3. Canonical variables

- **Check type** (the registered IV): *re-readable* (falsity certified by re-reading a stated line — zero recomputation) vs *recomputable-only* (falsity certified only by re-executing ≥1 op; no conflicting stated value anywhere).
- k_r (re-read distance) and k_c (recompute depth) remain the underlying dials (regime2 §2.1) but are secondary axes here.
- Cross-regime point values remain incomparable; only ordering/sign claims transfer (PLAN2 §3).

## 4. Design commitments (binding, enforced at generation — the STATUS-mandated fixes)

| # | Commitment | Enforcement |
|---|---|---|
| **G-FIX-1** | **Re-anchored plants.** Re-readable-family plants are re-anchored as **note-style plants** ("note: v = X" contradicting a value stated on a visible line), not as computed-result lines — the STATUS L65 re-anchoring mandate. Template frozen at Stage-1 gate; the EXPH2 note-cell unparse problem (≤0.386) must be fixed: parse ≥ 0.95 required on note cells before Stage 2 (MC-G1). | Stage-1 mechanical gate; template hash in manifest |
| **G-FIX-2** | **kc5 2.5× oversample.** `deep_kc5` cells generated at 2.5× target n (attrition/rejection-sampling losses at depth). | Generator config; funnel published |
| **G-FIX-3** | **Deep-cell grammar disclosure.** k_c ≥ 3 cells use the restricted arithmetic sub-grammar (all-`+`, single-digit operands; |v| ≤ 999 both worlds) — printed in the paper as a disclosed easiness concession, per regime2 S12. | Generator constants; disclosure sentence pre-written here |
| G-4 | MAJOR-5 (Axis-R chain roots are computed values, never listing literals), MOD-9 (fixed δ policy), MOD-10 (σ_cf(out) ≠ σ_true(out)), forward-use gap fixed at 2, distinct-values constraint — all carried unchanged from PLAN2/regime2. | Rejection sampling + MC-4-style asserts |
| G-5 | Parse ≥ 0.95 AND matched (≤5pp gap) across all four primary-contrast cells; dual-denominator everywhere; the primary must survive best/worst-case unparsed imputation bounds. | MC-G1; bounds computed in the analysis script |
| G-6 | No goal anchor in the prompt (parroting structurally absent; MC-8 reported). | Prompt builder |

## 5. THE confirmatory family (single slot, Holm m = 1, α = 0.05)

**G1 (THE single registered primary — family-level contrast):**
next-read absorption(**re-readable family**) < next-read absorption(**recomputable family**), one-sided, program-cluster bootstrap, pooled within family.

- **Re-readable family** = {`adjacent_conflict` (wrong restatement of the immediately-preceding line's value — the d≈0 self-conflict analogue), `note_opfree_kr1` (note-style plant contradicting a value stated 1 line back; zero compute; G-FIX-1 construction)}.
- **Recomputable family** = {`onehop_kc1` (wrong result of current line, operands adjacent, 1 op), `deep_kc5` (wrong deep/accumulator claim, ≥4–5 unstated ops, no conflicting stated value)}.
- **DV:** next-read absorption (first discriminating read cf-matches; regime2 §2.5). Chosen by the §5.1-style evidence procedure on the pilot: the effect lives in the absorption channel (26% resistance); doubt floors at 0 (both detectors) and is therefore unfit as a registered DV in this regime.
- **Passes iff:** one-sided p < 0.05 (m=1) **AND both** best/worst-case unparsed-imputation bounds preserve the sign (a bound-crossing result is INVALID-instrument: reported, no verdict).
- **Pre-stated fallback (fixed now, not post hoc):** if `note_opfree_kr1` fails MC-G1 (parse) at the Stage-1 gate, the re-readable family reduces to `adjacent_conflict` alone and G1 is adjudicated on the 1-vs-2-cell contrast; this fallback is disclosed in print if used.
- Robustness fits (reported beside G1, no stars): per-subcell contrasts (each re-readable cell vs each recomputable cell); final-output absorption as DV.

## 6. Registered secondary (adjudicated at unadjusted α, clearly labeled; NEVER promoted)

- **S-kr — k_r gradient (Axis-R, k_r ∈ {1,2,4,8}):** **secondary/descriptive by registration, with the reason printed: the pilot showed k_r=1 = k_r=8 = 1.00 absorption for Qwen2.5-7B (known flat; a confirmatory registration of this test would be registering a known-dead hypothesis).** Reported as estimation with CIs; its scientific role is the cross-tier contrast with the frontier gradient (0.043→0.217, EXPH2 descriptive).
- S-kc — Axis-C k_c trend (Cochran–Armitage, secondary).
- S-cert — certificate triples (operand restatement vs irrelevant vs baseline; regime2 R5 ordering).
- S-verify — verify-instruction satellite ({onehop, deep} × mid).
- S-repair — repair-event rate crossed with doubt (silent vs flagged repair; headline column in descriptive reporting).
- S-judge — judge explicit-rejection (upper-bound detector) per cell.

## 7. Exploratory (labeled; estimation only)

CRUXEval externally-authored arm (direction-only); redundancy moderator; k_r×k_c crossing cells; position effects; δ-sensitivity satellite; spot cells on E1 models (Llama-3.1-8B, OLMo-2-7B: {adjacent_conflict, note_opfree_kr1, onehop_kc1, deep_kc5} × mid × 150 — cross-model generality of G1's ordering, exploratory here, feeds E1's story); EXPG_FORCE cells (registered separately in E3 — not scored here).

## 8. DV registry (regime2 §2.5 mapping, restated)

Primary: next-read absorption. Secondary: final-output absorption, trace-valid, final-output-valid, repair events, line-skip rate, judge explicit-rejection. Descriptive only: verbalized doubt (frozen code-domain lexicon + judge; floor known; within-regime contrasts only; cross-regime doubt levels incomparable). 6-class run taxonomy reported per cell with unparsed visible.

## 9. Judge pass plan

Qwen2.5-32B via `doubt_judge_v2.py` per-regime prompt registry; EXPG prompt entry **frozen at the Stage-1 gate** (before Stage 2). One pass over **all** Stage-2 continuations (~13–14k rows; ~1–2 h vLLM). Outputs: per-cell judge explicit-rejection (secondary), judge-doubt (descriptive), judge/lexical agreement table. Family-affinity caveat printed; ≥50 EXPG rows are contributed to the E4 human-κ audit sample (coordinated with E4's protocol).

## 10. Cells, N, power, compute

Grid = regime2 §2.6 with amendments: core 6 families × 3 pos × 150 = 2,700; Axis-R 4 × 2 pos × 150 = 1,200 (note-style per G-FIX-1); Axis-C 4 × 2 pos × 150 with `deep_kc5` generated at 2.5× (G-FIX-2); crossing 3 × 2 × 150; certificates 3 × 2 × 150; verify 2 × 150; CRUXEval 2 × 150; redundancy 2 × 150; E1-model spot cells 4 × 2 models × 150. Gold 2,500 programs (cohort filter: greedy trace interpreter-exact + every line parses; funnel CONSORT-published). **Total ≈ 14–15k generations.**

Power (anchored on the pilot, no new power model): pilot effect adjacent 0.74 vs computed 1.00 (Δ = 0.26 at n=35). G1 pools ≥600 rows/family; two-proportion power ≈ 1.0 for Δ ≥ 0.10 at α=0.05. **The binding constraint is the imputation-bound gate**, hence G-5's parse requirements. Compute: ~3–6 GPU-h generation (vLLM) + 1–2 h judge; engineering 4–6 days (re-anchoring templates + tests dominate).

## 11. Decision rules and outcome branches (pre-stated)

- **G1 passes** → the re-read-vs-recompute boundary is confirmed at family level in a second regime with external ground truth; feeds the paper's §6 regime map; abstract may gain one clause (replication branch only).
- **G1 fails, both families ≈ ceiling** (recomputable ≥ 0.95 AND re-readable ≥ 0.95) → **floor branch, publishable and pre-registered as scope-bounding**: at scale the model absorbs even self-conflicting re-readable plants; the pilot's 26% resistance did not survive power; the regime documents a total in-stream verification failure; the paper's generality claim retreats to protocol-and-taxonomy transfer (regime2 §3 floor language).
- **G1 fails inverted** (re-readable > recomputable) → reported as registered; the production-vs-verification asymmetry claim is dropped for this regime and the cross-regime story is re-scoped in print.
- **Bounds-crossing** → INVALID-instrument, no verdict, parse post-mortem published.
- In every branch: all cells reported; nothing renamed post hoc; S-kr stays secondary regardless of what it shows.

## 12. Exclusion rules

1. Cohort: gold programs pass the double filter (interpreter-exact greedy trace ∧ all lines parse); funnel published; "upper bounds on a doubly filtered cohort" language kept verbatim.
2. Rows failing fail-closed audits (value bounds, distinct-values, σ_cf(out) ≠ σ_true(out), ≥2 discriminating reads, measured plant ≠ true value) are rejected and counted.
3. Cell n floors: as E1 §8.2 (100/50 thresholds; an underpowered primary subcell triggers the pre-stated fallback in §5 if applicable, else INVALID-underpowered).
4. Stage gates are mechanical only (parse/solve/template); **no outcome peeking**: no Stage-1 look at the G1 contrast may alter this registration except by dated pre-generation addendum.

## 13. Disclosure of deviations (pre-written)

- This registration supersedes regime2.md's H7 (k_r pair) as EXPG's confirmatory contribution; the change is pilot-forced and documented in §1 (the honest-provenance requirement).
- Deep cells use a disclosed easier sub-grammar (G-FIX-3); the deep-vs-re-readable contrast therefore mixes check-type with arithmetic-difficulty — mitigated by `onehop_kc1` (full grammar) sitting in the recomputable family; printed as a limitation.
- Note-style plants change the plant's discourse status (a "note" is not a claimed computation); this is the point of the re-anchoring, and the adjacent-conflict subcell (computed-style) guards the family contrast against being purely a discourse-status artifact; printed.
- EXPH2 frontier regime-2 rows are cross-format descriptive context only.

## 14. Open parameters needing Elyas/Brando decision (before timestamp)

1. Final note-style plant template wording (G-FIX-1) — freeze at Stage-1 gate.
2. Confirm the recomputable family = {onehop_kc1, deep_kc5} (alternative: kc1-only, cleaner grammar match, less depth coverage).
3. Confirm kc5 oversample factor 2.5× (vs 2×) against the Stage-1 rejection-rate measurement.
4. E1-model spot cells: keep (adds ~1,200 gens) or drop.
5. Whether S-cert (certificate triples) should be promoted to a second Holm slot (would make m=2) — recommendation: no (sprawl; single-primary discipline is the point).
6. Confirm G1's DV = next-read absorption (vs final-output absorption).

---

**DRAFT — not registered. Register by timestamped commit after Elyas + Brando sign-off.**
