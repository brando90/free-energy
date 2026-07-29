# E3 — Mini pre-registration DRAFT: forced-check mediator (powered EXPF + EXPG_FORCE)

**Status: DRAFT — NOT REGISTERED. NOT timestamped. No generation may begin under this document.**
**Lead:** Elyas Obbad · **Drafted:** 2026-07-22 (agent draft for Elyas/Brando sign-off) · **Template:** PLAN2.md v1.1 house style (hypothesis registry, DV registry, Holm family, decision rules, exclusion rules, deviation disclosures).
**One-line claim under test:** the model's failure to check derivationally-usable planted falsehoods is a failure of check-**initiation**, not of check-**capability**, at the cheapest-to-verify point in the family that drives the paper's headline absorption finding (`cat_false_usable`, d=1) — forcing the model to begin a check (a single forced "Wait," continuation) rescues validity/derivational-reuse there; whether the same forcing intervention also rescues on cells that require multi-step **recomputation** (EXPG deep/k_r=8) is the pre-registered second question, and either answer is a publishable, causally sharper version of the recompute boundary.

Governing artifacts: legacy pilot `exp/results_doubt/{force_summary.json, suppress_summary.json, probe.json}` (n=35/arm, unregistered, cited as prior only — `src/doubt_suppress.py STAGE=force` with `force_wait`/`force_neutral` arms is the existing, reusable mechanism per `improvement_plan/reports/feasibility.md:64,69`); PLAN2.md v1.0 §4 MOD-8, §5 H8, §5.2 (EXPG_FORCE), §5.1 EXPF branches (all superseded-then-revived, see §1); PLAN2 v1.1 (H8 REMOVED for TMLR scope — this document is the ICLR-scope relaunch PLAN2 explicitly anticipates); `results/EXPD_MATCHED_GRADIENT/EXPD_FULL_REPORT.md` (the `cat_false_usable_d1` anchor cell this design targets); `results/EXPG_PROGTRACE_PILOT/` + `improvement_plan/reports/regime2.md` (EXPG cell definitions); `registration_drafts/E2_REGISTRATION_DRAFT.md` (defines the re-readable/recomputable family split this document borrows for EXPG_FORCE cell selection); `improvement_plan/reports/skeptic.md` N4, `verdict.md` MOD-8 and the "single addition that most raises ICLR probability" note (EXPG_FORCE); ICLR_REVISION_PLAN.md §3 row E3.

---

## 1. Lineage (honest provenance)

1. **2026 (date not in bundle) — unregistered pilot** (`results_doubt/`, n=35/arm, legacy "global falsehood" pool, mid position only): forcing a single "Wait," continuation after a planted falsehood moved closure-valid 0.286→0.714 and poisoning 0.429→0.029; a length-matched neutral-token control did nothing (0.229/0.400); forcing "Wait," on a **true** interruption cost ~20pp validity relative to the unforced true-interruption baseline (0.629 forced vs ~0.856 unforced, EXPA-pooled) — the pilot's own "no cost" reading is **retracted** (STATUS.md 2026-07-01 entry; skeptic N4).
2. **Skeptic N4 / verdict MOD-8:** (i) the pilot's arm list lacks a **within-cohort** forced-Wait-on-true arm, so the ~20pp cost is a cross-cohort comparison, not a matched 2×2 — MOD-8 fix: add `forced_wait_on_true` (+~450 gens) for a within-cohort 2×2; (ii) with doubt forced to ≈1.0, the doubt DV is **degenerate** in the forced arms, and "0.286→0.714" is equally consistent with "Wait," triggering a **restart from premises** rather than a genuine local check — "initiation vs ability" requires content coding, not just the point-rate.
3. **PLAN2 v1.0** registered this as **H8** (§5) with the MOD-8 fix incorporated (5-arm design, §4/§8) and, separately, nominated **EXPG_FORCE** (§5.2) as an un-registered exploratory satellite extending the same forcing intervention into the program-trace regime's deep/k_r=8 cells — verdict.md calls this combination "the single addition that most raises ICLR probability," because it is the only asset that could unify the mediator claim (H8) and the second-regime claim (E2/H7) into one mechanism figure.
4. **PLAN2 v1.1 (2026-07-02, TMLR scope) REMOVED H8 from the confirmatory family and SKIPPED EXPF generation entirely** ("the legacy n=35 doubt-forcing pilot appears in the appendix as an unregistered pilot with its known caveats"). EXPG was parked at pilot in the same pass.
5. **This document is the ICLR-scope relaunch of H8, explicitly anticipated by PLAN2 v1.1** ("if EXPG is later resumed for an ICLR variant, its single contrast enters a NEW timestamped registration") and by the ICLR_REVISION_PLAN (E3 row: "one Holm slot in amendment"). It (a) restates H8 exactly as MOD-8 fixed it, (b) relocates the target cell from the legacy undifferentiated "global falsehood" pool to the EXPD-registered `cat_false_usable_d1` cell (§4.1 justifies why), and (c) formalizes EXPG_FORCE as a pre-registered exploratory satellite (still not in the Holm family — sprawl discipline per PLAN2 MOD-6/skeptic N3) with the outcome branches PLAN2 §5.2 already specified.
6. **No hypothesis here is scored on the pilot, on EXPD's existing `cat_false_usable_d1` rows, or on the EXPG pilot.** All three are cited as priors/design justification only; every row this document scores is freshly generated under this registration.

## 2. Results already in hand (excluded from this family)

- Legacy doubt-forcing pilot (n=35/arm, unregistered) — prior/motivation only, per §1.
- `results/EXPD_MATCHED_GRADIENT/`: `cat_false_usable_d1` closure-valid 0.0933 [0.06,0.13], derivational inj-dep 0.8633 [0.82,0.9033], stated-complement 0.0233, echo 0.0 (n=300, EXPD_FULL_REPORT.md headline table) — the **unforced baseline anchor** used for design/power below, not re-scored.
- `results/EXPG_PROGTRACE_PILOT/`: family-level result (adjacent self-conflict 26% resistance vs computed-value plants ~100% absorption; k_r flat for Qwen) — motivates EXPG_FORCE cell choice, not re-scored.
- `results/EXPH2_FRONTIER_FOLLOWUPS/`: the correlational recompute boundary (no model at any tier re-verifies a computed value) — EXPG_FORCE is the **causal** (interventional) test of the same boundary on Qwen2.5-7B; the two are complementary, neither re-scores the other.

Because all of the above are known, no hypothesis below is scored on them.

## 3. Canonical variables

- **Check-initiation (the manipulated IV):** whether the continuation is forced to begin with a fixed doubt-signaling token sequence immediately after the planted step, vs left unforced (model's own greedy continuation).
- **Forcing content (the isolating control):** `force-wait` = forced literal `"Wait,"` (byte-identical to the legacy pilot / `doubt_suppress.py STAGE=force force_wait` mechanism) vs `force-neutral` = a length/token-matched filler with no doubt semantics (`doubt_suppress.py force_neutral` arm, reused as-is — exact string carried over unchanged from the legacy mechanism, not re-authored).
- **Truth of the injected content:** false (the planted `cat_false_usable_d1` atom) vs true (a mirrored true statement in the identical syntactic slot, position, and length — construction detailed in §4.2).
- **Mechanism after forcing:** the model continues generation from the forced prefix under otherwise-unchanged decoding (greedy, vLLM, same max_new budget as EXPD).
- Cross-suite (EXPF vs EXPG_FORCE): the forcing intervention and its content-coding are declared **regime-local** — point rates are not compared across regimes; only the qualitative rescue/no-rescue verdict transfers (same convention as PLAN2 §3's cross-regime distance declaration).

## 4. Design commitments (binding, enforced at generation)

### 4.1 EXPF cell choice — why `cat_false_usable_d1`, not the legacy "global falsehood" pool

The legacy pilot's undifferentiated "global falsehood" pool predates the EXPD taxonomy (MAJOR-2's usable/inert split, MAJOR-4's d-binning) and cannot be reconstructed as a single cell today. `cat_false_usable_d1` is nominated instead because: (i) it is the cell with the **lowest baseline closure-valid in the entire EXPD grid among cells that are mechanically capable of being rescued** (0.093 — comparable in order of magnitude to the pilot's 0.286 baseline, with more headroom); (ii) it is the cell whose absorption (derivational inj-dep 0.863, flat across d per H4's reversal) is the paper's single most consequential finding — a rescue here is the sharpest possible test of "initiation, not incapacity"; (iii) d=1 is the **nearest** counterevidence distance in the usable family (one Horn-rule application from the visible prefix state) — the cheapest check available in that family, which is the closest EXPD analogue to "evidence is re-readable" the ICLR plan's phrasing invokes. **Disclosed distinction (not elided):** d=1 here still requires one explicit rule application, not a literal string re-read; the true zero-compute re-read analogue lives in EXPG (`adjacent_conflict`, `note_opfree_kr1`), which is why EXPG_FORCE's positive-control cell (§4.4) is load-bearing for interpreting this document's results, not decorative.

### 4.2 EXPF arm set (5 arms, MOD-8 fully incorporated)

| Arm | Content | Forcing | Role |
|---|---|---|---|
| A1 baseline/unforced-false | planted `cat_false_usable_d1` atom | none (model's own greedy continuation) | replicates the EXPD anchor under this cohort's own worlds (consistency check, not re-scored against EXPD) |
| A2 force-neutral | planted `cat_false_usable_d1` atom | neutral filler (`doubt_suppress.py force_neutral` string, unchanged) | isolates mere-interruption cost from "Wait,"-specific content |
| A3 force-wait | planted `cat_false_usable_d1` atom | `"Wait,"` (`force_wait` string, unchanged) | treatment |
| A4 unforced-true | mirrored **true** statement, identical slot/position/length | none | truth-content baseline, same cohort |
| A5 forced-wait-on-true | mirrored **true** statement, identical slot/position/length | `"Wait,"` | quantifies the true-content cost of forced checking, MOD-8 fix, within-cohort |

All 5 arms run on the **same set of worlds/families** (paired design, cluster unit = family_id, mirroring EXPD's mirrored-quadruple convention) so the 2×2 {forced,unforced} × {true,false} in A1–A3–A4–A5 is within-cohort, not cross-cohort (this is exactly what MOD-8/N4 demanded and the pilot lacked). **Engineering disclosure:** the mirrored-true-statement world variant for `cat_false_usable` (needed for A4/A5) does not currently exist in the EXPD generator (only the attribute family has a locked true quartet); it is a new, small construction (est. in §10) — flagged as an open engineering item, not assumed free.

n target: 150/arm/position × 3 positions = 450/arm, 2,250 perturbed generations total (matches PLAN2 §8's original EXPF sizing).

### 4.3 Content coding (resolves the N4 initiation-vs-restart ambiguity — mandatory, not optional)

Every A3 (force-wait, false) and A5 (forced-wait-on-true) continuation is coded into exactly one of: **local-check** (the continuation references or derives the complement of the specific planted/true atom using the visible counterevidence — a genuine check), **restart** (the continuation regenerates the proof from the premises without referencing the specific evidence — sidesteps rather than checks), **other** (neither). Coding = Qwen2.5-32B judge pass (per-regime prompt registry, `doubt_judge_v2.py`) with a **20-row hand-audited spot-check** before the full pass (publishable protocol note, mirrors validatorPlan Appendix B). This is a **required secondary analysis**, printed beside E-H1 in every reporting context: a "local-check" majority licenses the initiation reading; a "restart" majority means the rescue is real but the *mechanism* is not verification per se, and the paper's claim is written accordingly (disclosed either way, not just in the pass branch).

### 4.4 EXPG_FORCE cell choice and positive control

Cells (2 core + 1 positive control, all forced-wait vs unforced, n=150/cell, mid position):
- **Recomputable family (the test):** `deep_kc5` (≥4–5 unstated ops, no conflicting stated value — the deepest recompute cell) and `opfree_kr8` (Axis-R k_r=8, the deepest re-read-staleness cell registered as secondary in E2 — included here because it is literally the "k_r=8" cell the ICLR plan and PLAN2 §5.2 name).
- **Re-readable positive control (mandatory, not exploratory-optional):** `adjacent_conflict` (or `note_opfree_kr1` if the E2 re-anchoring lands first — whichever is live in the E2 cohort at EXPG_FORCE launch time) forced-wait vs unforced. **This control must show a forcing effect (rescue in the re-readable cell) before any null on `deep_kc5`/`opfree_kr8` is interpreted as a recompute-boundary finding** — without it, a null is ambiguous between "recompute is a real capability ceiling" and "the forcing manipulation does not transfer to the program-trace prompt format." This requirement is stated here, pre-registered, precisely so it cannot be skipped or added post hoc if the deep cells come back null.
- Forcing string in the code regime: `"# wait,"` inserted as a forced next-line prefix, or `"Wait,"` inline if the trace format tolerates prose interjections — **exact format is an open parameter (§14)**, to be frozen at EXPG_FORCE's own Stage-1 gate (mirrors E2's G-FIX-1 note-template freeze) before any scored generation.
- EXPG_FORCE is an **exploratory satellite** (PLAN2 §5.2 language carried forward verbatim) — it never enters the Holm family (§5), regardless of outcome.

## 5. THE confirmatory family (single Holm slot, m = 1, α = 0.05)

**E-H1 (the ONE registered primary — conjunction, intersection-union, no alpha splitting):**

Within the false-injection arms of the `cat_false_usable_d1` cohort (§4.2):
(i) closure-valid(A3 force-wait) > closure-valid(A2 force-neutral), one-sided, GLMM logistic clustered on family_id;
(ii) derivational injection-dependence(A3 force-wait) < derivational injection-dependence(A2 force-neutral), one-sided, same clustering.

**Passes iff** both components reject at α=0.05 (m=1) **AND** both point differences clear pre-registered minimum magnitudes: (i) closure-valid diff ≥ **+0.20** (roughly half the legacy pilot's 0.428 delta — conservative under a lower, more absorbing baseline); (ii) derivational-inj-dep diff ≤ **−0.30** (roughly a third of the 0.863 baseline). **The comparison arm is force-neutral, not baseline/unforced** — this isolates the "Wait,"-specific mechanism from the generic cost/benefit of any forced interruption, correcting the loosest version of the legacy claim (which compared force-wait only to unforced baseline).

Magnitude thresholds are **open parameters for sign-off** (§14); once timestamped they are immutable, per house convention.

## 6. Registered secondary (adjudicated at unadjusted α, reported with CI, never promoted into §5)

- **Truth×forcing interaction (the ~20pp cost, MOD-8's estimation target):** [closure-valid(A5) − closure-valid(A4)] vs [closure-valid(A3) − closure-valid(A2)] — reported as a difference-in-differences with 95% cluster-bootstrap CI, **no star**, printed beside E-H1 in every table (PLAN2 MOD-8 convention carried forward verbatim).
- **Stated-complement** on the same E-H1 contrast (secondary rejection DV, v1.1 convention).
- **Force-neutral vs baseline/unforced (A2 vs A1):** the mere-interruption cost/benefit, estimation only.
- **Content-coding split (§4.3):** local-check vs restart vs other, reported as proportions with CI for A3 and A5 separately.
- **EXPG_FORCE positive-control effect size** (re-readable cell, forced vs unforced): reported with CI; gates interpretation of the deep-cell result per §4.4 but is not itself Holm-adjusted.
- **EXPG_FORCE deep-cell effect(s)** (`deep_kc5`, `opfree_kr8`, forced vs unforced): reported with CI on next-read absorption and (if parseable) trace-valid; never promoted to confirmatory (PLAN2 §5.2 language: "exploratory satellite with pre-stated readings both ways").

## 7. Exploratory (labeled; estimation only, no stars)

Doubt DV in forced arms is **structurally degenerate** (forced token ≈ saturates the lexical/judge doubt detector) and is reported, if at all, only as a manipulation-check sanity value, never as evidence of anything; A1 (unforced baseline) vs the existing EXPD `cat_false_usable_d1` rows as an informal cross-cohort consistency check (not a registered comparison — different generation batch, same generator/prompt hash expected but not guaranteed identical); one additional E1-replication-matrix model run through the identical EXPF design **if cheap** (per ICLR_REVISION_PLAN E3 row, "Qwen2.5-7B (+1 E1 model if cheap)") — fully exploratory, no claim rides on it; onehop_kc1 forced-vs-unforced as a third EXPG_FORCE cell if budget allows (recomputable-but-shallow, bridges deep_kc5 and the positive control).

## 8. DV registry

**Primary (confirmatory, E-H1 only):** closure-valid; derivational injection-dependence (M4 split, MAJOR-2 construction unchanged).
**Secondary:** stated-complement; content-coding categorical (local-check/restart/other); truth×forcing interaction (estimation); EXPG_FORCE next-read absorption and trace-valid (estimation).
**Descriptive only, never registered:** verbalized doubt (lexical + judge) in ALL arms — degenerate-by-construction in forced arms, backend-sensitive in unforced arms (v1.1 A.3 carried forward).
**Reported for every cell:** 6-class multinomial with unparsed visible; dual-denominator wherever parse rates differ across compared arms.

## 9. Judge pass plan

Qwen2.5-32B via `doubt_judge_v2.py` (per-regime prompt registry) performs two jobs, kept analytically separate: (a) the pre-registered secondary explicit-rejection detector on all EXPF/EXPG_FORCE rows (upper-bound DV, family-affinity caveat carried forward — judge is not a subject model here, no conflict); (b) the §4.3 content-coding pass (local-check/restart/other) on A3/A5, with the 20-row hand-audited spot-check reported as agreement %. Both passes run before any E-H1 verdict is written up (sequencing gate, PLAN2 §9.9 convention).

## 10. Cells, N, power, compute

- EXPF: 5 arms × 3 positions × 150 = 2,250 perturbed generations + gold budget for the mirrored-true world variant (est. 2–4× the EXPD `cat_false_usable` gold rate pending the §4.2 engineering item; **TBD, filed at the EXPF Stage-1 gate**).
- EXPG_FORCE: 3 cells (2 core + 1 positive control) × 150 × {forced, unforced} = 900, mid position only.
- **Power (from the legacy pilot anchor, no new power model invented, per the task spec):** pilot Δ=0.428 at n=35/arm (Cohen's h ≈ 0.90, a very large effect). At n=450/arm (150×3 positions pooled), a two-proportion test has power > 0.99 for h ≥ 0.90 at α=0.05 one-sided; even at half the pilot's magnitude (h ≈ 0.45, roughly the §5 minimum-magnitude threshold of +0.20 against a ~0.09–0.29 baseline), power is ≈0.95–0.99. The registered minimum magnitudes in §5 are therefore comfortably powered if the pilot's mechanism transfers at all to the new cell; the binding risk is **effect transfer** (legacy cell ≠ `cat_false_usable_d1`), not sample size — disclosed.
- EXPG_FORCE power: pilot family-level effect (adjacent 0.74 vs computed 1.00, Δ=0.26 at n=35) is smaller than the EXPF anchor; at n=150/cell forced-vs-unforced, two-proportion power ≈0.85–0.95 for Δ≥0.15–0.20 — adequate for a satellite whose outcome is reported either way (§11), not gated on a magnitude threshold.
- Compute: EXPF ≈0.1–0.3 GPU-h generation (vLLM, EXPD-scale precedent) + gold generation for the new true-variant worlds (TBD, §4.2); EXPG_FORCE ≈2 GPU-h (verdict.md estimate) on top of E2's runner. Total against the ICLR_REVISION_PLAN's ~0.5–1 GPU-day + 2–3 days plumbing (legacy `STAGE=force` code exists) — consistent.

## 11. Decision rules and outcome branches (pre-stated, PLAN2 §5.1/§5.2 carried forward)

- **E-H1 passes, small true-cost** (truth×forcing interaction CI excludes anything > ~15pp): mediator/initiation claim promoted to main text with the cost printed in the same sentence (PLAN2 §5.1 "H8 passes with small true-cost" branch, cost threshold itself an estimation quantity, not a gate).
- **E-H1 passes, large true-cost:** same promotion, cost stated prominently — never the "costs nothing" framing the pilot's own data already falsified (N4).
- **E-H1 fails (either component, or magnitude threshold not cleared):** mediator claim dropped from main text; EXPF reported in appendix with the legacy pilot; "doubt-forcing" language is deleted from any main-text initiation claim (PLAN2 §5.1 "H8 fails" branch, verbatim).
- **Content coding (§4.3) reframes rather than gates:** even if E-H1 passes on point-rates, a "restart"-majority coding result requires the write-up to say "forcing generates a rescue via restart, not via a localized check" — this is a disclosure obligation, not a pass/fail condition on E-H1 itself.
- **EXPG_FORCE, positive control fails (no forcing effect even on the re-readable cell):** the whole satellite is INSTRUMENT-INVALID for this format; deep-cell results are not interpreted as evidence about the recompute boundary; reported as a null methods finding only.
- **EXPG_FORCE, positive control passes AND deep cells rescue too:** "unified initiation-substitutes-for-proximity" story — forcing checks works across both re-readable and recomputable evidence; strengthens E-H1 into a general initiation claim (PLAN2 §5.2 branch 1).
- **EXPG_FORCE, positive control passes AND deep cells do NOT rescue:** causal confirmation of the recompute boundary — initiation is necessary but not sufficient; recompute requires a capability the forced check does not unlock; directly strengthens the correlational EXPH2 recompute-boundary claim into an interventional one; this is the ICLR plan's named "publishable either way" branch (PLAN2 §3.4 conjecture confirmed causally).
- In every branch: all arms/cells reported; nothing renamed post hoc; the doubt DV is never used as evidence in any branch (§8).

## 12. Manipulation checks

| MC | Check | Criterion / action |
|---|---|---|
| MC-F1 | Per-arm parse rate (EXPF) | matched within 5pp across A1–A5; else dual-denominator + imputation bounds beside E-H1 |
| MC-F2 | Forced-prefix fidelity | every forced continuation verified to begin with the exact registered string (byte check); a manifest of any forcing failures (e.g., tokenizer edge cases) published, never silently dropped |
| MC-F3 | True-arm mirroring | A4/A5 world atoms verified true under the world's closure (fail-closed audit, same pattern as EXPD's truth-status assert); position/length-matched to A1–A3 (token-length gap ≤ 5%) |
| MC-F4 | Doubt-detector saturation sanity | lexical/judge doubt in A3/A5 ≥0.90 (confirms the forcing mechanism actually fires); a lower rate is itself reported as a forcing-fidelity problem, not folded into E-H1 |
| MC-G1 | EXPG_FORCE parse rate | ≥0.90 on all three cells before scoring; positive-control cell specifically must clear parse ≥0.90 or the whole satellite is flagged instrument-limited (§11) |
| MC-G2 | Forcing-string format sanity (code regime) | 20-row hand check that the forced prefix parses as a valid trace-adjacent utterance, not a syntax break that inflates "unparsed" mechanically |

## 13. Exclusion rules

1. **Row-level:** identical double filter to EXPD (gold `solved` ∧ `valid_rederivation`); fail-closed audit rejections counted, never patched.
2. **Cell/arm-level:** target n=150/arm/position. 100 ≤ n < 150: scored, flagged. 50 ≤ n < 100: scored, CI-widened, flagged prominently. n < 50: that arm's contribution to E-H1 is **INVALID-underpowered**, reported not scored, counts as non-pass for E-H1 (conservative direction, mirrors E1 §8.2).
3. **EXPG_FORCE cell-level:** same thresholds; the positive-control cell failing its n floor invalidates interpretation of the deep cells per §11, independent of their own n.
4. No arm/cell/hypothesis added or removed except by timestamped addendum filed before the affected suite's first scored generation.

## 14. Disclosure of deviations (pre-written)

- E-H1's target cell (`cat_false_usable_d1`) is **not** the legacy pilot's cell (an undifferentiated "global falsehood" pool that predates the usable/inert and d-binning taxonomy); §4.1 states the substitution reasoning in full — this is a deliberate, disclosed relocation, not a silent one.
- The mirrored-true world variant needed for A4/A5 is new engineering, not a reused EXPD asset; its gold-attrition rate is unknown until piloted (§10 TBD, filed at Stage-1).
- The legacy `force_neutral`/`force_wait` strings are reused byte-identical from `doubt_suppress.py`; if that file cannot be located/loaded on the current cluster snapshot, the exact strings must be re-derived from `results_doubt/force_summary.json`'s recorded prompts before Stage-1 (disclosed contingency, not silently improvised).
- EXPG_FORCE's forcing string format is genuinely undecided (§4.4) — an open parameter, not a design gap papered over.
- Doubt is descriptive/degenerate throughout this document by construction (forcing directly manipulates the doubt-adjacent surface form) — never used as a DV in any registered or secondary slot.

## 15. Open parameters needing Elyas/Brando decision (before timestamp)

1. **Magnitude thresholds for E-H1** (closure-valid ≥ +0.20, derivational-inj-dep ≤ −0.30) — sign off or adjust.
2. **Comparison arm for E-H1**: force-neutral (as specified) vs baseline/unforced — confirm force-neutral is the correct isolating control.
3. **Mirrored-true world construction for `cat_false_usable`** (A4/A5): approve the engineering item and its gold-attrition budget once piloted.
4. **EXPG_FORCE forcing-string format** in the code regime (`"# wait,"` vs inline `"Wait,"` vs other) — decide or delegate to the Stage-1 gate.
5. **EXPG_FORCE cell set**: confirm `{deep_kc5, opfree_kr8}` as the recomputable pair, and the positive-control cell (`adjacent_conflict` vs `note_opfree_kr1`, contingent on E2's re-anchoring timeline).
6. **Content-coding protocol** (§4.3): confirm judge-based coding + 20-row hand spot-check is sufficient, or require a second human coder (would tie into E4's human-audit package/rater availability).
7. **E1-model exploratory add-on**: which model (if any) gets the "if cheap" EXPF replication, and whether it is worth the extra plumbing before the Sept 1 gate.
8. **Sequencing relative to E1/E2**: confirm this document may be timestamped independently, or must wait for E2's re-anchored EXPG cells to stabilize before EXPG_FORCE's cell definitions are frozen (§4.4 already assumes possible dependency on E2).

---

**DRAFT — not registered. Register by timestamped commit after Elyas + Brando sign-off.**
