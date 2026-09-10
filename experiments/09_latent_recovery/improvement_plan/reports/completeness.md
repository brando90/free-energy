# Diff: Four Plans vs. PROFESSOR_FEEDBACK.md (verbatim file read at `/private/tmp/claude-501/-Users-elyas-Desktop-Alara/ad633660-4a3e-4cbc-99dd-093a0feae95e/scratchpad/latent_recovery/PROFESSOR_FEEDBACK.md`)

## 0. Provenance discrepancy (must be resolved first)

The feedback file **exists locally** and was read in full. Plans C and D cite it correctly (C: "Read: PROFESSOR_FEEDBACK.md"; D quotes its sub-bullets verbatim and they match the file exactly). **Plans A and B open by asserting the file does not exist** ("confirmed by fresh mdfind", "all task paths were literal undefined") and claim to be keyed to a second-hand quote of the confound list. Their reconstructed demand lists happen to match the real file (A's seven axes = the file's line 19/21 axes + style), so no design content is invalidated, but A's added claim that "the experiment source tree is not local" is also false (C/D used a local rsync mirror at `.../scratchpad/latent_recovery/` incl. `exp/src/*.py`, `RESULTS_LOCK.md`). A's instruction to "verify file-level integration points on skampere2" is partly unnecessary — most can be verified against the local mirror (raw JSONLs are cluster-only per C).

## 1. Demand-by-demand coverage matrix

Feedback line numbers refer to the file. Status: **C** covered / **P** partial / **M** missing (across all four plans combined).

### Framing / recommendation (lines 3–10)

| Demand | Status | Where |
|---|---|---|
| TMLR-first, ICLR only if causal isolation strengthened + story simplified | **C** | D §5 (go/no-go gate 2026-09-01, dual-submission handling); A provides the causal-strengthening content, D the simplification (ICLR variant ~9 pp) |
| Reframe core contribution as "audited perturb-and-validate protocol + local counterevidence modulates absorb/reject/route" | **C** | D §1 (single defended sentence is a near-paraphrase of the professor's line 8); D §6 title alternate (b) foregrounds the protocol |
| Credibility via exposition, appendices, validator detail, claim calibration | **C** | D Part 2 (skeleton + ~14 pp appendices), C Parts 1–2 |

### Weakness 1 — causal variable not isolated (lines 14–24)

| Sub-demand (line 19/21 axes) | Status | Where |
|---|---|---|
| Stylistic change (benign paraphrase confound) | **C** | A §2.1 (identical sentence frame pinned; ±1 token logged as covariate; benign controls regenerated on new worlds) |
| True-but-off-path (true interruption confound) | **C** | A §2.1 `neg_true_d*`/`aff_true_d*` mirror cells (truth × polarity 2×2 within-world); P6 |
| Rule-like vs entity-fact-like (distractor confound) | **C** | A §2.2 `rule_false_d0`, `rule_false_far` |
| Explicit lexical contradiction | **C** | A §2.2 `lex_decoy`; overlap covariates logged all cells + retro-fitted (A §5.1) |
| Negation (1-hop falsehoods all negated) | **C** | A §2.1 full polarity × distance factorial via attribute route; balance property (negation count constant across polarity cells); P3 TOST |
| Affirmative categorical / off-path / usability (global falsehood confound) | **C** | A §2.1 `aff_false_unent` bridge cell; §2.2 `cat_*_inert/usable` (usability × distance), `*_offpath_d3` (path relevance) |
| Category frequency | **C** | A §2.2 `freq_high_d3` + frequency covariates |
| "Is it inferential distance or …?" — formalize the variable | **C** | A §1 refutation-distance d(P,S) definition + C §1.2/Alg 5 (adopts EXPC `shortest_rule_distance`) — **two definitions, need one canonical owner** (see §3) |
| Distance-k + cert-flip "not presented with enough force or detail" | **C** | A §5.2 Figs A/B/C + design-matrix table (axes × conditions ✓/✗/pinned); A EXPE (byte-identical sentence, evidence moves); D §4.2/4.3 (named subsections; cert-flip recast vs irrelevant-cert arm, macros named) |

### Weakness 2 — sprawl (lines 26–28)

| Sub-demand | Status | Where |
|---|---|---|
| Too many experiments; TMLR OK if long/well-organized; ICLR needs one-big-thing | **P** | D Part 2 fate table (every existing experiment: MAIN/APPENDIX/CUT/RESTORE) + ICLR variant cuts. **Partial because the four plans jointly re-create the sprawl**: D's fate table predates and does not budget A's EXPD/EXPE (~15 new cells), B's entire program-trace regime, or C's strict-metrics tables. No plan owns the combined page budget (D targets 14.5 pp main before A/B/C additions land). This is the largest cross-plan integration gap. |

### Weakness 3 — validator under-specified (lines 30–35)

All nine sub-demands are individually enumerated and answered by Plan C §1.2 (with file:line grounding in `validator.py`), pseudocode in C §1.3, disclosure checklist A.8; Plan D §3 independently lists all nine as writing requirements.

| Sub-demand | Status | Where |
|---|---|---|
| Proof grammar allows what | **C** | C §1.2 "Rule grammar" (six surface templates, validator.py:33–54) + Prop A.1 |
| Facts/rules/categories/negations representation | **C** | C §1.2 "World" (three predicate shapes, validator.py:56–68) |
| How closure is computed | **C** | C §1.2 "Closure" (lfp, validator.py:87–105), Alg 2 |
| Unentailed = false under CWA? | **C** | C §1.2 "Truth semantics" — the deliberate asymmetry (CWA for positives, open-world for negations) stated verbatim from `audit_truth_status` |
| Injection-dependence detection | **C** | C §1.2 (derivable-with-a\*-only; echo vs derivational split, M4); A §1 Move 3 adds the belief-probe DV fixing the †-measurability confound |
| Unparsed → denominators | **C** | C D2 (retained in denominators; stale docstring fix; dual-denominator appendix table); A treats unparsed as first-class outcome class + imputation bounds (the 0.317 cert-flip hole) |
| Parroted vs closure-valid | **C** | C §1.2 precedence list (unparsed ≻ derailed ≻ inj-dep ≻ parroted ≻ closure-valid), definitional statement |
| What happens after an invalid step | **C** | C §1.2 "Step validity and state evolution" + D4 two-state taint-tracking v2 |
| Discourse markers stripped before/after doubt detection | **C** | C D1 (doubt runs on raw text; lexicons disjoint; order provably irrelevant — printed verbatim in A.5) |
| "Black box" overall | **C** | C §1.1 promoted §3 Validator + Appendix A; `validator_spec_export.py` (appendix autogenerated from code, cannot drift); C Part 4 human audit + κ gates; D §3 validator self-audit (~100 labels — subsumed by C's ~380–430 protocol) |

### Weakness 4 — "recovery" is not always recovery (lines 37–41)

| Sub-demand | Status | Where |
|---|---|---|
| Closure-valid permits skipped derivations and goal jumps | **C** | C M2 (goal-jump rate promoted to every cell; hop-sound validity = no skipped derivations), D restores 4.7–9.2% numbers |
| "Detect and repair, or avoid/skip/exploit goal?" | **C** | C M1 strict-repair (flag∨re-derivation ∧ valid ∧ ¬goal-jump ∧ ¬stated-reliance), M3 stated-reliance (the `d45c87be` case), M4 unconditional injection-use |
| Replace "recovery" with the five precise terms | **C** | C §2.1 defs box contains all five professor terms; §2.2 usage rules ("recovery" banned as metric term); §2.3 line-by-line main.tex map + lint script. D Part 3 has an overlapping 15-row calibration map — **merge required** (see §3) |
| "Repair" only when correction or logical routing-around | **C** | C splits the professor's disjunction into *strict repair* vs *routing around* (justified by the Llama/OLMo dissociation); D calibration #1 fixes the abstract |

### Weakness 5 — generality (lines 43–50)

| Sub-demand | Status | Where |
|---|---|---|
| GSM8K/arith "not deeply integrated or equivalently validated" | **C** | B §5: arithmetic absorbed as degenerate cell of the new regime; Table 7 → regime map; GSM8K demoted to non-audited corroboration with caption caveat |
| One more serious regime (options: Lean/miniF2F; theorem-proving traces; richer synthetic logic; program-execution traces; symbolic math) | **C** | B §1 evaluates **all five listed options** in a criteria table; picks program-execution traces (one of the professor's own options) + CRUXEval external-provenance arm + optional SymPy secondary; full design §2, replication criteria §3 |
| TMLR risk "too narrow/synthetic unless methodological contribution explicit" | **C** | D §6 (title alt (b), contribution reordering with protocol first); B's external-interpreter ground truth + externally-authored arm directly answers "artifact of a grammar the authors wrote" |
| ICLR: "rewrite much more aggressively, cleaner causal story" | **C** | D §5 ICLR variant; A ICLR arms (72B, GSM8K evidence-mover, verification×distance) |

### Proposed TMLR structure (lines 54–60)

| Bullet (with sub-items) | Status | Where |
|---|---|---|
| Intro: compounding assumption; local refutability; protocol | **C** | D §1 (+ construct-trap vignette restored) |
| Protocol: PrOntoQA setup; **proof filtering**; injection; families; truth audit | **C** | D §2 (funnel table with exact exclusion counts; inversion cross-tab restored); C §1.2 "Cohort and filtering funnel" (CONSORT-style, A.7) |
| Validator: closure def; the six classes; doubt detector | **C** | C Part 1 (defs box, §3, App A); D §3 restores 32B doubt-judge validation (dropped from current draft) — professor's "doubt detector" sub-bullet |
| Main results: 7B sweep; distance-k; cert-flip | **C** | D §4.1–4.3 exactly mirrors the three bullets; A's Figs A/B replace/augment; D §4.4 conditional doubt-forcing addition (an extra beyond the professor's three — defensible but adds length) |
| Alternative explanations: target copying; polarity; task structure; scale/lineage | **C** | D Part 4: all four professor items as §5.1–5.4 with Hypothesis→Test→Verdict, plus two extras (elicitation gap, interruption) and conditional 5.7 |
| Limitations: synthetic domain; closure vs strict; perturbation confounds; Qwen-heavy; deterministic decoding | **C** | D §6 (five bolded items in his order + additions). Deterministic decoding gets an actual *experiment* only as A's optional 3-seed T=0.7 arm — recommend promoting from optional |
| Appendix: prompt templates | **C** | D App A; C Part 5 (templates located in code: common.py:17–31, expa:52–66, anchor_qa.py:20–29, arith_pilot.py:21–29) |
| Appendix: validator pseudocode | **C** | C §1.3 Algs 1–5 (≤15 lines each, mirroring code) + `validator_spec_export.py`; D App B |
| Appendix: all tables | **C** | D App C (full per-cell family×position×model incl. Mistral disposition) + C App C strict-vs-permissive |
| Appendix: all confidence intervals | **C** | D App C; C (Wilson + cluster bootstrap on every strict summary); A (bootstrap CIs beside every GLMM estimate) |
| Appendix: filtering pipeline | **C** | D App D (EXPA eligibility funnel 22,132 rows, EXPB/EXPC availability, late-write disclosure); C A.7 |
| Appendix: examples | **C** | C App B (1–2 verbatim transcripts per class *with validator trace*); D App A worked instances |
| Appendix: model details | **P** | D App J (models, decoding, compute) and C D18 (seed/greedy/budgets) — but **no plan pins exact HF model revisions, vLLM version, dtype, or discloses that vLLM "greedy" is not bit-deterministic under dynamic batching** — a real reproducibility hole given the one-seed design |
| Appendix: regression details | **C**/P | D App J (cluster bootstrap + fixed-effect logistic spec); A §4 adds GLMMs — the appendix must reconcile *two* stats frameworks (legacy bootstrap-logistic for locked tables, new GLMM for EXPD/EXPE); no plan writes the reconciliation text |

**Bottom line on professor coverage:** every demand, sub-demand, and structure bullet is covered by at least one plan; the only P-grades are (i) sprawl-after-integration (W2), (ii) model-detail version pinning, (iii) dual stats-framework reconciliation, (iv) deterministic-decoding mitigation being optional.

## 2. Cross-plan conflicts the orchestrator must adjudicate

1. **Namespace collision:** Plan A's new suites are `EXPD` (`expd_matched_gradient.py`) / `EXPE`; Plan D independently names its powered doubt-forcing rerun `EXPD_DOUBT_FORCE`. Rename one (suggest doubt-forcing → EXPF) before anything is pre-registered.
2. **Lean disposition:** B rejects Lean this cycle with a fatal-flaw argument (cohort collapse under frozen-greedy) and mandates editing the Limitations promise; D's timeline schedules "Lean/VeriBench arm (Aug 4–29)" and makes it ICLR gate criterion #4 (with "or program-execution traces" as escape). Adopt B's analysis; rewrite D's criterion #4 to name B's program-trace regime as the satisfying instrument and keep Lean as B's concrete future-work sketch.
3. **Refutation-distance definition:** A defines d over "facts in S under the world's rule closure" emitted by BFS; C's Alg 5 adopts EXPC's `shortest_rule_distance` (direct rule applications from prefix state). Nearly identical, but subtle differences (prefix state vs stated premises; direct vs closure steps) will produce off-by-one disagreements at exactly the d=1 vs d=2 boundary the confirmatory hypotheses test. One canonical implementation (C's retroactive M5 audit is also A's §5.1 retro-audit — same job, do once).
4. **Terminology ownership:** C §2.3 and D Part 3 are overlapping line-by-line edits of the same `main.tex`, keyed to different line-number snapshots, with slightly different replacements for the same sentences (e.g., abstract "doubt and repair": C makes it conditional on computed SRR; D replaces with "explicit flagging and closure-valid completion"). Merge into one map; C's lint script should be the enforcement mechanism; C's defs box must absorb A's new DV terms (Δbelief probe, echo rate) and B's regime metrics (trace-valid, repair event) or the controlled vocabulary fractures across sections.
5. **Polarity-cell mechanism:** D's ICLR criterion #1 prescribes a "disjoint-category grammar" for the local-affirmative-false cell; A's §2.1 attribute route is the worked-out, provably feasible version. A supersedes; update D's gate wording.
6. **Cert-flip re-analysis:** A §2.3 and D §4.3 both re-present EXPB vs the irrelevant-cert arm; A additionally demands joint arm modeling and unparsed imputation bounds. Do A's version once; D's prose adopts it.
7. **Doubt-forcing asset:** D discovered the unpublished `results_doubt` forcing result (0.286→0.714) and explicitly says "coordinate with the causal-isolation designers"; A does not incorporate it. It belongs in A's confirmatory framework (it is a mediator manipulation) — add as a pre-registered EXPF with A-style stats rather than D's looser spec.
8. **Sprawl budget (repeat of §1 W2):** integrating A+B+C+D naively yields ~18–20 pp main text. Someone must own a fate-table v2 covering the *new* experiments (e.g., A's satellites → appendix by A's own ranking; B's regime gets one main-text subsection + regime map, rest appendix).
9. **Evidence-base mismatch:** A and B were written against the extracted PDF only and repeatedly flag "verify file names on skampere2"; C has the actual `exp/src` mirror with file:line cites. C's ground truth should be used to pre-verify A's §3 integration points (e.g., A assumes `matched_pert.py`, `expb_local_cert_flip.py` names — C's file list confirms `expc_polarity_control.py`, `perturb.py`, `audit.py` etc., but not `matched_pert.py` or `expb_local_cert_flip.py`; check before PLAN2.md cites them).

## 3. Prioritized gap list — important for a real TMLR submission, mentioned by neither professor nor any plan

**P0 (submission-blocking or reviewer-facing):**
1. **Integration page-budget / fate-table v2** (detailed in §2.8) — without it the four plans jointly regress Weakness #2, the professor's second-biggest complaint.
2. **Double-blind anonymization plan.** TMLR review is double-blind. The plans promise artifact release ("we release the validator, truth-status audit, and locked result artifacts" — D's abstract draft) but nothing about an anonymized repo (e.g., anonymous.4open.science), and the artifacts as described are de-anonymizing: `RESULTS_LOCK.md` hashes cluster paths like `/lfs/skampere2/0/eobbad/free-energy/...`, commit ids, and the repo name. Scrubbing procedure needed before any supplementary upload.
3. **Related-work refresh for the 2023–2026 CoT-perturbation/self-correction literature.** D only re-adds two cut cites; no plan sweeps the field. Closest prior art that reviewers will demand differentiation from: Lanham et al. 2023 (adding mistakes to CoT to measure faithfulness — essentially an un-audited version of this protocol; the paper's "audited" contribution should be positioned *against* it explicitly), Huang et al. ICLR'24 ("LLMs cannot self-correct reasoning yet"), Tyen et al. 2024 (mistake-finding vs mistake-correcting, BIG-Bench Mistake), premise-perturbation/distractor robustness work (GSM-variant studies), process reward models / step-level verifiers (external detection of exactly these planted errors), 2025 CoT-faithfulness results, and R1-era analyses of backtracking/"aha" tokens (directly relevant to the doubt-forcing "Wait" result and to the R1-Distill lineage claim). A missing-differentiation finding here is a "claims not supported" risk, not just polish.

**P1 (required or strongly expected TMLR sections/statements):**
4. **Broader Impact Statement.** A standard, expected TMLR section; absent from the current draft (no appendices at all) and from all four plans' skeletons. Trivial to write for synthetic-domain work, but its absence is a desk-level flag.
5. **Environment pinning + greedy-decoding determinism caveat.** No plan records HF model revisions, vLLM version, dtype, or the fact that vLLM greedy decoding is not bit-reproducible under dynamic batching/kernel nondeterminism — material because the entire design is one-seed greedy and the plans build confirmatory statistics on single continuations. One paragraph in App J + pinned `requirements`/manifest per result dir.
6. **Dataset/model licensing and citation hygiene.** PrOntoQA-OOD (Saparov & He) code license and generator provenance (the paper *modifies* the generator — state which); model licenses (Qwen: Apache-2.0/Tongyi license depending on size — 72B has a different license than 7B, relevant to A's ICLR arm; Llama 3.1 community license constrains derivative artifact release of continuations). Nobody mentions any of this.

**P2 (reviewer-question insurance):**
7. **Training-set contamination of PrOntoQA.** PrOntoQA is public since 2022 and plausibly in Qwen2.5's pretraining mix; the cohort filter selects model-solved problems, so memorized proof templates could inflate gold validity and interact with injection response. B handles contamination for the *new* regime (random programs + perturbation-novel CRUXEval) but no plan adds even a sentence for the primary domain. Cheap mitigation: note that worlds in EXPD are freshly generated nonce-vocabulary ontologies (A's generator), so the deconfounded results are contamination-proof even if the legacy ones aren't — a free argument A's design already earns but never states.
8. **Stats-framework reconciliation text** (legacy cluster-bootstrap logistic vs A's GLMMs) — flagged as P-grade in §1; assign to the regression appendix owner.
9. **External timestamped pre-registration.** The house PLAN.md-in-repo convention is good but self-hosted; an OSF or commit-hash-published timestamp for PLAN2.md / PLAN_PROGTRACE.md costs nothing and hardens the "pre-registered" claims TMLR reviewers are invited to check.

**P3 (polish):**
10. Reproducibility statement formatting per TMLR template (D App J has the content; needs the named section), compute reporting consolidation (A §6 + B §4 + D App J into one table), figure accessibility (colorblind-safe palettes for Figs A–C / Fig 2), and a data-availability note distinguishing releasable artifacts (summaries, specs) from cluster-only raw JSONLs (C explicitly notes raw rows exist only on skampere2 — the release plan must say what a reader can actually re-run).