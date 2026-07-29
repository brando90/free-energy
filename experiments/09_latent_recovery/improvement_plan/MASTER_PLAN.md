# MASTER IMPROVEMENT PLAN — latent-recovery paper → TMLR (ICLR-gated)

Synthesized 2026-07-01 from an 17-agent analysis of the paper source, experiment code, locked artifacts, and Brando's feedback. Supporting documents in `reports/` (this directory): `storyPlan.md` (restructure), `validatorPlan.md` (validator spec + strict metrics), `causal2.md` (EXPD/EXPE causal-isolation design), `regime2.md` (EXPG program-trace regime), `verdict.md` (final adversarial review), `skeptic.md`/`feasibility.md`/`completeness.md` (first-round critiques), `validator2.md`/`experiments2.md`/`perturbations2.md` (code-grounded reference).

**Bottom line.** Adversarial-review verdict on the combined program: *"accept-with-minor-revisions at TMLR with high confidence"* — conditional on the five design amendments in §3 (all pre-GPU edits) and the two artifact corrections in §1 (both recomputable by any reviewer from the promised release). ICLR remains a gated option (§7).

---

## §1. CRITICAL — two recomputable artifacts that must be fixed before any prose (both verified this session)

**C1. Figure 1's closure-valid dose-response is substantially a parse-failure artifact.**
Recomputed from `results/validated_summary_negstep.json` + `neghop{2..5}.json` (early position, k=1..5):
- unparsed rates: 0.115 / 0.262 / 0.323 / 0.240 / 0.246 (vs ~0 for k=∞)
- closure-valid, all-runs denominator (as plotted): 0.646 / 0.562 / 0.531 / 0.594 / 0.594 — the plotted "decline" (not even monotone)
- closure-valid, parsed-only: **0.730 / 0.760 / 0.784 / 0.781 / 0.788 — flat-to-increasing**
- doubt (raw text, pre-parse): 0.392 / 0.323 / 0.231 / 0.229 / 0.246 — **the doubt gradient is real and survives**

Fix (zero GPU): re-analyze the neghop suite as a 6-class multinomial with unparsed visible + best/worst-case imputation bounds; **retire the legacy validity curve, keep the doubt curve**; regenerate the validity gradient from EXPD (§3) with hardened parsing; per-cell parse rate becomes a pre-registered manipulation check everywhere. Note k=4,5 n-collapse (n=96/69 early, 38/22 mid) must be disclosed.

**C2. EXPC's n=7 was a self-imposed cap, not a grammar bottleneck.**
`expc_polarity_control.py:1476` defaults `--target 7`; `run_metadata.json` records `target_fully_matched_problems: 7`; the locked eligibility audit shows **127 fully-matched 2×2 problems available** (RESULTS_LOCK). The paper text ("the rigid ontology grammar … bottlenecking the fully matched cohort to 7 problems", main.tex:334-336) is contradicted by the project's own lock file — true only for the *strict categorical* cell. Fix: rerun as `EXPC_POLARITY_CONTROL_FULL --target 127` (~1,651 generations, 1–2 GPU-h, zero new code, new result dir) and correct the caption. This converts the "underpowered polarity control" (n=21/cell) to n≈381/cell.

**Also fix before submission (same class — internally inconsistent record):**
- `validator.py:16-17` docstring says unparsed runs are "excluded from headline stats"; the code retains them in denominators (validator.py:204-216, and RESULTS_LOCK confirms). Fix the docstring; disclose the denominator convention globally.
- EXPA wording: PAPER_CHANGELOG calls EXPA "partial/infeasible"; RESULTS_LOCK says target reached (1800 rows) with a stale-writer anomaly (raw 4832 vs 1800). One reconciliation paragraph in Appendix D + re-lock with hashes.
- Mistral sweep: locked but n=4 (negstep/paraphrase) and **n=1** (falsehood) per cell — exclude with one cohort-collapse sentence. Silent omission and naive inclusion are both liabilities.

## §2. Unpublished assets discovered (locked on disk, absent from the paper)

1. **Doubt-forcing pilot** (`results_doubt/force_summary.json`, n=35): forcing a single "Wait" token after a global falsehood moves closure-valid 0.286→0.714 and poisoning 0.429→0.029; forced neutral token does nothing. **The strongest causal-mediator evidence in the project.** Caveat the pilot's "no cost" reading: wait-on-true = 0.629 valid vs ~0.856 unforced true-interruption — forcing plausibly costs ~20pp on true content; and suppression data (doubt 0.246→0.046, validity 0.654→0.615) shows doubt *tokens* are nearly epiphenomenal to validity. Powered rerun = EXPF (§3).
2. **Context-placement pilot** (`results_ctx/summary.json`, n=130/cell): graph-distance doubt decays (0.400/0.331/0.238) while token-distance is flat (0.454/0.392/0.408) — a locked pilot of exactly the token-vs-inferential-distance question; cite as prior for EXPD's T-arm.
3. **32B doubt-judge validation** (`results/doubt_judge.json`): judge/lexical agreement 0.76–1.00 — restore to main text as the doubt-detector defense (professor's sub-bullet). Note agreement is lowest (0.76–0.80) in the high-doubt families → judge-rated doubt becomes the primary DV in all new confirmatory tests.
4. **Truth-audit inversion cross-tab** (UPDATE.md: true-injection 0.868 valid/0.000 poisoned vs false 0.257/0.457; 92% of naive "off-path" injections are entailed-true): dropped between v3 and the current draft. Restore — it is the paper's best methodological selling point and the "audited" in the protocol name.
5. `gsm8k_gradient.py` — written, never run. **Warning (verdict MAJOR-3): as written its k is not a manipulation** (injection site depends only on position; k only filters). ~10-line fix (choose site s.t. steps-remaining == k) before running, or relabel as position-only.

## §3. New experiment program (all in new full-schema result dirs; locked dirs never re-run)

Namespaces: **EXPC_FULL** (polarity rerun) · **EXPD** (matched refutation-distance factorial, generated worlds) · **EXPE** (evidence-mover, natural worlds, question-side) · **EXPF** (powered doubt-forcing) · **EXPG** (program-execution-trace regime). One master `PLAN2.md` pre-registration, externally timestamped (OSF or public commit), single Holm-corrected confirmatory family.

**Causal variable (single canonical owner):** refutation distance d = BFS depth over direct rule applications from the visible prefix state to the complement of the planted claim — `shortest_rule_distance` (expc_polarity_control.py:340-356). Fail-closed audit `measured d == designed d` in every new cell; retro-computed for all locked injections.

### EXPD — deconfounded gradient (core instrument; full spec in reports/causal2.md §1.1 + verdict amendments)
Generated worlds: goal chain C0→…→C_L (L≈6–8 per verdict minor-c) + off-path refutation spur A0→…→A_{d-1} + refuting rule head = complement of the planted claim; question length/rule-count/token-frequency pinned across d; planted sentence **byte-identical across d** in the affirmative family (only the world's rule graph moves the counterevidence — the negation-free dose-response Brando asked for). Mirrored quadruples (worlds differing by one polarity token) give truth×polarity decoupling with everything else pinned.
**Verdict amendments (mandatory, pre-GPU):**
- MAJOR-1: attribute cells cannot exist at d=∞ (attribute truth audit is open-world; decidability ⟺ finite d). Attribute gradient runs d∈{0,1,2,3,5} only; all ∞ anchors move to the categorical family (`cat_false_∞`, genuinely CWA-false); print as grammar theorem T3 beside T1 (on-path×affirmative×false is unconstructible) and T2 (attributes derivationally inert by grammar).
- MAJOR-2: the usable-categorical reuse branch must route through ≥1 **fresh intermediate category** underivable without the planted atom ("Every zumpus is a florpus. Every florpus is a C_k.") + world-audit assert, else injection-dependence is mechanically zero and H4 is flat by construction. H4's DV = derivational injection-dependence (M4 split), stated-reliance secondary.
- MAJOR-4: H1's confirmatory slope runs over d∈{1,2,3,5}; d=0-vs-d=1 is its own two-sided pre-registered contrast (locked data show d=0 doubt < d=1 — the design must not assume the monotonicity its own Table 3 contradicts).
- Mirrored-quadruple cohort: condition confirmatory contrasts on jointly-solved quadruples AND identical gold prefixes up to the injection site (or report divergence rate as covariate) (MOD-13).
- Anchors: benign-paraphrase + true-interruption in every world batch; TOST gate scoped to cross-generator continuity claims only (MOD-12); per-cell parse-rate manipulation check (C1 lesson; legacy negated families unparse at 11–36% vs ~0–3% affirmative).

### EXPE — evidence-mover (highest evidence per GPU-hour)
Legacy gold cohort; fixed global falsehood in the trace; the **question** gains exactly one rule sentence per arm (REFUTING_d1/d2/d3, FREQ_MATCHED_NONREFUTING, TEMPLATE_IRRELEVANT, BASELINE_FILLER) — trace byte-identical across arms, so the EXPB 0.317-unparsed imputation problem cannot arise by construction. FREQ vs REFUTING is the kill shot for the lexical-overlap confound; it also adjudicates EXPB's unexplained irrelevant-cert inj-dep rise (0.240 vs 0.153) as priming or not. H5 amendment (MOD-7): pre-register as the conjunction (REFUTING > FREQ) ∧ (REFUTING > TEMPLATE_IRRELEVANT) in one Holm slot; print the empty-cell note (entailed-antecedent × refuting-head × non-refuting is unconstructible).
Disclose (MOD-11): arms replay the own-trace under an augmented question (equally off-policy across arms; report re-solve rate).

### EXPF — powered doubt-forcing (mediator; unlocks ICLR)
n=150 × 3 positions, arms {force-wait, force-neutral, baseline, unforced-true, **forced-wait-on-true** (MOD-8)} → within-cohort 2×2 {forced,unforced}×{true,false} quantifying the ~20pp cost; restart-vs-local-check content coding; framed as intervening on check-*initiation*. Reuses `doubt_suppress.py` STAGE=force (~1 day plumbing).

### EXPG — program-execution-trace regime (weakness #5; full spec in reports/regime2.md)
Interpreter-validated restricted-Python traces; two-world (σ_true vs σ_cf) validation makes injection-dependence measurable for **every** family (fixes the † asymmetry); no goal anchor → parroting structurally absent; polarity/statement-type constant by construction. Distance decomposes into re-read distance k_r × recompute depth k_c, with an **op-free k_c=0 row** (copy/alias chains — checking is a zero-compute re-read) as the d≈0 analogue and the floor-gate insurance: Stage-1 gate G-B decides full grid vs pre-registered scope-bounding branch.
**Verdict amendments:** MAJOR-5 — Axis-R chain roots must be computed values (≥2 listing constants combined), never listing literals, else the listing provides an O(1) re-read path that flattens k_r; add manipulation check (does doubt cite listing vs trace lines?). MOD-9 — fixed planted-δ policy (last-digit-preserving for Axis-C; typo-plausible digit swaps for Axis-R) + δ-sensitivity satellite; extend distinct-values constraint to the planted value. MOD-10 — reject worlds where σ_cf(out) == σ_true(out).
Matched-strictness reporting both ways (PrOntoQA hop-sound-valid beside trace-valid; final-output-valid beside closure-valid) + D14 header fix on the regime table. Lean rejected this cycle with printed reasons (cohort collapse under frozen-greedy own-trace protocol); replace main.tex:488-489 promise with concrete future-work sketch. Chained arithmetic re-presented as the regime's degenerate corner; GSM8K demoted to corroboration.
**ICLR unifier (verdict's top recommendation): EXPG_FORCE** — force " Wait," after the planted value in EXPG's deep/k_r=8 cells (2–3 cells × 150, ~2 GPU-h): if check-initiation forcing rescues absorption in *both* regimes, the one-big-thing story becomes "nearby evidence gates whether checking is initiated, and initiating the check substitutes for proximity — in two formally validated regimes."

### Master confirmatory family (single Holm family in PLAN2.md; everything else explicitly exploratory)
H1 negation-free doubt gradient (EXPD aff_false_attr, slope d∈{1,2,3,5}) · H2 polarity equivalence at matched d (TOST ±0.10) · H3 truth-not-negation (neg_true≈aff_true TOST ∧ neg_false_d1>neg_true_d1) · H4 confound-free reuse gradient (cat_false_usable, amended) · H5 inferential-vs-lexical conjunction (EXPE) · H6 within-problem flip (EXPE) · H7 EXPG op-free k_r pair (doubt ∧ absorption, imputation-bounded) · [H8 EXPF initiation effect iff promoted to main]. Print the Holm denominator once. Judge-rated doubt (32B, adapter) is the primary doubt DV everywhere; lexical regex secondary with per-cell agreement. Belief probe dropped from confirmatory (IV-in-measurement).

## §4. Validator + terminology (weakness #3, #4 — reports/validatorPlan.md; zero GPU)

- Promoted §3 "Validator": formal grammar, atom representation, closure fixpoint, the CWA asymmetry stated exactly (closed-world for positive categoricals in the *audit*; open-world for negations/attributes; the validator itself never assigns truth), precedence ladder (unparsed ≻ derailed ≻ poisoned ≻ parroted ≻ valid), poison propagation, denominator conventions, doubt detection on raw text (order provably irrelevant — lexicons disjoint). Pseudocode appendix auto-generated from code (`validator_spec_export.py`) so spec and code cannot drift. Known holes disclosed as propositions: distractor-family inj-dep undetectable (corrupted rule → parse_fact None) — fixed retroactively by the v2 `used_injected_rule` flag; fabricated-rule restatements skipped; entity = first token.
- Strict metrics from stored generations (zero GPU, cluster-side — raw JSONLs are NOT in the local mirror): strict-repair rate, hop-sound validity, goal-jump rate (restore the 4.7–9.2% numbers), stated-reliance (the `d45c87be` case), unconditional injection-use with echo/derivational split (fixes the misleading 0.000† cells). **Hard gate: run strict metrics and C1/C2 re-analyses FIRST; freeze abstract/figure wording only after the numbers exist.** Strictness cannot touch the doubt gradient (0.336 vs 0.013) or inj-dep gradient (0.016 vs 0.342) — the phenomenon survives re-centered on flagging + absorption even if "repair" shrinks.
- Terminology: one merged replacement map (validatorPlan §2.3 is the single source of truth; storyPlan Part 3 merged into it), lint script as enforcement; "recovery" banned as a metric term; "repair" only for flagged/routed corrections. Definitions box in §3 of the paper.
- Human audit: ~380–430 stratified labels, κ-gated, **second independent rater** (not an author) for the headline κ.

## §5. Restructure (weakness #2 — reports/storyPlan.md, with verdict CONSISTENCY-1 amendments)

Target ~14.5pp main + ~14pp appendices on the TMLR class (retarget from colm2026_conference.sty). Professor's outline mapped section-by-section; every experiment has an explicit fate (storyPlan Part 2 fate table). **Hard main-text budget (replace, don't add):** EXPD/EXPE replace the legacy validity curve and the EXPC pilot in main text; EXPG gets exactly one subsection + regime map + one figure; satellites → appendices.
**storyPlan amendments before any prose is drafted from it** (it predates the verdict): evidence-leg 2 and the abstract draft must drop "closure-valid declines then plateaus" (C1 — doubt-only until EXPD lands); fate-table row "Fig 1 → MAIN §4.2" superseded by EXPD Panel A/C; §4.4/§5.7 doubt-forcing wording must carry the ~20pp true-content cost and initiation-vs-ability nuance; ICLR criterion #1 wording → attribute/negative-head route (not "disjoint-category grammar"); `EXPD_DOUBT_FORCE` → `EXPF`.
New decisive Figure 1 (three panels): A — deconfounded gradient (byte-identical sentence, doubt + reuse curves, parse-rate rug); B — within-problem evidence flip (EXPE paired slopegraph + EXPB inset with bounds); C — legacy reconciliation (neghop 6-class stacked multinomial, validity curve retired, doubt overlay + 32B + EXPC-127 inset). Plus the design-matrix table (conditions × confound axes: varies/pinned/structurally-empty-T1/n.m.-T2/T3).

## §6. Paper-side items no design owns (from completeness report — assign explicitly)

P0: (1) hard page budget enforced by one editor (weakness #2 recurrence is the program's biggest self-inflicted risk); (2) **double-blind anonymization scrub** — RESULTS_LOCK hashes contain `/lfs/skampere2/0/eobbad/...` paths and the repo name; anonymized artifact release (anonymous.4open.science) required; (3) related-work refresh vs Lanham 2023 (un-audited mistake-insertion — position "audited" against it explicitly), Huang ICLR'24, Tyen 2024 (mistake-finding vs -correcting), PRM/step-verifier line, R1-era backtracking/"Wait" analyses (directly relevant to EXPF).
P1: Broader Impact statement; environment pinning (HF revisions, dtype, vLLM version + nondeterminism caveat for one-seed greedy); PrOntoQA generator license/provenance + model licenses (Qwen-72B differs from 7B; Llama community license constrains releasing continuations).
P2: contamination sentence (EXPD/EXPG worlds are freshly generated nonce ontologies / random programs → contamination-proof — a free argument the designs earn but never state); stats-framework reconciliation paragraph (legacy cluster-bootstrap logistic vs new GLMMs); external timestamp for PLAN2/PLAN_PROGTRACE.

## §7. Compute, engineering, timeline

Hardware verified this session: skampere2 has **8× NVIDIA H200 (144GB), all idle**. Harness is HF transformers batch-1 (no vLLM in repo — G1): write `src/vllm_gen.py` (~120 lines, 0.5–1 day, shared by all suites) with `shard_launch.sh` fallback; one owner for judge adapter + launcher (verdict minor-d).
GPU totals (HF worst case): EXPC_FULL 1–2 h · EXPE ~2 h · EXPD ~15–20 h + pilot · EXPF 2–3 h · 32B judge passes 15–30 h · EXPG 45–70 h + judge 10–20 h · 32B replications 35–70 h (optional) ⇒ **~85–130 GPU-h ≈ ~2 node-days worst case; single-digit hours after the vLLM port.** Engineering is the binding constraint: ~2.5–3 weeks combined (EXPD generator+runner ~3–4 d; EXPE ~1 d; EXPF ~1 d; EXPG ~5–8 d; adapters/GLMM/figures ~2 d; vLLM port 0.5–1 d).

Sequenced timeline (no TMLR deadline; ICLR 2027 ~late Sept 2026):
- **Wk 1 (Jul 1–8): Stage-0 zero-GPU gates on skampere2** — neghop multinomial (C1), EXPC caption + `--target 127` rerun (C2), retro distance audit, strict metrics, EXPB bounds, docstring/EXPA-wording fixes. Freeze figure/abstract claims. In parallel: PLAN2.md (with MAJOR-1/2/4 amendments baked in) + external timestamp; vllm_gen.py.
- **Wk 1–2:** writing restructure to TMLR skeleton (amended storyPlan); Validator section + Appendices A/B/D; terminology lint pass; validator self-audit labeling.
- **Wk 2–3:** EXPD pilot → full; EXPE; EXPF; 32B judge passes.
- **Wk 3–5:** EXPG Stage 0–2 (+ EXPG_FORCE cells); gsm8k_gradient fix + run; 32B replications as slack allows.
- **Sep 1 — ICLR go/no-go:** flip iff H1+H2+H4 clean ∧ EXPF lands with matched controls ∧ EXPG shows the gradient (G-B passed). Go → freeze, write ~9pp ICLR variant (storyPlan §5 cuts), submit late Sept; TMLR after decision or as the non-overlapping protocol paper. No-go → submit TMLR ~mid-Sept/Oct with the REQUIRED set (which already fully answers weakness #1).

## §8. What the paper's story becomes

Core claim (defended sentence): *Using an audited perturb-and-validate protocol — every planted step verified false against the rule closure, every continuation formally validated — whether a language model absorbs, flags, or routes around a false step planted in its own reasoning trace is strongly modulated by the availability of refuting evidence in the local context, not by falsity alone.* Three decisive legs: the matched audited sweep (EXPA, locked); the **deconfounded, negation-free dose-response** (EXPD, byte-identical planted sentence); the **within-problem evidence-mover** (EXPE + EXPB recast vs irrelevant arm). Mediator (EXPF) and second regime (EXPG) either upgrade this to the ICLR story ("nearby evidence gates whether checking is initiated; initiating the check substitutes for proximity — in two formally validated regimes") or land as TMLR §4.4/§5.3 strengtheners. The doubt/validity dissociation across lineages stays as the interface-vs-mechanism finding.
