All verification is complete. The load-bearing facts check out on disk: zero negative-category rules in ProofsOnly (present only in Composed), `parse_rule` already accepts `not_cat` rule heads, EXPC's availability audit shows 127 fully-matched problems vs 7 generated, and the skeptic's N1 recomputation reproduces exactly (parsed-only neghop validity at early = 0.730/0.760/0.784/0.781/0.788 for k=1..5 — flat-to-increasing, so the plotted validity decline is an unparsed-rate artifact, while the doubt gradient .392/.323/.231/.229/.246 survives). Composing the design.

---

# CAUSAL-ISOLATION EXPERIMENT PROGRAM — isolating local refutability from the professor's confound list (Weakness #1)

Mirror: `/private/tmp/claude-501/-Users-elyas-Desktop-Alara/ad633660-4a3e-4cbc-99dd-093a0feae95e/scratchpad/latent_recovery/` (canonical: `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`). All `file:line` cites verified against the mirror. All new runs in **new result dirs** (RESULTS_LOCK compliance); all numbers regenerated only via `src/paper_artifacts.py`.

## 0. Frame, adjudications, and the causal variable

**Causal variable (canonical, single owner).** Refutation distance `d(a*) =` BFS depth over **direct** rule applications from the visible prefix state needed to derive the complement of the planted claim `a*` — exactly `shortest_rule_distance` (`expc_polarity_control.py:338–355`), i.e., validatorPlan's Alg 5. `d=0`: complement already in state; `d=k`: k rule applications; `d=∞`: complement underivable, falsity certified only by closed-world closure search. Every new condition is **audited fail-closed against this function** (`assert measured d == designed d`, EXPB `is_fully_audited` pattern), and it is retro-computed for every locked injection (validatorPlan M5). This resolves the completeness-report conflict #3 (two competing distance definitions) in favor of the code-native one.

**Confound axes to be cut loose from d** (professor's list, line 19/21 of PROFESSOR_FEEDBACK.md): (A1) negation presence, (A2) lexical contradiction/overlap, (A3) statement type (fact/rule; categorical/attribute), (A4) on/off-path relevance, (A5) derivational usability, (A6) stylistic change / step deletion, (A7) category frequency. In the locked families these are almost perfectly aliased with d: every finite-d family is negated+on-path+goal-blocking; the only d=∞ family is affirmative+off-path+categorical.

**Two grammar theorems to state in the paper (they turn missing cells into propositions, not gaps):**
- **T1 (empty cell):** an on-path positive entity fact is entailed-true by construction; ON-PATH × AFFIRMATIVE × FALSE is logically unconstructible in this fragment. Print it in the design-matrix table.
- **T2 (usability):** rule bodies are always positive categories (`validator.py:33–54`), so attribute claims are derivationally inert *by grammar* and categorical claims are the only usable ones. Hence polarity purity is tested on attributes, reuse on categoricals — by necessity, not choice.

**Namespace adjudication** (completeness §2.1, skeptic W2): **EXPD** = deconfounded matched gradient (generated worlds); **EXPE** = evidence-mover (natural worlds, question augmentation); **EXPF** = powered doubt-forcing rerun (Plan D's asset, renamed); **EXPG** = program-trace regime (Plan B's lane, name reserved, not designed here); **EXPC_POLARITY_CONTROL_FULL** = the `--target 127` rerun. One master **PLAN2.md** pre-registration with a **single confirmatory family** (skeptic N3), externally timestamped (public commit hash or OSF; completeness P2-9).

**Hard gate ordering (skeptic's minimum set, item 1):** Stage 0 zero-GPU re-analyses run **first** and freeze figure/abstract claims before any prose: (i) neghop 6-class multinomial + parse-rate re-analysis (N1 — verified here: parsed-only validity is flat 0.730→0.788, doubt gradient survives), (ii) retro refutation-distance audit of all locked injections (M5; may reveal "global" falsehoods with finite d via negative-head rules — re-bin or disclose), (iii) strict metrics (validatorPlan M1–M4), (iv) EXPB unparsed imputation bounds (N9). The legacy **validity** dose-response curve is retired; the legacy **doubt** curve survives as corroboration.

---

## 1. Factorial matched-perturbation design

### 1.1 EXPD — matched refutation-distance factorial on generated worlds (the core instrument)

**World construction** (new `src/gen_worlds_expd.py`; emits `prontoqa_ood`-schema JSON so `build_dataset.py`/EXPA-style ingestion works unchanged — feasibility §2.6; the vendored `exp/prontoqa/theory.py` supports negation/disjointness but its random topology fights matching, so use a purpose-built deterministic constructor with the same nonce-noun vocabulary):

- **Goal chain:** premise `E is a C0.` + rules `C0→C1→…→C_L` (L=8–10, supporting early/mid/late injection and d≤5), target = chain-end predicate. Nonce `-pus` nouns drawn without replacement.
- **Refutation spur (the d dial):** second premise `E is an A0.` (A0 **off** the goal chain) + rules `A0→A1→…→A_{d−1}` + terminal refuting rule whose head is the complement of the planted claim. The continuation walks the C-chain and never derives the A-spur, so d **stays static through the continuation** (kills skeptic W1-attack-3; the on-path variant where the spur attaches at `C_{j+d}` is a satellite with a pre-registered doubt-onset-timing analysis).
- **Length/frequency padding:** each world carries exactly `D_max` spur+filler rules — d spur rules plus `(D_max − d)` filler rules over entity-disconnected nouns — so question token count and rule count are constant across d (kills "more rules = harder recall"). The planted predicate token appears **exactly once** in the question (the refuting rule head) at every d; all nouns are nonce (corpus frequency ≈ 0). Frequency covariates logged anyway (A7).
- **Mirrored quadruples (polarity × truth decoupling):** world A has spur head `A_{d−1}s are not bright.`; world B is byte-identical except `A_{d−1}s are bright.` Cells per (d, position):

| Cell | World | Injected sentence | Truth | Polarity | d |
|---|---|---|---|---|---|
| `aff_false_attr_d` | A | `E is bright.` | false (opposite entailed at d) | + | d |
| `neg_true_attr_d` | A | `E is not bright.` | true | − | (entailed at d) |
| `neg_false_attr_d` | B | `E is not bright.` | false | − | d |
| `aff_true_attr_d` | B | `E is bright.` | true | + | (entailed at d) |

The two **false** cells differ by one token ("not") across worlds differing by one token — template, length (±1), position, path status, usability (inert, T2), frequency, style all pinned. **Crucially, within `aff_false_attr` the planted sentence is byte-identical across all d — only the world's rule graph moves the counterevidence.** This is the negation-free dose-response the professor asked whether we could produce.
- **d levels:** {0, 1, 2, 3, 5, ∞}. `d=0` = the complement stated as a question **premise** (`E is not bright.` among the facts) — the first *affirmative-injection* d=0 cell, which directly addresses the locked d=0<d=1 doubt non-monotonicity (contradiction 0.231/0.269/0.177 vs one-hop 0.467/0.273/0.267, `paper_results.tex:51–57`) that the skeptic notes every plan silently skips. `d=∞` = no refuting rule anywhere (CWA-only falsity; disclosed as the one cell where falsity *strength* differs — inherent to the construct).
- **Categorical reuse suite (the flagship absorption DV, skeptic W1-attack-1):** same spur geometry with **negative-category heads** (`Every a_{d−1} is not a zumpus.` — in-grammar per `parse_rule` `not_cat` branch, verified; absent from natural ProofsOnly data, verified — this is exactly why EXPC's strict cell starved and why generation is required). Inject `E is a zumpus.` (affirmative, categorical, false at d). Cross with **usability**: `cat_false_d_usable` (world also contains `Every zumpus is a C_k.` feeding the goal chain — injection-dependence mechanically measurable, poison propagation live) vs `cat_false_d_inert` (zumpus has no outgoing rules). Include `d ∈ {1, 3, ∞}` × {inert, usable} at 2 positions, n=150 — the confound-free **reuse gradient** promoted to core per the skeptic's preempt.
- **Anchors & manipulation checks in every world batch:** `benign_paraphrase` (reuse `make_paraphrase`, `common.py:109–113`) and `true_interruption` (reuse EXPA's `make_true_interruption` logic) at 3 positions — pre-registered **equivalence gate vs EXPA levels (TOST ±0.10)** before any confirmatory test (skeptic N11), plus per-cell **parse-rate manipulation check** (legacy negated families unparse at 11–36% vs ~0–3% affirmative; the polarity contrast inherits differential measurement error unless parse rates are shown matched — report and, if unmatched, run the doubt-judge over unparsed rows and report multinomial bounds).
- **Style micro-satellites (A6):** `exact_repetition` (step restated, no marker) and `marker_false` (discourse marker + falsehood) at mid, n=150 — decomposes benign-paraphrase's two simultaneous differences (marker; step preserved).
- **Rule-type satellites (A3, rescues the distractor family):** `rule_false_d0` (rule contradicting a stated rule: `Every C_j is bright.` when the question states `C_js are not bright.`), `rule_false_far` (CWA-false rule), `rule_true_offpath` (verbatim restatement of a real off-path rule). Requires the ~20-line rule-truth audit extension (false iff closure entails the opposite head for that body; else CWA/unknown) and a v2 derived flag `used_injected_rule` (fact derivable only with the injected rule added to R) — which retroactively makes injection-dependence **measurable for the locked distractor family too**, replacing the current dagger/`---` cells (`paper_artifacts.py:178–185`).
- **Token-distance T-arm:** spur rules placed front/mid/back of the question at fixed d — affirmative-family replication of the locked `results_ctx` pilot (G-decay .400/.331/.238 vs flat T .454/.392/.408, n=130/cell — cite as prior, promote provenance per storyPlan flag 2).

**Protocol invariants (all cells):** model's own gold rollout, gold must be `solved` ∧ `valid_rederivation` (double filter); injection = replacement of the step at `si` (paraphrase excepted); greedy, `max_new=192`, prompt bytes SHA-pinned (EXPC pattern); fail-closed audits per cell: truth status as designed, `shortest_rule_distance == d` exactly, planted-token question frequency == 1, gold still validates, mirrored quadruple jointly available at all positions.

**Cell × position × n table (primary):** `aff_false_attr`, `neg_false_attr` × d{0,1,2,3,5,∞} × 3 positions × n=150; `aff_true/neg_true` × d{0,1,3,∞} × mid × 150; `cat_false` × {inert,usable} × d{1,3,∞} × 2 pos × 150; anchors ×3 pos ×150; satellites (style 2, rule 3, on-path 3, T-arm 3) × mid × 150. ≈ **64 cells ≈ 9,600 perturbed generations + ~4,000–4,500 gold rollouts** (mirrored-quadruple joint-solve oversampling ~2×).

### 1.2 EXPE — evidence-mover on natural PrOntoQA worlds (byte-identical stimulus; bridges legacy results)

Runs on the **existing legacy gold cohort** (168 gold-validated instances + `make_falsehood` candidates — no new gold collection). Fixed global falsehood `F = "E is a fcat."` injected into the model's own trace at mid (± early/late for d=1); the **question** (not the trace) is augmented with exactly one rule sentence, so `F`, the prefix, and the entire trace-side stimulus are **byte-identical across arms** — stronger than EXPB, whose manipulation inserted a trace sentence and inherited the 0.317-unparsed hole (N9-immune by construction):

| Arm | Added question rule | Audit invariant | What it isolates |
|---|---|---|---|
| `REFUTING_d1` | `Every g is not a fcat.` (g entailed for E from prefix at 0 hops) | ¬F derivable at d=1; F flips CWA-false → provably-false (disclosed) | rule-mediated local refutability |
| `REFUTING_dk` (k=2,3) | head antecedent h entailed at k−1 hops | measured d == k | dose-response on identical F |
| `FREQ_MATCHED_NONREFUTING` | `Every z is not a fcat.` (z never entailed for E) | does **not** locally falsify; mentions `fcat` once, same as refuting arms | **lexical co-presence of "not a fcat" without derivability — the decisive lexical-axis (A2) control** |
| `TEMPLATE_IRRELEVANT` | `Every g is not an icat.` (icat not reachable from fcat) | true, entailed, non-refuting — EXPB's irrelevant-cert semantics | negation-token + true-negative-knowledge control |
| `BASELINE_FILLER` | `Every x is not a y.` (x,y entity-disconnected) | inert | equalizes question length/negation count (fixes EXPB's +1-sentence asymmetry) |

All arms +1 rule, same template, same polarity, length-matched; rule inserted at a fixed slot ("back", just before the facts — `results_ctx` shows token position is inert, and `move_rule` (`context_check.py:60–78`) is reusable). Gold-revalidation and no-accidental-refutation asserts fail-closed. The FREQ vs REFUTING contrast also adjudicates EXPB's unexplained irrelevant-cert inj-dep rise (0.240 vs 0.153): if a non-refuting mention of `fcat` raises reliance, that is priming, now measured. ~6–8 cells × 150 ≈ **1,000–1,200 generations**.

### 1.3 What exists vs what must be written (file-level)

**Exists — reuse:** `shortest_rule_distance`/`directly_derivable`/`world_state`/`audit_truth_status`/`grammar_template` (`expc_polarity_control.py:150–390`); fail-closed audit + paired-triple + `prompt_sanity` + manifest/lock scaffolding (`expb_local_cert_flip.py`); `make_negstep`/`make_neghop`/`make_paraphrase`/`make_falsehood` (`common.py:109–174`); `make_true_interruption` + eligibility ladder + `wilson`/`cluster_boot_rate`/fixed-effect logistic (`expa_global_expansion.py:738–953`); `injection_points` (`perturb.py:48–61`); rule-relocation (`context_check.py:60–89`); `doubt_judge.py` (needs adapter); `doubt_suppress.py` STAGE=force (EXPF core); EXPC runner as-is for the 127 rerun; `expa_tests.py`/`expc_tests.py` fixture pattern; vendored `exp/prontoqa/` (vocabulary + schema reference).

**Write — new (with honest sizes):** (1) `src/gen_worlds_expd.py` ~300–400 lines (deterministic chain+spur+filler constructor, mirrored quadruples, world-level audit manifest); (2) `src/expd_matched_gradient.py` ~700 lines (EXPA/EXPC-skeleton clone: conditions, fail-closed audits incl. `d`-assert, resume-safe generation, validated outputs, summary tables → `results/EXPD_MATCHED_GRADIENT/`); (3) `src/expe_evidence_mover.py` ~500 lines (EXPB clone with question-augmentation arms → `results/EXPE_EVIDENCE_MOVER/`); (4) `src/judge_adapter.py` (hours; G3 — judge is hard-wired to legacy `results/<dir>/runs.jsonl`); (5) `src/reanalyze_neghop.py` ~150 lines, zero-GPU (multinomial, parse rates, best/worst-case bounds; **needs cluster JSONLs** — mirror has none; run on skampere2 after `kinit`, or one rsync); (6) `src/mixed_models.py` ~200 lines (GLMM via `pymer4`/lme4 or statsmodels — new cluster dependency — extending `expa`'s logistic); (7) generation backend: either `scripts/launch_sharded.sh` (`CUDA_VISIBLE_DEVICES` sharding, hours) or a vLLM port of the 4 rollout loops (0.5–1 day) — **there is no vLLM in the repo** (feasibility G1); the harness is HF batch-1; budget this explicitly; (8) rule-truth audit + `used_injected_rule` (~35 lines into validator-v2 per validatorPlan, shared); (9) `PLAN2.md`; (10) `paper_artifacts.py`/`make_workshop_figures.py` extensions (lock policy: all figures/tables via these two, RESULTS_LOCK:111). EXPF: `src/expf_doubt_force.py` (~1 day plumbing: EXPA-style pool + arms {force-wait, force-neutral, baseline, wait-on-true-matched-unforced} × 3 pos × n=150).

---

## 2. Statistical analysis plan (pre-registered in PLAN2.md)

**Model.** Mixed-effects logistic per DV: `outcome ~ d + polarity + d:polarity + position + covariates + (1 | world_family)` (world_family = mirrored-quadruple/base-chain id; one instance per world, so quadruple is the clustering unit). Confirmatory contrasts restricted to **jointly-solved mirrored quadruples** (skeptic residual: the double filter must not select different cohorts across polarity cells); per-world-version solve rates reported. Covariates logged everywhere and used in robustness fits, not headline: planted-sentence token length, template class, in-question planted-token count (constant=1 by design), parse-rate cell means, injection `sent_idx`. Legacy tables keep the cluster-bootstrap logistic; one appendix paragraph reconciles the two frameworks (completeness P2-8).

**Primary DVs.** (P-doubt) **judge-rated doubt** (Qwen2.5-32B judge, adapter over new suites; lexical regex secondary with per-cell agreement — skeptic N10); (P-reuse) **injection-dependence** (validator class + M4 unconditional/echo-vs-derivational split); (P-repair) **strict repair** (validatorPlan M1). The belief probe is **dropped from confirmatory** entirely (skeptic W1-attack-2 kill: its measurement has the IV built in); if run, exploratory-only with a trace-free control arm. Multinomial class distribution reported descriptively for every cell with unparsed shown, never buried in denominators.

**Single confirmatory family (Holm-corrected, 6 tests):**
- **H1 (negation-free gradient):** within `aff_false_attr`, judge-doubt slope over d∈{0,1,2,3,5} < 0 (GLMM Wald; ∞-vs-5 as a separate contrast since ∞ is not numeric).
- **H2 (polarity equivalence at matched d):** `aff_false_attr_d` vs `neg_false_attr_d` doubt, TOST margin ±0.10, pooled over d∈{1,3}.
- **H3 (truth, not the "not" token):** `neg_true_attr_d` doubt ≈ `aff_true_attr_d` (TOST ±0.10) **and** `neg_false_d1 − neg_true_d1 > 0`. The negation-cue account predicts high doubt on `neg_true`; refutability predicts low.
- **H4 (confound-free reuse gradient):** injection-dependence slope over d∈{1,3,∞} within `cat_false_usable` > 0 (reuse rises as counterevidence recedes) — the title-era absorption claim, measured with polarity/path/frequency pinned.
- **H5 (inferential vs lexical, EXPE):** `REFUTING_d1` doubt > `FREQ_MATCHED_NONREFUTING` doubt on byte-identical F (McNemar within problem cluster). This is the kill shot for A2.
- **H6 (within-problem flip replication):** `REFUTING_d1` inj-dep < `BASELINE_FILLER` inj-dep (EXPB's flip reproduced without touching the trace).

Everything else — position interactions, on-path doubt-onset timing, usability main effect on absorption (inert vs usable at d=∞), style/rule satellites, T-arm flatness, d=0 non-monotonicity — is **explicitly exploratory** (estimation with 95% cluster-bootstrap CIs, no stars).

**N and power.** n=150/cell (matching EXPA), 3 positions for the two primary false families → 450/d-level. Anchors from locked data: adjacent-d doubt contrast .323 vs .231 → h≈0.20, two-proportion power ≈0.85 at Holm-α with n=450; the 5-level slope test is far better powered (>0.99 for the legacy-sized logit slope ≈ −0.38/hop). TOST at ±0.10 with n=450/cell: 90% CI half-width ≈ 0.050 — decisive. Mid-only satellites (n=150) are estimation-only by pre-registration. EXPE: 150 paired problems/arm; EXPB's paired McNemar on smaller deltas was p≈0 at 100 triples/position, so 150 is comfortable.

**Confirm / refute (pre-stated outcomes).** *Local refutability CONFIRMED* iff H1, H4, H5 pass and H2/H3 TOSTs pass — the gradient exists with zero negation tokens, reuse tracks d in usable categoricals, and derivability (not lexical co-presence) moves doubt. *Professor's confound account SUPPORTED* (a publishable pre-registered outcome, stated as such in PLAN2.md) iff: H2 fails with neg≫aff and H1 flat (negation is the driver); or H5 fails with FREQ≈REFUTING (lexical priming is the driver); or EXPD gradients vanish while legacy negated gradients reproduce (the legacy curve was a parse/negation artifact — Stage 0 will already have told us the validity half of this). *Partial*: doubt gradient without a reuse gradient → claims re-centered on flagging + absorption boundary, per the skeptic's pre-authorization.

---

## 3. Models and honest compute (8 GPUs on skampere2; harness is HF `transformers` batch-1 — no vLLM exists in-repo; "8×H200" is unverified, hostname suggests Ampere — estimates given for both backends)

**Models.** Primary: **Qwen/Qwen2.5-7B-Instruct** (pinned revision) for every suite. Replication (ICLR arm): **Qwen2.5-32B** on EXPD primary cells (`aff/neg_false_attr` × d × mid, 1,500 gens). Optional 4-cell lineage check (aff_false d∈{1,∞}, mid) on **OLMo-2-7B** and **Llama-3.1-8B** (appendix). **Mistral excluded** with one cohort-collapse sentence (n=4/1 per cell on disk — N5). **R1-Distill excluded** from new suites (protocol non-equivalence, N12 — one protocol note). Judge: **Qwen2.5-32B**.

| Suite | Generations (≈ tokens) | HF batch-1, 8-GPU-sharded | After vLLM port |
|---|---|---|---|
| Stage-0 re-analyses | 0 (CPU on cluster JSONLs) | 0 GPU | — |
| EXPC_FULL (target 127) | 1,651 (~0.3M) | 1–2 GPU-h (~15 min wall) | trivial |
| EXPE evidence-mover | ~1,200 (~0.25M) | ~2 GPU-h | <0.5 |
| EXPD worlds pilot (200 worlds, gates) | ~1,500 | ~2 GPU-h | <0.5 |
| EXPD full (gold ~4.5k + perturbed ~9.6k) | ~14k (~2.7M) | ~15–20 GPU-h (~2–3 h wall) | ~3–4 GPU-h |
| 32B judge passes (~15k continuations) | — | ~15–30 GPU-h | ~4–6 |
| EXPF doubt-forcing powered | ~1,800 | ~2–3 GPU-h | <1 |
| 32B EXPD replication | 1,500 @ ~12 s/gen | ~5–8 GPU-h | ~2 |
| **REQUIRED total** | | **~40–60 GPU-h ≈ ≤1 node-day** | **~10–15 GPU-h** |

Engineering is the binding constraint: generator+audits 1–2 d, EXPD runner 1–2 d, EXPE runner 1 d, adapters/GLMM/figures 1–2 d, optional vLLM port 0.5–1 d ≈ **one focused week**, then ≤1 node-day of GPU.

---

## 4. Re-presenting distance-k and cert-flip "with force": the decisive figure + design-matrix table

**Figure 1 (new, three panels): "Moving the evidence, not the words."**
- **Panel A — the deconfounded gradient (EXPD):** x = refutation distance d (0,1,2,3,5, broken axis, ∞), y = rate. Curves: judge-doubt for `aff_false_attr` (solid; **zero negation tokens anywhere in stimulus**), judge-doubt for `neg_false_attr` (dashed — superimposition is the polarity result), injection-dependence for `cat_false_usable` (dotted — the reuse gradient), `aff_true`/`neg_true` flat-low reference band. Cluster-bootstrap 95% ribbons; per-cell parse rate as a rug annotation; caption line: "the planted sentence is byte-identical across d; only the world's rule graph moves the counterevidence."
- **Panel B — within-problem evidence flip (EXPE + EXPB):** paired slopegraph per problem: baseline-filler → refuting-rule (d=∞→1) for doubt and inj-dep, with FREQ_MATCHED and TEMPLATE_IRRELEVANT arms plotted flat beside; McNemar discordant counts printed; EXPB's locked flip (doubt .023→.380, inj-dep .240→.027 vs irrelevant control) as the inset showing the trace-side version, with its 0.317-unparsed bound band.
- **Panel C — legacy reconciliation:** neghop k=1..5 as a **6-class stacked multinomial** (unparsed visible, validity curve retired — the honest re-presentation N1 demands), doubt overlay (lexical + judge-at-mid), 32B curve, and the EXPC-127 polarity 2×2 as an inset. Caption states plainly: "the previously plotted validity decline is carried by parse failures (parsed-only 0.730→0.788); the doubt gradient is not."

**Design-matrix table (main text, the professor's requested "force and detail"):** rows = every condition (legacy 6 families + EXPD cells + EXPE arms); columns = d, A1–A7 axes; entries = varies / pinned(value) / **structurally empty (T1)** / n.m.(T2). One glance answers "is it distance or negation/lexical/type/relevance/usability/style/frequency?" — each axis has a condition pair differing only on it.

**Cert-flip prose recast (storyPlan §4.3 alignment):** headline contrast is vs the irrelevant-certificate arm; the "valid re-derivation does not improve (0.500→0.467, p=.46)" null is kept and now interpreted through EXPE's H5/H6: local evidence controls **flagging and absorption**, not proof repair — with the FREQ arm explaining (or refuting) the irrelevant-cert inj-dep rise as priming.

---

## 5. Ranking by evidential value per GPU-hour

| Rank | Experiment | GPU-h (HF) | Status | Why |
|---|---|---|---|---|
| 1 | Stage-0 zero-GPU: neghop multinomial, M5 retro-distance audit, strict metrics, EXPB bounds | ~0 | **REQUIRED-TMLR (gate)** | Removes the two artifacts any reviewer can find from released JSONs (N1, N9); freezes claims |
| 2 | **EXPC_FULL --target 127** (~1,651 gens, new dir, caption fix) | 1–2 | **REQUIRED-TMLR** | Cheapest powered polarity test in the program; converts the "noisy pilot" (n=21/cell → 381/cell) and fixes the lock-vs-caption contradiction (N2). Grammar imbalance (attribute-typed LFP, token mean 4.8 vs 7.2–8.5) persists at any n — disclosed, and EXPD supersedes it |
| 3 | EXPE evidence-mover | ~2 (+judge) | **REQUIRED-TMLR** | Highest evidence-per-hour: byte-identical stimulus, within-problem, kills the lexical axis (H5), replicates the flip on natural worlds |
| 4 | EXPD attribute quartet gradient (+anchors, parse gates) | ~12–15 | **REQUIRED-TMLR** | The negation-free dose-response; H1–H3 |
| 5 | EXPD categorical usability suite | ~4–5 | **REQUIRED-TMLR** | The confound-free **reuse** gradient (H4) — without it the flagship absorption claim is never demonstrated deconfounded (skeptic W1-attack-1) |
| 6 | 32B judge passes over all new suites + unparsed legacy rows | 15–30 | **REQUIRED-TMLR** | Judge-primary doubt DV (N10) |
| 7 | EXPD satellites: style, rule-type, on-path timing, T-arm | ~5 | REQUIRED-lite (appendix) | Closes A6/A3/A4 residuals; rule-truth audit also retro-fixes distractor daggers |
| 8 | **EXPF** powered doubt-forcing (matched unforced true arms, restart-vs-check coding) | 2–3 (+1 d eng) | unlocks-ICLR | Mediator/initiation claim; pilot overstates without true-arm cost (0.629 vs 0.856) and content coding (N4) |
| 9 | 32B EXPD replication (scale × distance) | 5–8 | unlocks-ICLR | Parametric scale claim on the clean gradient |
| 10 | OLMo/Llama 4-cell lineage check; `gsm8k_gradient.py` run | ~3–5 | optional/appendix | Corroboration; second regime is EXPG's (Plan B) lane |

**ICLR go/no-go** (aligned with storyPlan §5, 2026-09-01): flip iff H1+H2+H4 land clean, EXPF lands with matched controls, and EXPG (program traces) shows the gradient — else TMLR with the REQUIRED set, which already fully answers weakness #1.

---

## 6. Skeptic-attack preemption map (attack → design change that kills it)

| Skeptic attack | Design change |
|---|---|
| **W1-1**: deconfounded core (attributes) can't measure reuse; absorption gradient never confound-free | `cat_false_{1,3,∞}×{inert,usable}` promoted to core (2 positions, n=150), validator inj-dep + M4 echo/derivational split as the DV; H4 is confirmatory |
| **W1-2**: belief probe has the IV in its measurement | Probe **dropped from confirmatory**; reuse DV is the validator's inj-dep; probe (if any) exploratory with trace-free control |
| **W1-3**: on-chain evidence collapses to d≈0 mid-continuation | **Off-path spur is the core geometry** (never derived en route); on-path is a satellite with pre-registered doubt-onset-timing analysis (doubt token position vs step where the refuting antecedent becomes derivable) |
| **N1**: Fig-1 validity dose-response is an unparsed artifact (verified: parsed-only 0.730→0.788) | Stage-0 multinomial re-analysis is a hard gate; legacy validity curve retired (doubt kept); Panel C stacked multinomial; per-cell parse-rate manipulation check pre-registered; hardened v2 parser cross-check |
| **N2**: EXPC n=7 was `--target 7`, 127 available; paper text contradicts lock | EXPC_FULL at 127 (rank 2); caption corrected; `run_metadata.json` verified on cluster first |
| **N3**: ~20 uncoordinated "pre-registered primaries" | One PLAN2.md, one confirmatory family (H1–H6), Holm; all else labeled exploratory; external timestamp |
| **N4**: doubt-forcing pilot overstated ("no cost" false: wait-on-true 0.629 vs 0.856) | EXPF includes matched unforced true arms, quantifies the ~20pp cost, adds restart-vs-local-check content coding; framed as intervening on check-*initiation* |
| **N9**: EXPB conclusion imputation-sensitive at 0.317 unparsed | EXPE moves the manipulation into the **question**: trace byte-identical across arms, so arm contrasts cannot be driven by differential trace unparse; EXPB re-reported with bounds + judge pass before abstract wording |
| **N10**: lexical doubt detector weakest where the story lives (agreement .76–.80) | Judge-rated doubt is the primary DV in every confirmatory test; per-cell agreement tables; lexicon inflection fixes in v2 as secondary |
| **N11**: new-generator distribution shift | Benign + true-interruption anchors in every EXPD batch; TOST ±0.10 equivalence gate vs EXPA levels before confirmatory analysis; 200-world pilot go/no-go |
| **Mirrored-world cohort selection** (skeptic residual) | Confirmatory contrasts conditioned on jointly-solved quadruples; per-version solve rates reported |
| **EXPE jump-depth confound** (skeptic residual on EXPE-inf) | No certificate is inserted into the trace at all in EXPE — the refuting fact lives in the question, so there is no inserted-sentence derivation-front offset to match |
| **d=0 < d=1 non-monotonicity silently skipped** | d=0 included in EXPD's gradient (affirmative premise-contradiction cell); legacy trace-stated d=0 distinguished from question-stated d=0; discussion pre-registered |
| **G1 (no vLLM)/G2 (hardware)** | Both backends budgeted; wall-clocks quoted for HF batch-1 sharded and post-port; no repro-section compute claims until measured |
| **Lock hygiene (G4/N7)** | New dirs only (`results/EXPD_*`, `EXPE_*`, `EXPF_*`, `EXPC_POLARITY_CONTROL_FULL`); promoted legacy summaries (neghop, ctx, doubt, judge) get hashes in a new lock section; artifacts via `paper_artifacts.py` only |

**Alignment notes:** terminology and DV definitions inherit validatorPlan's defs box verbatim (closure-valid completion, injection-dependent, strict repair, silent absorption; refutation distance = Alg 5); main-text budget per storyPlan = one figure (above) + one design-matrix table + EXPC-127 update, everything else appendix — the sprawl answer is that EXPD/EXPE **replace** the legacy validity curve and the EXPC pilot in the main text rather than adding beside them.