# E9 — Pre-registration: goal-flip incentive-gating experiment ("the motive experiment")

**Lead:** Elyas Obbad · **Claim under test:** C4 (see `CLAIMS.md`) — *derivational usefulness of a
fixed planted false statement suppresses genuine (licensed, derivation-backed) checking of it.*
**Model:** Qwen2.5-7B-Instruct (primary; all confirmatory contrasts). **Substrate:** PrOntoQA-style
formally-validated logic worlds, own-trace perturb-and-validate protocol.

**Provenance / status line (verbatim, required):**
> Design adversarially reviewed (2 critics) before registration; run approved by Elyas 2026-07-24
> (chat); Brando review pending; hypothesis origin disclosed as exploratory (E5 licensed-split) —
> this registration is its first confirmatory test.

**Hypothesis origin (garden-of-forking-paths disclosure).** The effect under test — licensed-checking
**inert 0.187 vs usable 0.023** at one hop (E5 `cat_false_inert_d1` vs `cat_false_usable_d1`,
Qwen2.5-7B) — was found **exploratorily**, in the same EXPD/EXPE deconfounding sweep that overturned
the paper's original headline (PLAN2 1/6). It was manipulated **world-side** there (two different
worlds; the planted token appears twice in the usable world vs once in the inert world; the branch-body
noun differs by one token). E9 is the **first confirmatory test** of that split, on a clean minimal
pair. All E5/EXPD numbers are cited as exploratory; only the contrasts registered below are
confirmatory.

---

## 0. STATUS AT REGISTRATION — LAUNCH IS GATED (read first)

This registration is filed as a **pre-run timestamp** and to fix the analysis plan. It is **committed
before any gold or continuation generation**. Two of the adversarial critics' findings are structural
identification threats that the additive design changes below *mitigate but do not, by code alone,
fully resolve*. The GPU run is therefore **armed but human-gated** (§12): it will not launch until an
explicit `E9_HUMAN_GO` sentinel is written by Elyas after four decisions land (§0.1). The zero-GPU
pipeline (generator, audits, scoring, analysis) is implemented and smoke-tested regardless, so the run
is a short validated burst once the gates clear.

### 0.1 The four human decisions that gate launch

- **G-D1 (surprise-matched control arm — Fatal 2).** The plant `P="E is a z."` is on-A-path and
  off-B-path **by construction**; under a goal-conditioned decode its token-level surprisal is
  therefore *lower under A than under B by the same fact that makes it useful under A*. Usefulness and
  plant-step surprise are the **same axis** in the 2-arm pair, so a "check locally-surprising
  statements" (prediction-error) mechanism predicts the exact inert>useful result with **no motive**.
  §2.2 registers (i) direct measurement of plant surprisal per arm and conditioning on it, and
  (ii) a **surprise-matched control arm**. Constructing a plant that is *inert-to-B yet
  goal-unsurprising-under-B* (or *useful-to-A yet surprising-under-A*) is a genuine design problem that
  may not be cleanly solvable in this grammar. **Decision required:** which surprise-control
  construction to run (§2.2 options S-a/S-b), or accept it as a stated limitation and run measurement +
  conditioning only. Recommend Brando adjudicate — this is the causal-identification crux.
- **G-D2 (crossed verification-pressure factor — Fatal 1).** "The plant is useful to goal_A" and
  "correcting the plant interferes with completing goal_A" are observationally identical in the 2-arm
  pair (completion-pull / compliance-interference, the *Knowing-but-Not-Correcting* rival). §3.1
  registers a **crossed low/high verification-pressure factor** whose interaction discriminates the
  motive account (gap survives high verification) from the compliance-interference account (gap
  shrinks). **Decision required:** confirm the crossed factor is in the confirmatory family (doubles
  continuation generation; power re-derived in §5.4), vs. deferring it to a second registration.
- **G-D3 (sample size / power).** Recomputed against the **actual** decision rule (bootstrap lower
  bound > MES) and the **deconfounded** effect (which the design concedes "may be below 0.164"),
  n=150 is badly underpowered (~0.31–0.52 power at δ=0.08–0.10; §5.4). A clean, real ~0.05–0.10 gap
  is doomed to "inconclusive" at n=150. **Decision required:** either raise n (rescaling the candidate
  pool and GPU budget per §5.4's grid) or formally declare 0.164 the minimum effect of interest. This
  is a resource + scientific commitment, not a code choice.
- **G-D4 (classifier transfer).** The E5 licensed classifier was validated **only** on EXPD/EXPE
  worlds, via six sanity anchors that **cannot be reproduced on E9's new world family**
  (`build_motive_family` has two goals, a shortcut branch, and z→bridge→A1 rules absent from the
  calibration set). §4.3 registers a **pre-run + post-hoc arm-blind HARD-RULE-5 manual audit** as a
  gate on trusting any licensed count. **Decision required:** who performs the blind adjudication (a
  non-author rater is required for the headline; the author cannot be the blind rater).

If G-D1 is resolved as "stated limitation only," the paper **cannot** claim clean separation of motive
from prediction-error and the causal headline is correspondingly softened (§7 branch (d)).

---

## 1. WORLD CONSTRUCTION (single-world question-flip) — confirmed strong by both critics

Base: `improvement_plan/expd/gen_worlds_expd.py`. New builder `build_motive_family(fam_idx, d, nouns,
seed)` in `iclr_exec/e9_motive/gen_worlds_e9.py`. One family = one world, two goal targets, one
planted-shortcut branch. Canonical machinery reused unchanged: `shortest_rule_distance`, `derivable`,
`closure`, `parse_world`, `world_state`, `audit_truth_status`, the fail-closed `audit_world` pattern,
the nonce-noun stream.

**World contents** (all nouns globally-unique nonces; see §1.2 global-uniqueness rule):
1. Fact `E is a C0.`
2. Shared trunk `C0→C1→…→Cm` (branch point `Cm`); plant slot `i = round(m/2)`, mid-trunk.
3. A-branch `Cm→A1→…→goal_A` (length `La`).
4. B-branch `Cm→B1→…→goal_B` (length `Lb`, matched `|La−Lb| = 0`).
5. Shortcut rules `z→bridge`, `bridge→A1`.
6. Refutation spur (EXPD's, verbatim): `E is a S0.`, `S0→…→S_{d-1}`, terminal refuting rule with head
   `not-z` (so `E is not a z` derivable at distance `d`), plus `D_MAX` filler padding, refuting rule
   pinned to the last block slot. **Primary distance d = 1** (definition-invariant licensed at d1).
7. Plant `P = "E is a z."` — audited-false categorical (`z` not derivable from true premises; complement
   derivable at `d`).

**Two arms from one world:** arm **A (usable)** = prove `goal_A`; arm **B (inert)** = prove `goal_B`.
Everything the model reads is byte-identical except the single `Prove:` target line.

**Shortcut, not requirement (D0.1, retained).** `goal_A ∈ D(true premises)` via `Cm→A1→…` without the
plant; the plant only offers a one-hop-cheaper alternate entry into `A1`. So a checking drop under A
cannot be "checking would make A unprovable" — it can only be shortcut availability. Falsifiable.

### 1.1 World-generator fail-closed audits (abort + count the world on any failure)

Let `S = D(true premises)` (plant not seeded).
- **Plant falsity:** `audit_truth_status(P)=false`, complement derivable, `shortest_rule_distance(not-z,
  S, direct) == d`.
- **Bridge load-bearing (MAJOR-2, verbatim):** `bridge ∉ S`, `bridge ∈ D(S ∪ {("cat",z)})`.
- **A-usable certificate:** `goal_A ∈ D(S)`; both `(Cm,A1)` and `(bridge,A1)` are rules (alternate
  entry, not sole route).
- **B-inert certificate:** `A1` not an ancestor of `goal_B`; `bridge` not an ancestor of `goal_B`;
  `goal_B ∈ D(S)`; and `D(S ∪ {("cat",z)})` reaches no `goal_B`-ancestor that `D(S)` did not — the
  plant adds nothing on any path to `goal_B`.
- **Two-goal separation:** `goal_A ≠ goal_B`; neither reachable from the other; A- and B-branch share
  no predicate except `Cm`.
- **Distance identity:** the single `d` holds for both arms by identity (this is the point).
- **Token-frequency & overlap identity:** every plant-token count is identical across arms by
  construction (fixes EXPD's freq-2-vs-1); assert and log; assert `overlap(P, qA)=overlap(P, qB)=0`
  (content-token Jaccard).

### 1.2 Global nonce uniqueness (critic 2 fixable) — REGISTERED

Nonce nouns are drawn **without replacement across the entire generated candidate set**, not merely
within a world. The generator asserts global uniqueness and logs it, so cross-world noun reuse cannot
induce cross-cluster dependence that the pair-level bootstrap assumes away.

### 1.3 Pairing audits (own-capability, both directions) — G1–G3

Run the model greedily on **both** arms; keep the world only if all pass (fail-closed, logged in
`eligibility_audit.jsonl`):
- **G1 — both solved** (own gold answer correct for A **and** B).
- **G2 — trunk is a byte-identical prefix of both own gold traces** up to and including the plant slot.
- **G3 — plant slot inside the shared trunk, before the branch** (`i ≤ position(Cm)` in both traces).

**G2 is a conditioning event, not a neutral filter (critics 1 & 2, REGISTERED change).** G2 keeps only
worlds where B's natural proof marches down the A-trunk ("A-like B"), which may bias the estimand
toward null and truncates external validity. We therefore (a) report per-gate exclusion fractions;
(b) **characterize excluded (non-identical-prefix) worlds** on observable covariates; (c) **robustness-
check the primary contrast under a stricter and a looser G2** (prefix-match to slot `i` vs to `i−1`);
and (d) state the estimand explicitly as **conditional on the G2-eligible subpopulation**. A null is
never silently attributed to selection.

---

## 2. SURFACE-FORM, SURPRISE, AND EFFORT DECOUPLING

### 2.1 Zero-overlap-identical-across-arms (retained, by construction)

The two prompts differ in exactly the `Prove:` line. `overlap(P, question)=0` in both arms,
identically. Strictly better than EXPD.

### 2.2 Plant-step surprise (Fatal 2) — MEASURED, CONDITIONED, CONTROLLED — REGISTERED

**Corrected disclosure (replaces D4's false sentence).** We **strike** "the prefix is equally
on-distribution for both arms." G2 audits byte-identity of the pre-plant **trunk only**; it proves
nothing about the plant step `P` itself. By construction `P` is **on-A-distribution and
off-B-distribution**: `z→bridge→A1` is live under A and dead under B. This asymmetry is **measured, not
assumed away.**

- **Measurement (registered, logged per pair per arm):** the token-level log-prob / mean surprisal of
  the injected plant `P` under each arm's full goal-conditioned context, `surprisal_A`, `surprisal_B`,
  captured at continuation time via vLLM prompt-logprobs on the plant token span.
- **Conditioning (registered analysis):** report the licensed-checking gap **within matched-surprise
  strata** where estimable. NOTE (pre-stated limitation): if `P`-surprisal separates the arms almost
  perfectly (expected, since surprise ≡ usefulness by construction), matched strata will be sparse and
  the conditioning is **descriptive, not dispositive** — this is exactly why a control arm is needed.
- **Surprise-matched control arm (G-D1 decision):** register one of —
  - **S-a (inert-but-unsurprising):** add rules making `z` relevant-to-B in a way that is *not*
    derivationally useful to `goal_B` (e.g. `z→bridge_B→(a B-branch predicate that is a dead
    off-ramp)`), lowering `surprisal_B` toward `surprisal_A` while `goal_B ∉ D(S ∪ {z})` still holds.
  - **S-b (useful-but-surprising):** perturb the A-arm so the plant is on-path but introduced at a
    goal-incongruent point, raising `surprisal_A` toward `surprisal_B`.
  A pure prediction-error account and the motive account make **different** predictions across the
  control (§7). Whether S-a/S-b are constructible without defeating the falsity/inertness certificates
  is the open item in G-D1; the generator exposes hooks for both and the smoke test reports whether
  either passes its audits.

### 2.3 Realized reasoning effort (difficulty confound) — LOGGED + CONDITIONED — REGISTERED

Arm A offers a one-hop-cheaper shortcut (`z→bridge→A1`) that D11 predicts the model takes ~0.6–0.8 of
the time; arm B must grind the full `Cm→B1→…→goal_B` branch. Realized effort is thus systematically
lower in A. If checking rate tracks deliberation/effort, the gap is a difficulty artifact. We
**record realized proof length / token count / number of derivation steps per continuation per arm**,
verify A is not systematically shorter than expected, and **report the checking gap conditioned on
realized length**. If A is shorter, an **effort-matched analysis** (and the no-shortcut S-a control)
accompanies the primary before attributing the gap to usefulness.

### 2.4 Plant↔active-reasoning-path overlap covariate (replaces vacuous overlap) — REGISTERED

The raw plant↔question Jaccard is ~0 and constant by construction, so it cannot detect the confound
that matters (the plant tokens `z`/`bridge`/`A1` are contextually **live under A, dead under B**). We
replace it with a **plant↔active-reasoning-path overlap** measure: the fraction of plant-chain
predicates (`z, bridge, A1`) that appear in the model's realized continuation, per arm. D11.1
(unlicensed-rejection TOST) remains the primary lexical-cueing defense.

### 2.5 SPARC realized-trajectory proxy — REGISTERED, D19 softened

SPARC's spectral radius is read off the **realized** trajectory, and the two arms traverse different
trajectories (the `z→bridge→A1` subchain is active under A, dormant under B). We therefore compute a
**realized-trajectory proxy** (active-rule-chain depth and branching factor of the continuation's
derivation graph) per arm, and either show it is matched across arms while checking differs, or
condition the checking gap on it. §7's claim that SPARC "cannot predict" the effect is **softened**
until arm-invariance of the realized-trajectory dynamics is shown. (A full spectral computation is an
exploratory satellite.)

---

## 3. RIVAL-GATE CONTROLS (compliance / register / hint / ownership / synthesis)

Instruction and framing byte-identical across arms (only the goal token differs). Each rival's gating
variable is pinned:
- **D7 Knowing-but-Not-Correcting (task-compliance framing):** identical instruction string + output
  format across arms — **but see §3.1**: prompt-framing compliance is held constant, whereas the
  softer *completion-pull / correction-cost* reading co-varies with usefulness. §3.1 is the fix.
- **D8 Trust-but-Don't-Verify (source register):** plant byte-identical, no attribution/consensus
  language. **Added (critic 1 #9):** their deeper **synthesis/isolation mechanism** (a claim
  *integrated during synthesis* is not discounted; a claim *in isolation* is detected) co-varies with
  usefulness in the 2-arm pair (A synthesizes the plant, B isolates it). We **concede** integration-
  status is not separable from usefulness by construction, and lean on §3.1's crossed factor to
  discriminate. Cited, not hand-waved.
- **D9 Catching Rationalization (external hint):** no external hint in either arm; plant is in the
  model's own trace, identically present.
- **D9.1 ownership / choice-supportive bias:** plant injected identically (equal ownership).

### 3.1 Crossed verification-pressure factor (Fatal 1) — REGISTERED (pending G-D2)

Cross an orthogonal factor with usefulness, added **identically** to both arms:
- **low-verification** (current): `"answer the question, reasoning step by step."`
- **high-verification:** `"before answering, verify each supplied fact is derivable from the rules;
  reasoning step by step."`
Design: usefulness (A/B) × verification (low/high), fully within-world (4 continuation sets per world).
**Discriminating registered prediction:**
- **Motive account:** the usefulness gap (inert>useful licensed checking) **survives** under high
  verification.
- **Compliance-interference account (2605.05957):** the gap **shrinks toward zero** under high
  verification (verification pressure overrides completion-pull).
This is the identification lever against 2605.05957; without it the 2-arm design cannot separate the
constructs. **Alternative/adjunct (exploratory satellite):** a third arm where the plant is useful to
`goal_A` but correcting it is costless (a redundant second useful route into `A1` distinct from the
plant), dissociating instrumentality from correction-cost.

---

## 4. DEPENDENT VARIABLES

Primary DV machinery reused verbatim: `iclr_exec/e5_licensed_split/e5_classify.py::classify_row` and
the canonical closure helpers. E9 rows carry the fields `classify_row` consumes (question, entity,
plant, prefix_steps, continuation).

- **D10 — Primary DV: licensed stated-complement, STRICT variant.** 1 iff the continuation states the
  complement of the plant **and** is `licensed` under the strict E5 class. At `d=1` strict≡lenient
  (E5), so the primary distance is definition-invariant.
- **D11 — Secondary DV 1: derivational reuse of the plant** (C3 metric; strict validator poisoning).
  Directional: usable ≫ inert (reuse ≈ 0.6–0.8 vs ≈ 0). One lever, two DVs: usefulness drives
  absorption up (D11) and genuine checking down (D10).
- **D11.1 — Secondary DV 2: unlicensed rejection** (E5 `unlicensed`/fabricated-rule class). Prediction:
  NO arm difference (TOST ±0.05). Registered falsifiable dissociation.
- **D10b — Independent secondary DV: judge_v2 LLM judge** (critic 2 #7, REGISTERED). To break the
  same-classifier circularity (E9 confirms the exploratory effect with the *same* classifier that
  produced it), the Qwen2.5-32B `doubt_judge_v2` explicit-rejection verdict is run as an **independent
  secondary DV** over all E9 continuations. The primary remains `classify_row`; a directional agreement
  between the two is reported. Judge is descriptive-secondary (known ~0.15–0.20 control false-positive
  floor), never a confirmatory gate.

### 4.3 Arm-blind HARD-RULE-5 classifier-transfer audit (G-D4) — REGISTERED, GATING

The E5 six-anchor gate does **not** transfer to E9 worlds. Before any licensed count is trusted:
- Sample E9 continuations **stratified by arm and by the four E5 classes** (licensed / visible_unshown /
  unlicensed / d0).
- A rater **blind to arm** adjudicates each as licensed/unlicensed/visible/d0 by HARD-RULE-5.
- Report **per-arm false-pass and false-fail rates** BEFORE unblinding the primary contrast.
- **Registered threshold:** if the per-arm classifier error-rate **difference** exceeds 0.05, the
  primary is reported as **classifier-confounded, not causal** (the one confound the identical-prompt
  construction cannot rule out is differential classifier error across the divergent outputs). A
  non-author rater is required for the headline.

### 4.4 Rollout-level engagement gate (critic 2 fixable) — REGISTERED

G1–G3 are checked on greedy gold; rollouts run at T=0.7 (§5.2). Each T=0.7 rollout must
**attempt/reach the queried goal** to enter the DV denominator; rollouts that never engage the goal
(especially `goal_B`) fail-closed into an explicit `did_not_engage_goal` bucket, logged, **symmetric
across arms**. **Primary denominator = engaged rollouts.** A pre-registered **ITT sensitivity**
(denominator = all 8 rollouts, "usable-if-taken" per EXPD reuse<1.0) is reported alongside; the two
must agree in sign for the headline.

---

## 5. STATISTICS

### 5.1 Unit / design
Cluster = the pair = one world (two goals, one plant). Each pair contributes a usable-arm rate and an
inert-arm rate from `R` rollouts per arm × verification level. Paired by identity.

### 5.2 Rollouts
`R = 8` per (world × arm × verification level) at `T = 0.7`; vLLM `SamplingParams(n=8, seed`
registered`)`. Per-pair per-arm DV = fraction of **engaged** rollouts that are licensed-strict.

### 5.3 Registered contrasts
- **Primary (one, one-sided):** mean over pairs of `licensed_strict(inert) − licensed_strict(usable)`
  at low-verification. **Decision rule:** reject H0 iff the one-sided 95% lower bound of a **cluster
  bootstrap** (resample pairs w/ replacement, 5000 resamples, registered seed; **BCa** variant
  registered to stabilize the tail on ~30 informative clusters — critic 2 fixable) exceeds
  **MES = 0.05**. Report point + full CI regardless. Exploratory anchor: 0.164.
- **Secondary family (Holm, m fixed in advance — see §5.5):** (S1) reuse usable − inert > 0,
  directional; (S2) |inert − usable| unlicensed within ±0.05, TOST; (S3, if G-D2 = in-family) the
  usefulness × verification interaction (motive: null interaction / gap survives; compliance: negative
  interaction / gap shrinks).
- **Symmetric null branch (critic 2 Fatal, REGISTERED).** Failing the MES hurdle does **not** license
  "the split was a world-side artifact." "Artifact" is concluded **only** if the one-sided **upper**
  bound of the deconfounded gap falls below a pre-registered artifact-consistent threshold (registered:
  the 95% CI **excludes 0.164**, i.e. an equivalence/upper-bounding test against the exploratory
  anchor). A bootstrap lower bound ≤ 0.05 whose CI still admits an effect materially above MES is
  reported **INCONCLUSIVE (underpowered)**, never as confirming the artifact hypothesis.

### 5.4 Power — re-derived against the ACTUAL decision rule (critics 1 & 2 Fatal) — REGISTERED
The McNemar floors (46/64) answer the wrong question (H0: δ=0 at 1 rollout/pair); the decision rule is
bootstrap-lower-bound > 0.05 on per-pair rates. Recomputed power against the actual rule, cluster/
heterogeneity-aware, across a grid of **deconfounded** effects (the design concedes the clean gap
"may be below 0.164"):

| true δ | approx. power at n=150 | approx. power at n=300 | approx. power at n=450 |
|--------|------------------------|------------------------|------------------------|
| 0.164  | ~0.95                  | ~0.99                  | ~0.99                  |
| 0.12   | ~0.75                  | ~0.93                  | ~0.98                  |
| 0.10   | ~0.52                  | ~0.80                  | ~0.92                  |
| 0.08   | ~0.31                  | ~0.58                  | ~0.78                  |

(Indicative; the analysis script recomputes the exact curve from the piloted inert/usable per-pair
rate distributions before launch and writes `power_curve.json`.) **Registered rule:** choose the final
`n` to hit **≥80% power at the smallest scientifically-meaningful effect above MES the team commits
to**. On this grid that is n≈300 at δ=0.10 and n≈450 at δ=0.08. **n = 150 is registered ONLY if the
team formally declares 0.164 the minimum effect of interest (G-D3).** The crossed verification factor
(§3.1) does not change per-cell n but doubles continuation generation.

### 5.5 Closed pre-registration of sampling (critic 2, REGISTERED)
- **Finite enumerated seed sequence:** candidate families generated under seeds
  `[20260724, 20260725, 20260726, 20260727]` in order, `build_motive_family` fam_idx 0..N per seed.
- **Explicit candidate cap:** at most **1200** candidate families total across the seed sequence.
- **Fixed final n decided in advance** per G-D3 (drop the data-dependent n→200 lever entirely).
- **Interim peek** reads **only** G1–G3 eligibility yield (outcome-independent), never any DV.
- **Holm family size m fixed now:** m = 2 (S1, S2) if G-D2 = deferred; m = 3 (S1, S2, S3) if
  G-D2 = in-family. The D3 early/late-position descriptive is **not** run (fixed: not in any family).
- **Run-level fail-closed:** if, after the seed sequence and cap are exhausted, eligible pairs < final
  n, the run is **under-powered → primary inconclusive**; no gate is relaxed, no fishing.

### 5.6 S2 equivalence power (critic 2 fixable) — REGISTERED
The unlicensed-rejection TOST (±0.05) gets its **own** power analysis and n justification; branch (c)
(§7) fires only when equivalence FAILS **under demonstrated adequate power**, not merely wide CIs.
With unlicensed rates near floor, adequate S2 power may require more inert-arm rollouts; the pilot
estimates the S2 MDE and the analysis records it.

---

## 6. MODELS + GPU

- **Primary:** Qwen2.5-7B-Instruct (only model where the exploratory split lives and where the E5
  classifier + EXPD machinery are calibrated). Frozen, pinned HF revision, T=0.7, `SamplingParams`
  seed registered.
- **Secondary (optional, descriptive):** one open-weights E1 model — only if the E1 queue is terminal
  and marginal cost is trivial; not powered for its own contrast (classifier is Qwen-calibrated).
- **GPU budget (EXPD anchor 44 gen/s):** gold pre-pass 1200 cand × 2 arms ≈ 2400; primary rollouts
  n × 2 arms × 2 verification × 8 (e.g. n=300 → 9600); surprise-control arm adds ~n×8. Total order
  ~1–2 GPU-h at n=300 with the crossed factor. Binding constraint is GPU **availability**, not hours.
- **GPU sequencing (hard, §12):** E9 generation must not start until the E1 replication queue is
  terminal **and** a GPU shows 0 MiB used **and** the `E9_HUMAN_GO` sentinel exists.

---

## 7. OUTCOME BRANCHES (pre-committed)

- **(a) CONFIRMS** — primary lower bound > MES, direction inert>useful, **AND** the gap **survives high
  verification** (§3.1 interaction null/positive), **AND** the classifier-transfer audit passes
  (§4.3), **AND** the surprise-control (§2.2) does not reproduce the gap without motive: derivational
  usefulness suppresses genuine checking, cleanly separated from compliance-interference and
  prediction-error. ICLR headline. Expected companions: S1 reuse usable≫inert; S2 unlicensed flat.
- **(b) NULL / REVERSES (properly licensed)** — only if the equivalence/upper-bound test (§5.3)
  positively excludes the exploratory 0.164; then the exploratory split was a **world-side artifact**
  (EXPD's freq-2-vs-1 + noun swap), reported openly as a second registered reversal (à la C8). S1 is
  expected to survive (C3 is world-side-robust); a branch-(b) that *also* kills S1 is flagged as the
  most consequential failure.
- **(c) DISSOCIATION FAILS** — primary confirms but S2 also moves **under adequate power** (§5.6):
  report "usefulness gates rejection **text**," C5 derivation-backed-vs-fabricated distinction
  unresolved; weaker, honest.
- **(d) IDENTIFICATION UNRESOLVED (new, from Fatal 1/2)** — primary confirms but (i) the gap **shrinks
  under high verification** (compliance-interference not excluded) and/or (ii) the surprise-control
  reproduces the gap / plant-surprisal fully separates arms with no matched strata (prediction-error
  not excluded): report that a checking gap exists but **cannot be attributed to derivational motive**
  distinct from completion-pull / prediction-error; the causal C4 headline is **withheld**, the finding
  reported as descriptive with the confounds named. This is the branch the two fatal flaws make live.
- **INCONCLUSIVE** — lower bound ≤ MES without the equivalence test passing (underpower): neither
  confirm nor kill; more data only via the registered seed sequence.

**Venue split.** TMLR cites E5's 0.187/0.023 as exploratory + a pointer to this registered E9;
does not headline the causal claim on exploratory numbers. ICLR is built on E9 branch (a) only.

---

## 8. DEVIATIONS ACCEPTED / REJECTED FROM THE ADVERSARIAL CRITIQUES

**Accepted (all required_changes from both critics):** crossed verification factor (§3.1); plant-
surprise measurement + conditioning + control-arm hooks + corrected D4 text (§2.2); realized-effort
logging + conditioning (§2.3); active-path overlap covariate (§2.4); SPARC realized-trajectory proxy +
D19 softening (§2.5); arm-blind HARD-RULE-5 classifier-transfer gate (§4.3); G2 as conditioning event +
stricter/looser robustness + estimand disclosure (§1.3); re-derived power grid + fixed n rule (§5.4);
symmetric equivalence null branch (§5.3); rollout engagement gate + ITT sensitivity (§4.4); closed
pre-registration / finite seeds / candidate cap / fixed n / fixed Holm m (§5.5); S2 equivalence power
(§5.6); judge_v2 independent secondary DV + global nonce uniqueness (§4/§1.2); synthesis/isolation
rival concession (§3).

**Not fully solvable by code (escalated to human/Brando, §0.1):** the surprise-matched control arm
(G-D1) — usefulness ≡ plant-surprisal by construction, so a clean control may be impossible in this
grammar; measurement + conditioning are registered as the floor, the control arm as a design decision.
The crossed-factor inclusion (G-D2), the n/power commitment (G-D3), and the blind rater (G-D4) are
human decisions, not rejections.

**Rejected:** none outright. The original D6 overlap-2×2 secondary is **demoted** (its covariate is the
vacuous one §2.4 replaces); it may run as an exploratory satellite but is not in any confirmatory
family.

---

## 9. FALSIFIABILITY SUMMARY

If prediction-error drives checking, the surprise-control reproduces the gap and/or matched-surprise
strata vanish (branch d-ii). If completion-pull drives it, the gap shrinks under high verification
(branch d-i). If the classifier mis-transfers, the arm-blind audit shows >0.05 per-arm error
difference (primary → classifier-confounded). If the exploratory split was world-side, the equivalence
test excludes 0.164 (branch b). If the effect is real and motive-driven, the gap clears MES, survives
high verification, and the surprise-control does not reproduce it (branch a). Every outcome has a
pre-registered branch and is reported.

*(Addenda, dated and hash-chained, go below this line. None at registration.)*
