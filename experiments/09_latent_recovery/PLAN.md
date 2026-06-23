# 09_latent_recovery — Local latent recovery in AR transformers (pilot)

**Lead:** Elyas Obbad · **Status:** pilot, pre-registered before first run (2026-06-10)
**One-line claim under test:** pretrained causal transformers sometimes exhibit *local hidden-state
contraction* back toward a gold rollout after a controlled semantic perturbation, with no external
feedback — and this contraction is regime-dependent (thick semantic manifolds vs. brittle formal ones).

## Relation to the free-energy program

This is `probe_06_error_compounding` ("the key test", FINDINGS.md) grown into its own experiment.
The $(1-\varepsilon)^T$ compounding bound assumes errors are **independent, constant-rate, and
unrecoverable**. The review paper attacks the premises via verifiers (external recovery); this
experiment attacks the third premise *internally*: is a single semantic error actually unrecoverable
in the model's latent trajectory? Outputs feed Proposal 2's error-model fits
(geometric vs. recoverable-Markov) directly.

## Definitions

- **Gold rollout:** greedy decode on an instance the model solves correctly; hidden states
  $h^{gold}_{\ell,t}$ cached for all layers $\ell$, steps $t$.
- **Perturbed rollout:** at reasoning step $t_0$, replace the model's correct intermediate
  conclusion with a *wrong-but-well-formed* one; continue greedy decoding with **no external
  feedback, no critique prompting, no extra token budget**; cache $h^{pert}_{\ell,t}$.
- **Null ensemble (calibration):** $n$ temperature-resampled continuations from the same prefix
  *without* perturbation, restricted to ones that remain correct. Defines the natural spread of
  valid trajectories.
- **Recovery coefficient** $\rho_k$: divergence(perturbed, gold) at horizon $k$ after $t_0$,
  normalized by the null ensemble's spread at the same horizon. $\rho_k < 1$: contraction toward
  the gold manifold beyond what "any valid continuation" explains. $\rho_k > 1$: compounding.

## Measurement suite (ordered by trustworthiness)

1. **Behavioral recovery:** final-answer correctness of perturbed rollouts; doubt/backtracking
   language rate; length inflation.
2. **Probe-decoded task-state agreement:** linear probes trained on gold rollouts to decode the
   running task state (current entailed fact in PrOntoQA); agreement(perturbed, gold) over horizon.
   *Primary latent metric* — immune to the trajectory-alignment problem below.
3. **Raw geometry (exploratory only):** cosine / normalized $\ell_2$ between $h^{pert}$ and
   $h^{gold}$ per layer per horizon. Reported but not load-bearing.

## Known confounds, controlled by design

- **Trajectory alignment.** Perturbed and gold rollouts emit different tokens, so positionwise
  state comparison conflates "computation diverged" with "text differs." Primary metric is
  therefore probe-decoded *task state*, not raw geometry; raw geometry is always reported against
  the null ensemble, never absolutely.
- **Precommitment (Cox et al.).** If the final answer is probe-decodable *before* $t_0$,
  "recovery" may be answer inertia, not re-entry. Control: report answer-decodability at $t_0$
  per instance; the headline statistic conditions on *not-yet-decodable* instances; "answer
  preserved" and "trajectory re-entered" reported separately.
- **Extra serial compute (Li et al. objection).** Fixed decoding budget after perturbation;
  no multi-turn revision. If the model spends extra tokens to recover, length inflation is
  reported as its own outcome, not hidden inside recovery.

## Pilot scope (this week)

- Model: Qwen2.5-7B-Instruct (frozen). Task: PrOntoQA (synthetic FOL world, parseable proofs —
  perturbations are programmatic, not hand-written).
- ~500 instances → gold set = solved subset. One perturbation family: wrong-but-well-formed
  intermediate conclusion. Injection at early / mid / late $t_0$. Null ensemble n=8 per instance.
- Deliverable: divergence-vs-horizon figure (perturbed vs. null, split by injection point) +
  behavioral recovery rate + the $t_0$-decodability control. Existence proof for $\rho_k < 1$,
  nothing more.

## Pre-registered predictions

- P1: nonzero behavioral recovery on PrOntoQA (model sometimes lands the correct answer despite
  an uncorrected wrong intermediate step), declining with later $t_0$.
- P2: probe-decoded task-state agreement recovers toward null-ensemble levels within a finite
  horizon on a nontrivial fraction of instances ($\rho_k < 1$ exists).
- P3: among instances where the answer was already decodable at $t_0$, apparent recovery is
  higher — i.e., the precommitment confound is real and measurable (this is a *control
  succeeding*, not the result).
- Failure mode that kills the pilot: probe accuracy on gold rollouts too low to define
  task-state agreement (< ~80%). Then the metric, not the hypothesis, failed — redesign probes
  before concluding anything.

## After the pilot (not now)

Regime axis: EntailmentBank (thick NL) ←→ FOLIO ←→ ProofNet ←→ VeriBench/Lean (brittle corner —
the program's testbed). Perturbation families 2–3 (irrelevant distractor, direct contradiction).
Mechanism localization via path patching only once the phenomenon exists. Lyapunov language only
with an operational definition, or not at all.

## Revision experiments (pre-registered 2026-06-11, before first run except where noted)

- **Geometry with lexical control** (cached states only): per-layer null-calibrated cosine
  divergence at matched boundaries; layer-0 = token-identity control; projection variant
  removes layer-0 direction. Primary prediction: none designated — diagnostic. Decides
  whether "latent" appears in the title.
- **Goal-anchor ablation** — PRIMARY: validated re-derivation survives goal masking
  (QA-format condition without stated target; attention knockout on goal tokens).
  Falsified if validated recovery collapses under either manipulation → anchor mechanism.
- **Brittle regime (chained arithmetic)** — PRIMARY: validated recovery collapses
  (no redundant derivation paths); poisoning dominant. *Note: job launched ~30 min before
  this entry was written; hypothesis was stated in the working conversation beforehand
  and in the task description; recorded here for completeness.*
- **R1-distill arm** — PRIMARY: doubt verbalization rises (von Recum-like), validated
  recovery flat. Injection site: visible proof body (matches existing protocol);
  think-block injection deferred unless compute allows a second cell. Think block
  stripped before validation, analyzed separately for private-channel doubt. Expect
  elevated unparsed-run rate; report, don't retry.
- **Belief-bias (false ontology)** — PRIMARY: validated recovery drops relative to
  fictional cohort at matched hop distribution (4-6). Cohort shrinkage under belief
  conflict reported as a finding, per-condition cohort sizes up front.
- **Null ablation** (n=8→32, T∈{0.6,0.8,1.0} on 40-instance subset): diagnostic for
  rho stability; no primary.
- **Multiple comparisons**: wrong/early was the original pre-registered primary contrast;
  all other behavioral contrasts labeled exploratory; Holm correction reported for the
  family-vs-benign tests.

## Batch-2 revision experiments (pre-registered 2026-06-11, before first run)

- **Inferential-distance dose-response (neghop k)** — PRIMARY: verbalized doubt (judge
  + lexical) decreases monotonically in k for k in {1 (=negstep), 2, 3}; validated
  recovery decreases with k. Poisoning unmeasurable for negations (grammar); curve is
  on flagging + recovery. Category-falsehood reported alongside as the global-distance
  reference point, flagged as a different construction.
- **Falsehood size sweep (1.5B/32B, negstep + category-falsehood)** — PRIMARY: silent
  absorption (category-falsehood poisoning, ~0 doubt) persists at 32B; secondary:
  negstep recovery scales like true-interruption recovery did. Behavioral only.
- **Matched-decoding calibration** — perturbed continuations sampled at T=0.8 (n=4),
  rho_func recomputed with decoding-matched numerator. PRIMARY: benign ≈ falsehood
  persists under matched decoding (style propagation, not decoding mismatch, drives
  the null result). If falsehood >> benign emerges, the latent metric is rehabilitated.
- **Second family (OLMo-2-7B-Instruct, ungated)** — headline cells (benign/negstep/
  falsehood). PRIMARY: falsifiability gradient direction replicates. Behavioral only.
- **GSM8K perturbation pilot** — first realistic-task regime cell: corrupt one
  intermediate numeric result in the model's own CoT, no feedback. PRIMARY: behavior
  falls between PrOntoQA (redundant) and chained arithmetic (brittle); poisoning
  substantial. n>=100 gold.
