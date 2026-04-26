# Hopfield equivalence: what's literal, what's metaphor

The README's reframe rests on a single technical claim: softmax
attention IS a continuous modern Hopfield network with a specific
energy. This document states the equivalence carefully and marks where
it stops being literal. It also seeds a catalog of candidate energies —
the actual research design space.

Reference: Ramsauer et al. 2020, *Hopfield Networks Is All You Need*
(arXiv:2008.02217), with predecessors Krotov & Hopfield 2016 (dense
associative memory) and Demircigil et al. 2017 (exponential-storage
Hopfield).

## 1. The literal equivalence

For stored patterns `X ∈ ℝ^(N×d)` (rows = patterns) and a query state
`ξ ∈ ℝ^d`, Ramsauer et al. define the energy

    E(ξ) = -lse(β · X ξ) + ½ ξᵀ ξ + (β⁻¹ log N + ½ M²)        (E1)

where `lse(z) = log Σ exp(z_i)` and `M = max_i ‖x_i‖`. The constant term
makes `E ≥ 0` and is irrelevant for dynamics.

The update rule from one gradient-flow / proximal step on `E` is

    ξ_new = X · softmax(β · X ξ)                                  (U1)

This is **exactly** scaled-dot-product attention with `Q = ξ`, `K = X`,
`V = X`, and `1/√d_k` absorbed into β.

The properties that follow from `E1`:

- **Storage capacity** scales **exponentially** in pattern dimension `d`
  (Demircigil et al. 2017; Ramsauer §3). This is the formal sense in
  which softmax attention is a strictly more powerful associative
  memory than the classical Hopfield network's polynomial capacity.
- **Convergence in one step** (with high probability, for well-separated
  patterns). Ramsauer §3, Theorem 3.
- **The fixed points of the dynamics correspond to stored patterns**,
  meta-stable mixtures, or the global-average state (the "rank-1
  attractor" — see §4 below).

This is the hard part of the equivalence and it is **literal**:
attention's forward pass is the gradient step of `E1`.

## 2. What the equivalence does not literally cover

The equivalence above is for **single-step retrieval with `K = V`**. The
transformer architecture does several things that take the equivalence
beyond literal:

| Architectural feature | Status of the equivalence |
| --- | --- |
| Distinct `K` and `V` (i.e., learned value projection ≠ stored pattern matrix) | Equivalence still holds with a redefined "stored patterns" matrix; Ramsauer §A.2 |
| Multi-head | Each head is its own Hopfield update with its own energy `E_h`; total step is parallel composition, **no joint energy stated** |
| Multi-layer (stacked attention) | Each layer is a Hopfield step with **its own learned energy**; the composition is not the gradient of any single energy in general — this is where "softmax attention IS Hopfield" becomes "softmax attention is a sequence of Hopfield retrievals" |
| Layer norm, residual stream | Modify the proximal-step interpretation; not modeled in `E1` |
| Causal mask | Per-position energy `E_t(ξ_t)` over keys 1..t; OK but `t`-dependent |
| MLPs between attention layers | Outside the energy framework entirely; pure feedforward modulation of the residual stream |
| Cross-attention | Query and stored-pattern domains differ; Hopfield framework still applies head-by-head |

**Bottom line:** the *forward pass of one attention head, in isolation*,
is literally a Hopfield update on a specific energy. The *full
transformer* is a stack of Hopfield updates with intervening
non-Hopfield operations (LN, MLP, residual). It is sound to call
softmax attention "an EBM update"; it is metaphor to call a full
transformer "minimizing a single energy."

This matters for the project because it bounds what claims we can make:

- **In scope (literal):** "we propose a different per-head energy whose
  gradient gives a different attention update."
- **In scope (metaphor, but useful):** "we propose multi-step inference
  by repeated application of the same energy step (= depth-recurrence
  with shared weights)."
- **Out of scope (over-claim):** "the whole transformer is energy
  minimization on `E_global`." The literature does not support this,
  and the project should not claim it.

## 3. Why `lse(β X ξ)` (and not something else) gives properties B1–B5

The graveyard's §B identifies five load-bearing properties of softmax.
Each maps to a feature of `E1`:

- **Global competition** ← `lse` is a *single* normalizer over all
  patterns; replacing `lse` by per-pattern functions removes this.
- **Injectivity of query → weights** ← `softmax(β X ξ)` is injective in
  `ξ` whenever `X` has full row rank; bounded feature maps `φ(ξ)` lose
  this (Bridging the Divide).
- **Spikiness / low entropy** ← `lse` saturates near its argmax for
  large β; bounded feature maps don't.
- **Streaming-stable kernel** ← `lse` admits the running-max
  recurrence (online softmax); other normalizers may or may not.
- **Exponential storage capacity** ← from `lse`'s exponential terms;
  polynomial activations give polynomial capacity (Demircigil 2017).

So the project's design space is constrained: any candidate energy
that wants properties 1, 3, 5 likely needs an `lse`-like term (or a
specifically engineered substitute). Properties 2 and 4 are downstream
of the choice of feature map / kernel.

## 4. Candidate energies — the actual research surface

For each candidate, we list the form, the gradient/update rule, what's
preserved vs lost relative to Hopfield's `E1`, whether `Z` is avoidable,
and the open question.

### 4.1 Hopfield (= softmax attention)

`E = -lse(β X ξ) + ½ ξᵀ ξ.` All five properties B1–B5. Baseline rung. `Z`
appears only inside `lse`; computing the update *requires* computing
the partition function over patterns each step.

### 4.2 Quadratic-only energy

`E = -ξᵀ M ξ + ½ ξᵀ ξ.` Update is linear: `ξ_new = M ξ`. *Loses* B1, B3,
B5 (polynomial capacity, no spikiness, no winner-take-all). `Z` is
trivially absent. **Open question:** is this strictly weaker than linear
attention? (Probably yes — and linear attention is in the graveyard.)

### 4.3 Sigmoid-coupled energy

`E = -Σᵢ log(1 + exp(β · xᵢᵀ ξ)) + ½ ξᵀ ξ.` Update is
`ξ_new = Σᵢ σ(β xᵢᵀ ξ) xᵢ`, i.e., *unnormalized* sigmoid attention. *Loses*
B1 (no global normalizer); *preserves* B3 (sigmoid saturates) and is a
reasonable candidate for B4 (streaming-stable). Recent literature on
sigmoid attention is the empirical version of this. **Open question:**
how much does dropping the normalizer cost on retrieval?

### 4.4 Bethe / pairwise free energy

`F[Q] = E_Q[E(ξ)] - H(Q)` with `Q` factored over cliques larger than
singletons. Goes up Claim 1's variational ladder. **Preserves** the
energy interpretation; *loses* the simple closed-form update — must
solve a fixed-point iteration per token. **Open question:** does the
fixed-point iteration cost outpace its accuracy gain?

### 4.5 Score-matched implicit energy

The energy is never written; only `∇ξ E` is learned (Hyvärinen score
matching). *Loses the explicit energy*, gains training without `Z`.
**Open question:** is this just "EBM with trained gradient"? Does it
still admit a streaming kernel? Counts as "EBM" for the project's
purposes only if a meaningful gradient-flow is recoverable at inference.

### 4.6 Energy descent at inference (depth-recurrence on `E1`)

Same `E1`, but apply the update `K` times instead of once at inference.
This is the Claim 3 lever — adaptive compute by iterating on the same
energy. *Preserves* B1–B5 by definition. **Open question:** is `K > 1`
ever better than the single-step retrieval Ramsauer §3 says is already
optimal for well-separated patterns? Probably yes for *not*-well-separated
patterns (= hard inputs), which is the whole point.

## 5. Falsification of the project's framing

If careful re-derivation shows the Hopfield equivalence is **metaphor**
(not a literal energy whose gradient is softmax attention) for the
single-head single-step case we actually use — i.e., if `E1`'s
update derivation has a hidden assumption we can't satisfy in standard
transformers — then the README's "different energy than Hopfield's"
framing must soften to "EBM-inspired heuristics for attention." This
would not kill the project; it would just rename it. We owe ourselves
that check before publishing.

The current best read (Ramsauer §A.1–A.2; community confirmation in
follow-ups) is that the equivalence is literal for single attention
heads with `K = V` or with redefined stored patterns. **The project's
framing is sound.** What is *not* sound — and we should not claim — is
that a full multi-layer transformer minimizes any single energy.

## References

- Ramsauer, H. et al. *Hopfield Networks Is All You Need.* arXiv:2008.02217 (2020).
- Krotov, D. & Hopfield, J.J. *Dense Associative Memory for Pattern
  Recognition.* NeurIPS 2016.
- Demircigil, M. et al. *On a Model of Associative Memory with Huge
  Storage Capacity.* J. Stat. Phys. (2017).
- Hyvärinen, A. *Estimation of Non-Normalized Statistical Models by
  Score Matching.* JMLR 2005.
