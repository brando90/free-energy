# Claims under test — strength ratings and what each motivates

Every claim is tagged with its **type** (theorem / empirical / conjectured), the
**layer** it must be tested at, and the **alternative architecture** it motivates
if true. The filter: a claim "has good reasons" only if it is a *theorem* (follows
from the math of the factorization, normalization, or objective) or a *robust
empirical finding* (reproduced, mechanism understood).

The object under scrutiny is the standard AR LLM:

```
p_θ(x) = ∏_t p_θ(x_t | x_<t),   each conditional a softmax over vocab,   trained by MLE (teacher forcing)
```

Three design commitments are bundled here and most "LLM cons" indict exactly one:
the **AR factorization**, the **softmax / local normalization**, and the **MLE
objective**. Separating them tells you which alternative each argument actually
motivates.

---

## The claims we test (each has real backing)

### 1. Softmax bottleneck — the output head is provably misspecified
- **Type:** theorem. **Layer:** architecture + data.
- **Statement:** a softmax over a `d`-dim hidden state can only produce a
  log-probability matrix of rank ≤ `d+1`. If the true context→next-token
  log-prob matrix has rank > `d+1`, the model class literally cannot represent it,
  regardless of data or training.
- **Why it's strong:** it's a rank inequality, not a conjecture. Mixture-of-softmaxes
  and higher-rank heads measurably help, which is the predicted consequence.
- **Indicts:** the softmax parameterization. **Motivates:** non-normalized / energy
  heads, mixture-of-softmaxes.

### 2. Mode-covering is intrinsic to MLE
- **Type:** theorem. **Layer:** objective.
- **Statement:** MLE minimizes forward KL `D_KL(p_data ‖ p_θ)`, which is
  zero-avoiding: it blows up if `p_θ` puts ~0 mass on any real datum, so the
  optimizer is compelled to spread mass to cover everything, including noise and
  contradictions.
- **Why it's strong:** it's the geometry of the objective, not a tuning failure.
  Directly drives hedging outputs and nonzero mass on falsehoods.
- **Indicts:** the MLE objective. **Motivates:** EBMs (unnormalized compatibility,
  Z paid only when needed), margin / contrastive losses, RLVR.

### 3. Rank collapse with depth
- **Type:** theorem (pure attention). **Layer:** architecture.
- **Statement:** pure self-attention (no MLP/residual) converges to rank-1
  doubly-exponentially in depth — token representations collapse to one vector.
  The proof is the row-stochastic attention map acting as a Perron–Frobenius
  contraction.
- **Caveat to test:** residuals and MLPs slow it; the integrated question is whether
  it survives training in a full model.
- **Indicts:** stacked softmax attention. **Motivates:** non-normalized attention,
  alternative mixing (SSMs).

### 4. The partition function is a removable per-step tax
- **Type:** established + ablation. **Layer:** architecture.
- **Statement:** softmax is a local partition function paid at every layer and
  position — `Z_attn = Σ_k exp(qᵀk/√d)` over the sequence axis, `Z_out = Σ_v
  exp(w_vᵀh)` over the vocab axis. Ramsauer 2020: softmax attention *is* a
  continuous Hopfield network whose exponential energy was chosen to make Z
  tractable. Most downstream uses need only energy *differences*, where Z cancels.
- **Decisive test:** ablation. Swap softmax → sigmoid / linear, match everything
  else. If task loss matches within CI, the necessity arguments are wrong.
- **Indicts:** local normalization. **Motivates:** EBMs (pay Z only when you need a
  probability), sigmoid / linear attention, SSMs.

### 5. Fixed compute per token is a representational ceiling
- **Type:** strong (complexity-theory backed). **Layer:** architecture + complexity.
- **Statement:** a standard transformer does O(1) sequential adaptive computation
  per token. Circuit-complexity results (constant-depth → TC⁰-type classes;
  Merrill & Sabharwal) imply a single forward pass cannot solve problems needing
  more sequential depth than the architecture has. Chain-of-thought externalizes
  serial computation into the token stream — itself evidence of the limit.
- **Indicts:** the AR + fixed-compute generation process. **Motivates:**
  inference-as-optimization (energy minimization, variable compute), search.

### 6. Error compounding — LeCun's (1−e)ⁿ argument — UNDER TEST
- **Type:** conjectured, **weak as usually stated**. **Layer:** trained behavior + verifier.
- **Statement:** if each token independently leaves the valid manifold with prob `e`
  and is unrecoverable, `P(coherent length-n) = (1−e)ⁿ → 0`.
- **Why it's weak:** errors are not i.i.d. in a trained model and "unrecoverable" is
  doing all the work. Long-form coherence empirically does not decay geometrically.
  **It collapses under a verifier** (e.g. Lean), which breaks unrecoverability.
- **This is the claim the integrated test is most likely to falsify in our domain.**
  See probe 06 and the recoverable-Markov alternative model.
- **Indicts:** AR generation under the unrecoverability assumption. **Motivates:**
  verifier-in-the-loop decoding, iterative refinement — *if* it holds, which we doubt.

### 7. Reversal curse
- **Type:** empirical, reproduced. **Layer:** trained behavior.
- **Statement:** trained on "A is B", the model does not reliably infer "B is A"
  (Berglund 2023). A controlled demonstration that AR learns directional
  co-occurrence, not symmetric relations.
- **Indicts:** the train-on-static-text paradigm. **Motivates:** relational / world
  models, bidirectional or any-order objectives.

### 8. Brittleness / Lipschitz–margin
- **Type:** empirical + theory link. **Layer:** trained behavior.
- **Statement:** small input perturbations swing outputs. Theory: the true
  input-space margin obeys `‖δ‖ ≥ f_θ(x) / L_global` with
  `L_global ≤ ∏_i ‖W_i‖₂`, so high output confidence does **not** imply a large
  input margin — confidence can be inflated by scaling weights while the geometric
  margin stays tiny. Brittleness is the predicted symptom.
- **Indicts:** unconstrained Lipschitz training. **Motivates:** spectral norm /
  weight-decay constraints, margin objectives.

### Data wall (context, not a plotted probe)
- **Type:** empirical resource curve. **Layer:** external.
- **Statement:** high-quality text is finite and being consumed; combined with
  power-law scaling (exponential data for linear loss gains), the cheap-gains
  regime is ending. Strong because it depends only on the curve LLMs already obey,
  not on any "can't do X" claim. Tested by fitting a scaling curve on VeriBench
  subsets and extrapolating.

---

## Explicitly excluded as weak (knowing why is part of the argument)

| Claim | Verdict | Why excluded |
|---|---|---|
| LeCun (1−e)ⁿ as a *proof* | weak | trivial math on false premises (i.i.d., unrecoverable); dies under a verifier. Kept only as probe 06 to *test*, not assert. |
| Exposure bias | overstated | mechanism real, empirical magnitude on strong models small; partly fixed by scale + RL. Not load-bearing. |
| "Stochastic parrot" / no understanding | not operationalizable | makes no falsifiable prediction as stated; its operationalizable pieces (reversal, planning) are tested on their own. |
| Hallucination is *inevitable* | partly conjectural | the structural *pressure* follows from #2, but "inevitable" exceeds the evidence; verification/abstention reduce it. |

---

## What survives, and what it motivates

The arguments with real backing cluster into three independent indictments — which
is exactly why they point at different fixes:

- **Against softmax parameterization:** #1, #3, #4 → energy/non-normalized heads,
  sigmoid/linear attention, SSMs.
- **Against the MLE objective:** #2 → EBMs, margin/contrastive losses, RLVR.
- **Against AR + fixed-compute generation:** #5, #8 → inference-as-optimization
  (energy minimization), search-augmented decoding.

The single most defensible claim is **#1 (softmax bottleneck)** — an exact
representational impossibility. The most defensible *empirical* one is the
**data wall** — it depends only on the known scaling curve. None of these say
AR+scale *stops working*; they establish provable ceilings and rising costs, which
is the honest case for investing in alternatives (EBMs), not a proof that LLMs fail.
