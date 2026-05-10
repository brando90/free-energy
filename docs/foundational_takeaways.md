# Free-Energy Project: Foundational Takeaways

*A working reference for the project at github.com/brando90/free-energy — synthesizing the conversation on softmax, partition functions, scaling laws, Hamiltonians, information, and Kolmogorov complexity.*

---

## Table of Contents

1. [The Central Thesis, Sharpened](#1-the-central-thesis-sharpened)
2. [Why KV Cache Matters Conceptually](#2-why-kv-cache-matters-conceptually)
3. [The Boltzmann Distribution: Three Derivations](#3-the-boltzmann-distribution-three-derivations)
4. [Assumptions Inherited from Softmax/Boltzmann — and What to Reject](#4-assumptions-inherited-from-softmaxboltzmann--and-what-to-reject)
5. [Hamiltonians: What They Are and Why They Matter](#5-hamiltonians-what-they-are-and-why-they-matter)
6. [Scaling Laws: Where the Power Law Comes From](#6-scaling-laws-where-the-power-law-comes-from)
7. [The Graveyard of Shannon Alternatives](#7-the-graveyard-of-shannon-alternatives)
8. [Kolmogorov Complexity as the "Right" Information](#8-kolmogorov-complexity-as-the-right-information)
9. [Measuring Diversity, Concretely](#9-measuring-diversity-concretely)
10. [Synthesis: Concrete Research Moves](#10-synthesis-concrete-research-moves)
11. [On Muse Spark (Briefly)](#11-on-muse-spark-briefly)
12. [Open Questions and Research Targets](#12-open-questions-and-research-targets)

---

## 1. The Central Thesis, Sharpened

The original framing: *the partition function and softmax — which appear three times in a transformer (attention, output head, MLE's implicit Z) — are design choices, not fundamental truths.*

After the conversation, the sharpest version of the thesis:

> **Softmax attention is a Boltzmann distribution over keys at temperature $1/\sqrt{d}$ with a bilinear Hamiltonian $H(q,k) = -q \cdot k$.** The KV cache is the resulting non-parametric memory that this distribution requires. The "softmax/Z tax" is the cost of paying for this distribution at every forward pass.

The escape routes from softmax are not "find a better softmax" but rejecting one of the assumptions that *force* softmax. Crucially:

- The exponential form is **not arbitrary** — it is the Legendre dual of Shannon entropy. Change the entropy functional, change the exponential.
- The *most ambitious* version of the project is: **use Hamiltonian *dynamics* (symplectic, reversible flow) instead of one-shot Boltzmann *lookup*.**

---

## 2. Why KV Cache Matters Conceptually

KV cache is more than an inference optimization — it's a **diagnostic** for what softmax attention is doing.

### Mechanics
During autoregressive generation, attention computes:
$$\text{attn}_i = \sum_{j \leq i} \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right) v_j$$

Keys $k_j$ and values $v_j$ for past tokens don't change between generation steps. Cache them. This turns per-step cost from $O(n^2 d)$ to $O(n d)$.

Note the asymmetry: queries are *not* cached. **Q is "current-token-side"; K, V are "context-side."** This asymmetry is structural to softmax attention.

### Cost
For $L$ layers, $h$ heads, head dim $d_h$, sequence $n$, batch $b$, fp16:
$$\text{cache size} = 2 \cdot L \cdot h \cdot d_h \cdot n \cdot b \cdot 2 \text{ bytes}$$

For a 70B-class model at 32k context: ~80 GB cache, **bigger than the weights**. Long-context inference is memory-bound, not compute-bound.

### Why this matters for the project

- KV cache exists because softmax attention is **non-parametric memory**. Every past token gets its own slot. There's no compression mechanism.
- The cache is read with a softmax over all positions: $Z = \sum_j \exp(q \cdot k_j / \sqrt{d})$ is recomputed every step. **The cache *is* the support over which $Z$ is summed.**
- Architecture design space, viewed as a triangle:
  - **Vertex (i):** Exact non-parametric recall — full softmax attention, $O(n)$ cache.
  - **Vertex (ii):** Lossy compressed state — RNNs, SSMs (Mamba), $O(1)$ cache.
  - **Vertex (iii):** Compressed but expandable — DeepSeek's MLA, low-rank caches, ~$O(\sqrt{n})$.

An energy-based / Hamiltonian-flow architecture sits **off this triangle**: energy is symmetric in its arguments (no Q/K/V split), inference is iterative minimization, and "KV cache" doesn't mean what it does in transformers.

---

## 3. The Boltzmann Distribution: Three Derivations

Boltzmann's distribution $p_i = e^{-\beta E_i}/Z$ is the *unique* answer to a specific optimization principle. It can be derived three ways, all landing on the same form.

### 3.1 Physics derivation (combinatorial)
$N$ copies of a system with microstates indexed by $i$, occupation numbers $n_i$. Multinomial counting: $W = N!/\prod_i n_i!$. With Stirling, $\log W / N \to -\sum p_i \log p_i$. Maximize subject to normalization and energy constraint $\sum p_i E_i = \langle E \rangle$ via Lagrange multipliers $\alpha, \beta$:

$$-\log p_i - 1 - \alpha - \beta E_i = 0 \implies p_i = \frac{e^{-\beta E_i}}{Z}$$

### 3.2 MaxEnt derivation (Jaynes 1957)
Strip off the physics. Among distributions satisfying $\mathbb{E}[f_k(X)] = \mu_k$, pick the one maximizing Shannon entropy. Result: the **exponential family**

$$p_i = \frac{1}{Z(\lambda)} \exp\left(-\sum_k \lambda_k f_k(x_i)\right)$$

Boltzmann = single energy constraint. Gaussian = mean + variance constraints. Categorical/softmax = indicator constraints.

**Key reframing:** putting softmax over logits $z_i$ = implicitly claiming "the only thing I know about the predictive distribution is the expected logit value, and I want maximum entropy consistent with that." This is a **strong implicit assumption** almost never justified by the actual problem.

### 3.3 Information-theory derivation (Sanov)
Sanov's theorem: probability that empirical distribution $\hat{p}_N \approx p$ when sampling iid from $q$ decays as $e^{-N \cdot D_{KL}(p\|q)}$.

MaxEnt = minimizing KL to uniform.

**Conditional limit theorem (Csiszár):** condition iid samples on $\frac{1}{N}\sum f(X_i) \approx \mu$, and the conditional distribution of any $X_i$ converges to the exponential-family distribution with parameter chosen to match $\mu$. **The Boltzmann distribution is the typical distribution under the constraint** — the inevitable asymptotic limit.

### 3.4 The structural punchline
The exponential form is **forced** by the form of $-p \log p$:
$$\frac{d}{dp}[-p \log p] = -\log p - 1$$
Setting this linear in $f_k$ gives $\log p$ linear in $f_k$, hence $p$ exponential in $f_k$.

**The exponential is the Legendre dual of Shannon entropy.** Change the entropy, change the exponential.

---

## 4. Assumptions Inherited from Softmax/Boltzmann — and What to Reject

Every softmax in a transformer carries these assumptions. Ranked by yield-for-this-project:

### (a) Shannon entropy is the right uncertainty measure
- **Reject it →** Tsallis/Rényi entropies → q-exponentials → sparsemax / α-entmax.
- **What improves:** sparsity, interpretability, polynomial tails (no exponential suppression of distant logits).
- **Verdict:** Low-yield. Graveyard is full, wins are marginal.

### (b) Linear/bilinear energy in Q and K
- **Reject it →** modern Hopfield networks (Krotov-Hopfield, Ramsauer) with polynomial or log-sum-exp energies.
- **What improves:** higher storage capacity (exponential vs linear in dimension), possibly fewer heads needed.
- **Verdict:** Medium-yield. Ramsauer is the anchor; well-studied but with room.

### (c) Tractable / computed partition function $Z$ ⭐ HIGH-YIELD
- **Reject it →** LeCun's energy-based models. Work with unnormalized energies, score rather than normalize.
- **What improves:** no $O(n)$ pass over context per forward step, no exponentiation, no softmax saturation, fundamentally different inference dynamics (iterative energy minimization).
- **What's lost:** MLE, calibrated probabilities, easy density estimation, easy sampling.
- **Verdict:** This is where the project's central claim lives. High-risk, high-reward.

### (d) Equilibrium / iid assumption ⭐ HIGH-YIELD
- **Reject it →** non-equilibrium statistical mechanics: Jarzynski equality, Crooks fluctuation theorem, large deviations for dependent sequences.
- **What improves:** principled treatment of non-iid sequence data, non-equilibrium training dynamics.
- **Verdict:** **Genuinely under-explored publicly.** Connects to the CLT-generalizations intuition. There is a real theoretical gap at the intersection of scaling laws and non-equilibrium stat mech.

### (e) Uniform reference measure
- **Reject it →** structured priors. MaxEnt becomes min-KL to a non-uniform $\pi$, giving $p_i \propto \pi_i e^{-\beta E_i}$.
- **What improves:** encoding inductive biases, sample efficiency.
- **Verdict:** Medium-yield. This is what DeepSeek-V4's Sinkhorn-Knopp / Birkhoff-polytope work is implicitly doing — choosing a doubly-stochastic geometric prior on routing.

### Composability
**The two highest-yield rejections, (c) and (d), compose.** An energy-based architecture without $Z$, with non-equilibrium dynamics, is a coherent research program — not just another softmax replacement. (b) is a natural starting point because Ramsauer already connects modern Hopfield to attention.

---

## 5. Hamiltonians: What They Are and Why They Matter

### 5.1 Three meanings of "Hamiltonian"

**Classical mechanics:** $H(q, p)$ on phase space generates time evolution via Hamilton's equations:
$$\dot{q} = \partial H / \partial p, \quad \dot{p} = -\partial H / \partial q$$
Three properties matter:
- **Energy conservation:** $dH/dt = 0$ along trajectories.
- **Symplectic structure:** the form $\omega = dp \wedge dq$ is preserved (Liouville's theorem). *Volumes in phase space don't change.* Contrast with gradient descent, which contracts volumes.
- **Reversibility.**

**Statistical mechanics:** same $H$, used as the energy in the Boltzmann distribution: $p \propto e^{-\beta H}$.

**Critical reframing:** softmax attention is a Boltzmann distribution with $H(q,k) = -q \cdot k$. **The $q \cdot k$ in attention is literally a Hamiltonian.**

**Quantum:** operator with eigenvalues = allowed energies. Skip; distraction for this project.

### 5.2 Hamiltonians in ML

- **Hopfield networks (classical):** $H(s) = -\frac{1}{2}\sum_{ij} W_{ij} s_i s_j$. Inference = energy minimization. Capacity ~$0.14N$ patterns.
- **Modern Hopfield (Krotov-Hopfield, Ramsauer):** higher-order/log-sum-exp Hamiltonians give exponential capacity. Ramsauer's energy *recovers softmax attention as one step of gradient descent on $H$*. This is the bridge to transformers.
- **Hamiltonian Monte Carlo (HMC):** augments target $p \propto e^{-U}$ with momentum, defines $H = U + \frac{1}{2}p^T M^{-1} p$, simulates Hamiltonian flow as proposal mechanism. Works because flow is volume-preserving and reversible — properties gradient descent lacks. Standard sampler for EBMs.
- **Hamiltonian Neural Networks (Greydanus et al. 2019):** networks that learn $H$, then enforce Hamilton's equations as architecture. Conserve energy by construction.
- **Lagrangian/Hamiltonian gradient descent (Su-Boyd-Candès, Wibisono):** Nesterov momentum has a Hamiltonian interpretation. Relevant to signal-propagation pillar.

### 5.3 Four research programs in the Hamiltonian space

| # | Program | Rejects | What it Buys |
|---|---------|---------|--------------|
| 1 | Different $H$, same Boltzmann framework | (b) | Modest empirical wins (modern Hopfield direction) |
| 2 | Drop Boltzmann, keep $H$ | (c) | LeCun's EBM territory; central thesis |
| 3 | **Use Hamiltonian *dynamics*, not just functions** | (c) + structure | Symplectic/reversible flow, conservation laws via Noether, century of mechanics machinery |
| 4 | **Non-equilibrium Hamiltonian/Lagrangian for *training itself*** | (d) | Principled non-iid theory; connects to free energy as natural quantity in non-eq thermo |

### 5.4 Resolving the "free-energy" name

Free energy $F = -\frac{1}{\beta}\log Z = \langle H \rangle - TS$ is the bridge object connecting Hamiltonian, partition function, and entropy.

**The thesis "Z is bad" is in tension with the project name "free-energy"** because computing $F$ requires $Z$. Two resolutions:
1. **LeCun's move:** work with $H$ directly, not $F$.
2. **Jarzynski move:** work with *free energy differences* (don't require $Z$ in absolute, only ratios).

**The most ambitious project version** is Programs 3+4: forward pass = Hamiltonian flow on representations (no softmax, no $Z$), trained by a non-equilibrium fluctuation-theorem-style objective handling iid-violation of sequence data.

---

## 6. Scaling Laws: Where the Power Law Comes From

### 6.1 Honest framing

Scaling laws are **reproducible empirical power-law fits with toy-model derivations and a track record of getting rewritten** — not physics. They lack:
- First-principles derivation (toy models *assume* the spectrum/manifold; the exponent is input not output).
- Predictive power outside fitted regimes (Kaplan was wrong, emergence claims partly metric artifacts per Schaeffer 2023, test-time compute opened a new axis).
- Universality classes derived from symmetry/dimension (Tay 2022 is suggestive but not derived).
- Conserved quantities, RG calculations landing on observed numbers.

More like Moore's Law or allometric scaling than thermodynamics. **Useful as the empirical bar to clear, not as a theoretical foundation.**

### 6.2 Where the power law comes from (theoretical accounts)

All accounts say the same thing in different language: **power-law scaling comes from power-law structure in the data.** Either spectral, manifold, or skill-frequency. Architecture and optimizer mostly don't matter for the *exponent*.

| Account | Source of exponent |
|---------|-------------------|
| Bahri et al. (2021) | Variance-limited ($1/D$ CLT rate) vs. resolution-limited ($D^{-s/d}$, smoothness over manifold dim) |
| Sharma-Kaplan (2020) | Intrinsic data manifold dimension: $D^{-4/d}$ for smooth targets |
| Maloney-Roberts-Sully (2022) | Spectral decay of data covariance: $\lambda_k \sim k^{-\alpha}$ |
| Hutter (2021) | Zipfian feature frequencies in data |
| Michaud et al. (2023) | "Quantization" — discrete skills with Zipf-distributed difficulty |
| Roberts-Yaida | $1/\text{width}$ perturbation theory; infinite-width limit (more about behavior than scaling exponents) |

For natural images $d \sim 10$; for language $d \sim 5$–$30$. These match observed exponents reasonably.

### 6.3 What this means for architecture changes

- Changing architecture should shift the *constant*, not the *exponent*, **unless** the architecture interacts with the data manifold differently than softmax does.
- The empirical bar is **shifting the exponent under a Chinchilla-style protocol** (5–7 model sizes, IsoFLOP curves) — not just winning at fixed scale. Tay 2022 is the prior to overturn.
- Most softmax replacements (Performer, Linformer, Linear Attention) died on scaling, not on fixed-scale benchmarks. The graveyard's "cause of death" column is mostly "scaling exponent."

---

## 7. The Graveyard of Shannon Alternatives

Shannon $H(X) = -\sum p \log p$ is built on three commitments, each rejectable:
1. Information is a property of *distributions over symbols*, not objects.
2. Entropy of independent systems is *additive*.
3. Coding-theoretic motivation (compression to channel capacity).

### 7.1 Rejecting additivity (axiom)
- **Tsallis $S_q$:** non-additive, controlled by $q$. Maxent → q-exponentials → sparsemax/α-entmax. Argued for systems with long-range correlations.
- **Rényi $H_\alpha$:** additive but parametric. $\alpha=1$ Shannon, $\alpha=2$ collision, $\alpha=\infty$ min-entropy. Used in differential privacy, generalization bounds. **More useful as a tool than a replacement.**

### 7.2 Rejecting coding motivation
- **Fisher information $I(\theta) = \mathbb{E}[(\partial_\theta \log p)^2]$:** measures information about a *parameter*. Local (distinguishing nearby distributions) rather than global. Cramér-Rao bound. **The natural metric on statistical manifolds (Amari's information geometry).**
  - **Deeply relevant to this project:** shows up in natural gradient, muP/Tensor Programs (Fisher-Rao metric implicitly sets learning rates), loss-landscape geometry. Worth its own deep dive.

### 7.3 Rejecting the "distributions over symbols" view → algorithmic information
- **Kolmogorov complexity $K(x)$:** length of shortest program producing $x$. *Information as a property of the object itself, no distribution needed.*
- **Logical depth (Bennett 1988):** running time of shortest program. Distinguishes "shallow random" from "deep structured" — both can have high $K$, only structured has high depth.
- **Sophistication (Koppel):** size of "model" part in two-part code (model + noise). Separates structure from randomness within an object.
- **Effective complexity (Gell-Mann, Lloyd):** length of description of regularities.

### 7.4 Resource-bounded / computable approximations
- **Time-bounded $K^t(x)$:** computable for fixed $t$. Crypto, pseudorandomness.
- **Solomonoff induction:** $P(x) \propto 2^{-K(x)}$. **Universal prior, provably optimal Bayesian predictor — but uncomputable.** AIXI is the agent version.
- **Speed Prior (Schmidhuber 2002):** computable variant penalizing long-running programs. Mostly ignored by mainstream ML.
- **MDL (Rissanen):** practical computable approximation using a chosen model class. **What people actually use** for Kolmogorov-flavored model selection.

### 7.5 Temporal / structural
- **Predictive information / excess entropy (Bialek, Crutchfield):** mutual info between past and future. Captures temporal structure that per-symbol entropy misses.
- **Statistical complexity ($\epsilon$-machines, Crutchfield):** info needed to predict future given past. Distinguished from Shannon entropy of the process.

### 7.6 Pattern in the graveyard

Most alternatives are either:
1. Functional generalizations not changing the philosophy (Tsallis, Rényi).
2. Shifts in what information is *about* (Fisher: parameters; Kolmogorov: objects; Predictive info: temporal).
3. Computable approximations to algorithmic ideal (MDL, $K^t$).

**Kolmogorov is the candidate "right" measure.** The reason it didn't take over ML: uncomputability. **This is solvable (sort of — see next section).**

---

## 8. Kolmogorov Complexity as the "Right" Information

### 8.1 What $K$ buys you

$K(x) = \min\{|p| : U(p) = x\}$ for fixed universal Turing machine $U$.

- **Universal** (up to $O(1)$ across $U$).
- **Distribution-free.** Property of $x$ alone.
- **Subsumes Shannon:** $\mathbb{E}_p[K(x)] = H(p) + O(1)$ for computable $p$. **Shannon = expected $K$.** Kolmogorov is strictly more informative — gives complexity of *individual* objects, not just averages.
- **Captures structure:** small for compressible, large for incompressible.

### 8.2 Five routes around uncomputability

| Route | What | Notes |
|-------|------|-------|
| A | Compression upper bounds | $K(x) \leq |\text{compress}(x)|$. Normalized Compression Distance (Cilibrasi-Vitányi). Crude but real. |
| B | Resource-bounded $K^t$ | Computable for fixed $t$. Renaissance via hardness-randomness in complexity theory. |
| C | MDL with chosen model class | Computable, optimizable. **What people actually do.** |
| D | **Neural compression as $K$ proxy** ⭐ | Modern LLMs are SOTA compressors (Deletang et al. DeepMind 2023). Cross-entropy = code length = upper bound on $K$. |
| E | Sampling from $K$-flavored posteriors | Program induction (DreamCoder), neural priors. |

### 8.3 The deep observation

> **A trained neural network is a computable approximation to Kolmogorov complexity. Cross-entropy loss is exactly the negative log-likelihood, which is the code length under the model's distribution.**
>
> $$\text{LM compression of } x = -\log_2 p_\text{LM}(x) \approx K(x) + O(1)$$
>
> if the LM is good enough.

### 8.4 Implications for the project

- **Scaling laws reframed:** $L(C) = aC^{-\alpha} + L_\infty$ has $L_\infty$ = irreducible Kolmogorov complexity rate of data. Power-law term = model's distance from this floor as function of compute. **Scaling laws literally track convergence to Kolmogorov-optimal compression.**

- **What is information / diversity:** if $K$ is the right notion, dataset information = $K(\text{dataset})$. Diversity = $K(\text{dataset})/|\text{dataset}|$ (compression rate). Meaningful diversity = sophistication or logical depth (structured, not random noise).

- **Softmax/$Z$ reframed:** cross-entropy training under softmax *is* MDL training. Transformers are already doing Kolmogorov-flavored learning. The project's question is whether **a different parameterization (energy-based, Hamiltonian-flow) approximates $K$ faster (better scaling exponent)** on natural data.

- **Architecture-as-prior view:** the right architecture is one whose induced family of computable distributions $\{p_\theta\}$ contains *short programs for natural data*. Universal TMs are too unconstrained; the "right" architecture is a restricted UTM whose natural programs match natural data. Connects to program induction (DreamCoder, Speed Prior, AIXI) and to Lean 4 as a substrate.

### 8.5 Concrete novel research direction

**Characterize the implicit prior of a transformer (or your softmax-free architecture) in Kolmogorov terms.** What's the equivalent of the Speed Prior for a transformer? What programs are "easy" to express? Connects to circuit complexity in mech interp and to why transformers learn certain in-context algorithms but not others.

---

## 9. Measuring Diversity, Concretely

Ranked from least to most Kolmogorov-flavored:

1. **Lexical diversity** (vocab size, type-token ratio): bad, surface form only.
2. **Embedding diversity** (mean pairwise distance): better but inherits embedding-model biases.
3. **Compression-based** ($|\text{gzip}(D)|/|D|$): crude $K$ proxy. Surprisingly works (Jiang et al. 2023: gzip + kNN beats neural classifiers on some text classification).
4. **Cross-entropy under held-out LM** ⭐ : **probably the best practical measure.** Tight upper bound on $K(D)/|D|$ if LM is strong.
5. **Conditional cross-entropy under your training run**: how much does adding $x_i$ reduce loss on others. Influence functions / data attribution. Directly measures Kolmogorov-novelty of $x_i$ relative to rest.
6. **Sophistication / logical depth proxies**: research target, hard to compute.

**For "scale diversity" research:** measure 4 or 5. Train a strong reference model, evaluate cross-entropy on candidates, rank by reference cross-entropy as proxy for $K$-novelty. Computable, principled, aligned with what we want to track.

---

## 10. Synthesis: Concrete Research Moves

Three composable moves:

### Move 1: Frame as Kolmogorov approximation
Take "approximating $K$ with the right architectural prior" as the project's deepest framing. Softmax/transformer is one prior; energy-based/Hamiltonian-flow is another. The empirical question: **which approximates $K$ faster on natural data, measured as scaling exponent on cross-entropy loss.**

### Move 2: Use cross-entropy under reference LMs as the operational measure
For information, diversity, and data quality. Computable. Principled in the Kolmogorov sense. Bypasses uncomputability.

### Move 3: Pillar 2 connection to scaling-law theory
Track **intrinsic dimension of representations** across layers. This is the Sharma-Kaplan exponent in operation. Architecture changes that improve scaling should be visible as changes in how the manifold dimension evolves through depth.

### The most ambitious version
**Programs 3+4 from Section 5:** forward pass = Hamiltonian flow on representations (no softmax, no $Z$, symplectic, reversible), trained by non-equilibrium fluctuation-theorem-style objective handling iid-violation of sequence data. Whether tractable in 12 weeks is separate — but it's the version that's *new*, not another graveyard entry.

### Roadmap connections

| Pillar | Connection from this conversation |
|--------|----------------------------------|
| 1 — Softmax-replacement graveyard | Add "scaling laws" column. Most replacements died on exponents, not benchmarks. Tag each with which assumption (a)–(e) it rejects. |
| 2 — Signal propagation theory | Track intrinsic representation dimension across depth. Connect to Sharma-Kaplan. Hamiltonian dynamics → conservation laws → built-in stability. |
| 3 — ML systems (FlashAttention/Triton) | Understand KV cache's $O(n)$ growth as the cost of softmax's non-parametric memory. MLA/SSMs are the empirical contrast. |
| 4 — EBM/JEPA baselines | Plant flag in Program 2 (LeCun) territory. Consider Program 3 (Hamiltonian flow) as differentiator. Use Jarzynski-style free-energy *differences* to dodge $Z$. |

---

## 11. On Muse Spark (Briefly)

Closed-weights, transformer-based catch-up release. AAII v4.0 score 52 (vs 57 GPT-5.4, 57 Gemini 3.1 Pro Preview, 53 Claude Opus 4.6). **Architecture undisclosed** — gives no signal on whether anyone at scale is moving off softmax/transformer.

The one architecturally interesting public detail: **RL-induced thought compression** cuts inference tokens by ~half (58M vs ~120–157M for similar capability on the AAII run). Phase-transition phenomenon under length penalty during RL.

**Relevance to this project:**
- Empirical hook for the "softmax/$Z$ tax" framing — 2–3× inference cost variance among frontier transformers shows architecture choices leave a lot on the table.
- Thought compression has free-energy flavor: model pushed toward complexity-penalized solution manifold. Good citation in Pillar 2, not a paper.
- Closed weights → can't ablate what we can't see → **strengthens the case for the Lean 4 testbed.** Formal methods give a domain to claim architectural wins rigorously without leaderboard-chasing.
- *No evidence Meta has attacked the partition-function problem.* Project direction unaffected.

---

## 12. Open Questions and Research Targets

Things that came up in conversation that are genuinely open and could become contributions:

1. **A unified non-equilibrium / non-iid CLT for scaling laws.** Heavy-tailed CLTs, dependent-sample CLTs, matrix/operator CLTs. Yang's Tensor Programs is closest framework. No clean public synthesis exists.

2. **Characterizing transformer's implicit Kolmogorov prior.** What programs are "easy" to express? Speed-Prior analog for attention? Connects to mech interp circuit complexity.

3. **Hamiltonian flow as inference layer.** What does symplectic, reversible, volume-preserving inference look like for sequence modeling? What conservation laws emerge? Is there a Noether's theorem for representational symmetries?

4. **Free-energy *differences* (Jarzynski-style) as training objective.** Avoids absolute $Z$. Has anyone tried this for large-scale generative models? (Answer: little; mostly small-scale physics-ML.)

5. **Lean 4 as substrate for program-induction architectures.** A formal-methods-native model class with strong inductive biases for compositional structure. Could give measurable Kolmogorov-flavored architectural advantages.

6. **Empirical measurement of intrinsic dimension across layers** for softmax vs softmax-free architectures. Is the manifold-dimension trajectory different? Does it predict scaling exponents?

7. **Fisher-Rao geometry of softmax-free architectures.** muP/Tensor Programs assumes Fisher-Rao for setting learning rates. What's the right metric for energy-based models? Does it give different scaling rules?

8. **The non-equilibrium / iid-violation gap** at the intersection of scaling laws and stat mech is genuinely under-explored publicly and is where the project's "free-energy" name actually points (free energy is the natural quantity in non-equilibrium thermo).

---

## Glossary of Key Equations

| Concept | Equation |
|---------|----------|
| Boltzmann | $p_i = e^{-\beta E_i}/Z$ |
| Partition function | $Z = \sum_i e^{-\beta E_i}$ |
| Free energy | $F = -\frac{1}{\beta}\log Z = \langle H \rangle - TS$ |
| Shannon entropy | $H(p) = -\sum p_i \log p_i$ |
| KL divergence | $D_{KL}(p\|q) = \sum p_i \log(p_i/q_i)$ |
| Sanov | $\mathbb{P}(\hat{p}_N \approx p) \asymp e^{-N \cdot D_{KL}(p\|q)}$ |
| Fisher information | $I(\theta) = \mathbb{E}[(\partial_\theta \log p)^2]$ |
| Kolmogorov | $K(x) = \min\{|p| : U(p) = x\}$ |
| Solomonoff prior | $P(x) \propto 2^{-K(x)}$ |
| Shannon-Kolmogorov bridge | $\mathbb{E}_p[K(x)] = H(p) + O(1)$ |
| LM compression | $-\log_2 p_\text{LM}(x) \approx K(x) + O(1)$ |
| Sharma-Kaplan scaling | $L \sim D^{-4/d}$, $d$ = intrinsic manifold dim |
| Scaling law form | $L(C) = aC^{-\alpha} + L_\infty$ |
| Softmax attention as Boltzmann | $\text{attn}_i = \sum_j \frac{e^{-\beta H(q_i,k_j)}}{Z(q_i)} v_j$ with $\beta=1/\sqrt{d}$, $H(q,k)=-q\cdot k$ |

---

## TL;DR (the one-paragraph version)

Softmax attention is a Boltzmann distribution with bilinear Hamiltonian $-q\cdot k$ at temperature $1/\sqrt{d}$, and the KV cache is the non-parametric memory it requires; the Boltzmann form is forced by Shannon entropy via Legendre duality, so escaping softmax means rejecting one of five inherited assumptions (Shannon-as-uncertainty, bilinear energy, tractable $Z$, equilibrium/iid, uniform prior), with the highest-yield rejections being tractable $Z$ (LeCun/EBM territory) and equilibrium/iid (genuinely under-explored, connects to non-equilibrium thermo where "free energy" is the natural quantity); scaling-law exponents are properties of the data manifold not the architecture, so any architectural win must shift the exponent under Chinchilla-style protocols, and the deepest reframing is that **trained networks are computable approximations to Kolmogorov complexity** (cross-entropy = code length = upper bound on $K$), making "approximating $K$ faster" the right empirical bar and enabling cross-entropy under reference LMs as a principled, computable diversity measure — and the most ambitious project version is forward pass as Hamiltonian flow (symplectic, reversible) trained by Jarzynski-style free-energy *differences*, which is a coherent program rather than another graveyard entry.
