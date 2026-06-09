# Meeting Notes — Free-Energy / EBM Project

**TLDR:** Decided the project's step 1 (convert a pretrained open-weight LLM into an EBM, Lean as eval testbed), picked score matching as the first training method to study, agreed on experiments B1/B2/E1–E3 over VeriBench, and drafted 6 paste-ready GitHub issues; whiteboard covered AR-vs-EBM cost accounting, score matching → Fisher divergence, and the (skepticism-flagged) Hessian-trace claim.

**Date:** Tuesday, June 09, 2026 (71 min)
**Attendees:** Brando Miranda, Elyas Obbad
**Recording:** private Fathom link (in Brando's Fathom account, meeting of 2026-06-09)
**Paper under study:** Song & Kingma, *How to Train Your Energy-Based Models*, [arXiv:2101.03288](https://arxiv.org/abs/2101.03288)
**Repo:** https://github.com/brando90/free-energy

---

## 0. Decisions

1. **North star (step 1):** convert a *pretrained open-weight LLM* into an EBM, with Lean as the eval testbed. Rationale: pretrained open weights embody billions of dollars of data digestion; training an EBM from scratch forfeits that and we don't have the compute anyway. The "fair fight" (AR vs. EBM from scratch, same data/compute) is **step 2**, attempted only after a conversion recipe exists.
2. **Training method to study first:** score matching (also the closest relative of what IRED does).
3. **Three starting threads:** EBT (Gladstone et al. 2025), IRED (Du, Mao, Tenenbaum 2024), and grafting (Keshe's grafting idea) — target some mixture of the three that beats the baselines.
4. **Two blog posts** (Brando drafting — both now published: [why EBMs vs. AR](https://cs.stanford.edu/people/brando9/2026/06/09/why-energy-based-models-the-toy-ar-vs-ebm-argument.html), [score matching](https://cs.stanford.edu/people/brando9/2026/06/09/score-matching-training-ebms-without-z.html)): (i) why EBMs vs. AR, the toy cost–error accounting; (ii) score matching → Fisher divergence → update rule → optimizer research questions, ending on a teaser for post 3 (the Hessian-trace claim).
5. **Process:** TODOs live in the Google Doc / slides, not Discord.

---

## 1. Whiteboard derivation (what was taught)

### 1.1 AR vs. EBM cost accounting

- AR next-token conditional: $p(x^{<t>} = v \mid x^{<1:t-1>}) = e^{f_\theta(v)} / Z_\theta$, with $Z_\theta = \sum_{v' \in X} e^{f_\theta(v')}$ a sum over the vocabulary $X$, $|X| = V$. Head output is the length-$V$ vector $[e^{f(v_1)}/Z_\theta, \dots, e^{f(v_V)}/Z_\theta]$. Cost $O(V)$ per step; $O(V \cdot T_x)$ per sequence.
- EBM: $E_\theta : X^{T_x} \to \mathbb{R}$ — one scalar "confidence score" (with a minus sign) per whole sequence; **not** a probability. Normalizing requires $Z_\theta = \sum_{\tilde x \in X^{T_x}} e^{-E_\theta(\tilde x)}$: $V^{T_x}$ terms, intractable.
- Punchline: $O(V \cdot T_x)$ vs. $O(V^{T_x})$. Same debt ($Z$), different payment plans — AR pays in per-token installments, the sequence EBM owes one balloon payment. ("Defer, don't escape.")

### 1.2 Error compounding + the scale hypothesis

- LeCun's argument: $\Pr[x^{<t>} \in \text{correct } \forall t] \approx (1-\varepsilon)^{T_x} \to 0$ exponentially. AR trades cheap normalization for exposure to compounding; the EBM scores the whole object and has no per-step commitment.
- **Brando's hypothesis:** frontier labs make AR work by brute-forcing $\varepsilon$ down via data scale/quality + compute + post-training, lengthening usable horizons (2022: human babysits every step → 2026: agentic trajectories). We cannot compete in that regime; the academic move is the fixed-resources comparison and the conversion strategy in §0.1.
- Caveat both agreed on: the $(1-\varepsilon)^{T_x}$ model assumes independent, unrecoverable, constant-$\varepsilon$ errors — see Brando's existing post on whether that survives verifier-guided systems.

### 1.3 Score matching

- **Observation (the tool):** $Z_\theta$ sums/integrates $x$ out, so it is constant in $x$; hence $\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$. Differentiating w.r.t. the *input* kills the partition function.
- **Score** (reserved term — this is why $E_\theta$ should not be called "the score"): $s(x) := \nabla_x \log p(x)$. Data score $s^*(x) = \nabla_x \log p^*(x)$, $p^* := p_{\text{data}}$.
- **The idea:** demand $\nabla_x \log p_\theta(x) \approx \nabla_x \log p^*(x)\ \forall x$. Matching gradient fields pins the log-densities up to a constant: $\log p_\theta = \log p^* + C$.
- **Why $C = 0$:** normalization has already spent the one remaining degree of freedom. Proof: $1 = \sum_x p_\theta(x) = \sum_x e^{\log p^*(x) + C} = e^C \cdot 1 \Rightarrow C = 0$. Intuition: if $p_1 + p_2 = 1$ must hold, a global constant has nowhere to hide. **Score matching ⇒ $p_\theta = p^*$.** This only works because both objects are probability distributions.

### 1.4 Fisher divergence + notation

- Loss = the dumbest norm available (L2) on the score mismatch, in expectation over data:
  $D^{F}_{p^*}(p^* \Vert p_\theta) := \mathbb{E}_{x \sim p^*}\big[\tfrac12 \Vert \nabla_x \log p_\theta(x) - \nabla_x \log p^*(x) \Vert_2^2\big]$.
- Notation point (Brando's convention): subscript the divergence with the distribution carrying the expectation, since the Fisher divergence is **not symmetric** and standard notation hides which argument is averaged under.

### 1.5 Update rule + optimizer research direction

- Vanilla: $\theta^{<t+1>} := \theta^{<t>} - \eta\, \nabla_\theta D^{F}_{p^*}(p^* \Vert p_\theta)$.
- 2026 generalization: $\theta^{<t+1>} = H\big(\theta^{<t>}, F(-\eta\, \nabla_\theta D^{F}_{p^*}(p^* \Vert p_\theta))\big)$ with $F \in \{\text{SGD, AdamW, Muon, Shampoo}, \dots\}$. Marked **TODO/Experiment**: sweep state-of-the-art optimizers on the SM objective *before* inventing a new one (the sweep doubles as the baseline suite for any invention).

### 1.6 The Hessian point (takeaway level only; derivation deferred)

- Via integration by parts (Hyvärinen), $D^F$ can be rewritten to depend only on $p_\theta$ and *samples* — the price is a Hessian term; only the **trace** (diagonal) is needed, not the full Hessian.
- Whiteboard accounting: full Hessian "$O(|\theta|^2)$" — out; trace "$O(|\theta|)$" — linear, comparable to a forward pass that already touches all parameters. The paper nevertheless claims the trace is impractical (~p. 5). **Brando is skeptical** and wants this tested. (See precision flag §5.1 — the relevant Hessian is w.r.t. $x$, not $\theta$, and the real cost question is "how many backward passes does the exact diagonal need.")
- Brando explicitly flagged he has *not* yet verified the integration-by-parts derivation ("blind trust for now, otherwise I never move forward") — to be re-derived before post 3.

---

## 2. Elyas's pushbacks (worth keeping)

1. **"LLMs aren't even good at Lean."** → Lean is the testbed, not the point of step 1. Step 1 is about inheriting pretraining; Lean enters as the evaluation/verification environment.
2. **"Won't grafting from an AR LLM import AR pathologies (irrecoverable error etc.)?"** → Probably yes for naive conversions: a naive "EBM" that is secretly still an LLM inherits everything and fails — which is informative. The research content *is* finding a conversion that yields a genuine EBM. Strategy: fail fast, iterate.
3. **"Why the trace? Why is it enough?"** → Falls out of the integration-by-parts identity; detailed derivation deferred to tomorrow / post 3.

---

## 3. Experiments agreed

Dataset: **VeriBench** train/val/test split (Fathom transcribed it "VariBench"; Brando to send the split link).

- **B1 (AR baseline):** SFT a small LLM on VeriBench-train; report cross-entropy on VeriBench-test.
- **B2 (EBM baseline):** EBT on the same data — SFT-style if the EBT codebase supports it, otherwise the original paper's recipe; report test cross-entropy.
- **E1 (proposal task):** a concrete grafting method to convert an open-weight LLM into an energy-based transformer; small-scale demo only (idea → hand to Claude to implement).
- **E2 (later):** optimizer sweep on the SM objective, $F \in \{\text{SGD, AdamW, Muon, Shampoo}\}$, fixed everything else.
- **E3 (later):** test the trace-of-Hessian cost claim — benchmark exact diagonal vs. Hutchinson estimator vs. sliced/denoising SM across input dimensions.

---

## 4. Open questions → GitHub issues (paste-ready)

**Issue 1 — Is exact tr(∇²ₓ log p_θ) actually infeasible in 2026? (Song & Kingma ~p.5)**
The tutorial claims the Hessian-trace term in score matching is the computational bottleneck motivating denoising/sliced variants. Benchmark: exact diagonal (one VJP per input dimension) vs. Hutchinson probes vs. SSM vs. DSM, as a function of input dim $D$ and model size, on 2026 hardware/autodiff. Deliverable: cost curves + a crossover analysis (when does exactness beat estimator variance?).

**Issue 2 — Optimizer sweep for score matching (sweep before invent)**
Fix $E_\theta$, data, batch size, step budget. Sweep $F \in$ {SGD, SGD+momentum, AdamW, Shampoo, Muon} on $D^F_{p^*}$. Question: does modern preconditioning change *whether/what* SM trains, or only *how fast*? This sweep is the baseline suite for any optimizer we later design.

**Issue 3 — Bespoke F/H for score objectives**
$\nabla_\theta D^F$ contains mixed $\partial^2 / \partial\theta\,\partial x$ structure absent from plain MLE gradients. Does that structure favor or break particular preconditioners? Design an update rule specialized to score losses (only after Issue 2).

**Issue 4 — Where does x live? Discrete tokens vs. continuous score matching**
$\nabla_x$ is undefined on token space. Decide: embedding-space SM / DSM-on-embeddings vs. discrete SM variants (ratio matching, Hyvärinen 2007; concrete score matching, Meng et al. 2022). Gates B2 and the grafting design.

**Issue 5 — Baselines on VeriBench (B1/B2)**
LLM-SFT test CE vs. EBT (original recipe) test CE, identical train split. Define the split, model scale, and token budget; log everything for the later fixed-resources comparison.

**Issue 6 — Grafting proposal: open-weight LLM → EBM**
Write up one concrete conversion recipe (grafting / distillation-adjacent / other), with the explicit failure hypothesis: naive conversions remain LLMs in disguise and inherit AR pathologies. Define the test that distinguishes "real EBM" from "LLM in a trench coat."

---

## 5. Precision flags (before posts 2–3 and the issues)

1. **x vs. θ Hessian.** The Hessian in the SM identity is $\nabla_x^2 \log p_\theta(x)$ — w.r.t. the **input**, not the parameters. The whiteboard wrote $O(|\theta|^2)$ / $O(|\theta|)$; the correct axes: full Hessian has $O(D^2)$ entries and the trace has $D$ terms, where $D = \dim(x)$. The real cost claim to attack: computing the exact diagonal naively takes ~$D$ backward passes (one VJP per basis direction), not one. Hutchinson's estimator ($\mathrm{tr}\,A = \mathbb{E}_v[v^\top A v]$) gets an unbiased estimate with $O(1)$ Hessian-vector products per probe — that *is* sliced score matching. So the sharp form of the skepticism: "is *exact* trace really infeasible on 2026 hardware, and where is the exactness-vs-variance crossover?" This is the right framing for Issue 1 and post 3.
2. **Discrete x.** Score matching as derived lives in continuous space. For the Lean/LLM setting this is a design decision (Issue 4), not a footnote — it gates the grafting recipe.
3. **Identifiability fine print.** SM ⇒ $p_\theta = p^*$ needs full support + smoothness (Hyvärinen 2005, Thm 2); matching holds $p^*$-a.e. One line added to post 2.

---

## 6. Action items

| Owner | Item | Due |
|---|---|---|
| Elyas | Re-derive today's derivation without notes (10 min/morning routine) | end of week |
| Elyas | Implement B1 + B2 on VeriBench | this week |
| Elyas | Fix SKAMPERE access: email CS IT (transcript: "the election" — likely action@cs), cc Brando | ASAP |
| Brando | Draft score-matching blog post (→ done, published) + why-EBMs post | today |
| Brando | File Issues 1–6 in free-energy repo | today |
| Brando | Send VeriBench train/val/test split link to Elyas | today |
| Brando | Continue Song & Kingma; teach part 2 (Hessian derivation) | tomorrow |
| Brando | Review Elyas's talk/slides | this week |
| Both | Meet **Wed June 10, 2:30 pm PT** (moved from 2:00) | — |

---

## 7. Stray context

- Elyas's lit search: IRED is the best available SOTA baseline he found.
- Claude Fable/Mythos 5 released today; Brando using it for EBM study.
- Brando shared the 10-minute morning re-derivation habit (alternate fundamentals ↔ current research papers).
