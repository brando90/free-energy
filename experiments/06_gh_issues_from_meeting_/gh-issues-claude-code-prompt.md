# Claude Code Prompt — Create GitHub Issues for `brando90/free-energy`

You are Claude Code operating in (or against) the repo **brando90/free-energy**. Your job: create the GitHub issues defined below

> **Status:** created 2026-06-09, assigned to @eobbad — Issue 1 → [#40](https://github.com/brando90/free-energy/issues/40), Issue 2 → [#41](https://github.com/brando90/free-energy/issues/41), Issue 3 → [#42](https://github.com/brando90/free-energy/issues/42).

Shared context links (already embedded in bodies where needed):
- Meeting 2026-06-08: https://fathom.video/share/QPr1xMJsakH_9Sd6zxUqSiA2yQvwHs3Q
- Meeting 2026-06-09: https://fathom.video/share/GsBj2jyUo3Xo_pNaqWdvoZ4Ud4SXgdfv
- Song & Kingma, *How to Train Your Energy-Based Models*: https://arxiv.org/abs/2101.03288
- Blog series: https://cs.stanford.edu/people/brando9/blog.html

---

## Issue 1

**Title:** Is the exact Hessian trace in score matching actually infeasible? (Song & Kingma ~p.5)
**Labels:** research-question, score-matching, experiment, theory

### Body
**Context.** Hyvärinen's integration-by-parts rewrites the Fisher divergence as the SM objective — eq. (8) in [Song & Kingma (arXiv:2101.03288)](https://arxiv.org/abs/2101.03288): $\mathbb{E}_{p^*}[\tfrac12\|\nabla_x \log p_\theta(x)\|_2^2 + \mathrm{tr}(\nabla_x^2 \log p_\theta(x))]$ — depending only on $p_\theta$ and samples, at the price of the Hessian-trace term. The authors treat this trace as *the* bottleneck motivating denoising/sliced variants. Brando's skepticism ([2026-06-09 meeting](https://fathom.video/share/GsBj2jyUo3Xo_pNaqWdvoZ4Ud4SXgdfv)): is it really impractical in 2026?

**Precision notes (frame the experiment — attack the real claim).**
1. **Axis.** The Hessian is w.r.t. the **input** $x$ (dim $D$), not the parameters $\theta$. Per *backward pass* the cost is $O(|\theta|)$ — that part of the "it's linear" intuition is right — but the relevant multiplier is the **pass count**, and $D$ scales with $\dim(x)$ (for sequences in embedding space, $D = T_x \cdot d_{\mathrm{emb}}$).
2. **The claim is pass-count, not term-count.** Nobody disputes the trace has $D$ terms. The standard claim (e.g., Meng et al., *Autoregressive Score Matching*, arXiv:2010.12810): the exact trace naively needs ~$D$ more backward passes than one gradient, and exact Hessian diagonals of *arbitrary* computation graphs are believed uncomputable in $O(1)$ forward/backward passes (Martens, Sutskever & Swersky, arXiv:1206.6464).
3. **Beyond-Scale caveat.** The diversity-coefficient/Task2Vec evidence is the **empirical Fisher diagonal w.r.t. $\theta$** = elementwise squared gradients = **one** backprop per sample. Different object: it shows *a* linear-sized curvature diagnostic is cheap, not that the exact Hessian diagonal is. Don't lead with this analogy when arguing the issue.
4. **Surviving (sharpened) skepticism — the actual experiments:** (a) is $D$ passes truly prohibitive on 2026 hardware/autodiff (vmap'd VJPs, forward-over-reverse), and does it amortize against total training budget ("expensive" ≠ "infeasible")? (b) where is the exact-vs-estimator crossover vs Hutchinson ($\mathrm{tr}\,A = \mathbb{E}_v[v^\top A v]$, $O(1)$ HVPs/probe — which *is* sliced SM, i.e., the authors' own remedy)? (c) the impossibility framing is for *arbitrary* graphs — transformers are not arbitrary; do structured $E_\theta$ admit cheaper exact diagonals?

**Tasks**
- [ ] Benchmark wall-clock + memory: exact diagonal ($D$ VJPs) vs Hutchinson ($k$ probes, $k \in \{1,4,16\}$) vs SSM vs DSM, sweeping input dim $D$ and model size.
- [ ] Try `torch.func` vectorized HVP/jacobian paths and forward-over-reverse; report where vectorization changes the picture.
- [ ] Produce crossover plot: when (if ever) does exactness beat estimator variance per unit compute? Include the amortization view (per-step cost ÷ total budget).
- [ ] Check the structured-architecture loophole: does a transformer $E_\theta$ admit an exact-diagonal shortcut that an arbitrary graph doesn't?
- [ ] Conclude: confirm or refute the eq.-(8) infeasibility claim; feed result into blog post 3.

**Deliverable.** Benchmark script in repo + cost-curve plots + one-paragraph verdict.
---

## Issue 2

**Title:** Optimizer sweep for score matching: SGD vs AdamW vs Shampoo vs Muon (sweep before invent)
**Labels:** experiment, optimizers, score-matching

### Body
**Context.** Training is descent on $D^F_{p^*}$: $\theta^{<t+1>} = H(\theta^{<t>}, F(-\eta \nabla_\theta D^F_{p^*}))$. The SM literature predates the modern optimizer stack; nobody has systematically swept 2026-grade $F$ against score objectives. From the [2026-06-09 meeting](https://fathom.video/share/GsBj2jyUo3Xo_pNaqWdvoZ4Ud4SXgdfv): plug in every SOTA optimizer *before* inventing one — the sweep is the baseline suite (and the alibi) for any later invention.

**Tasks**
- [ ] Fix $E_\theta$ architecture, dataset, batch size, step budget; vary only $F \in$ {SGD, SGD+momentum, AdamW, Shampoo, Muon}.
- [ ] Log $D^F$ train curves, eval metric, wall-clock, and stability incidents (divergence/NaNs).
- [ ] Answer: does preconditioning change *whether/what* SM trains, or only *how fast*?
- [ ] Repeat the winning $F$ on a second dataset for robustness.

**Deliverable.** Sweep config + results table + curves; recommendation of default $F$ for the project.
**Owner.** Elyas.
---

## Issue 3

**Title:** Propose a grafting recipe: open-weight LLM → EBT (activation distillation + light fine-tune)
**Labels:** research-question, grafting, proposal-1

### Body
**Context.** Project goal (step 1): convert a pretrained open-weight LLM (gpt-oss-class, Leanstral, DeepSeek) into an EBM, inheriting pretraining instead of forfeiting it. Template: [Chandrasegaran et al., *Exploring Diffusion Transformer Designs via Grafting*, NeurIPS 2025 oral, arXiv:2506.05340](https://arxiv.org/abs/2506.05340) — (i) activation distillation to initialize replacement operators against the pretrained model's activations, (ii) lightweight fine-tuning; new architectures at <2% pretraining compute.

**Tasks**
- [ ] Write ≥1 concrete recipe: which operators/heads get replaced, what the energy head is, what the distillation targets are, what gets fine-tuned.
- [ ] Small-scale demo only (idea first; implementation can be delegated to Claude Code).
- [ ] Consult Kesh (Keshigeyan Chandrasegaran) once a first recipe exists.
- [ ] Decide where $x$ lives (embedding-space SM vs discrete-SM variants) and define the kill-test the recipe must pass (the "real EBM vs LLM in a trench coat" battery).

**Deliverable.** Recipe doc + minimal demo run.
**Owner.** Elyas (proposal) / Brando (review, Kesh intro).
---