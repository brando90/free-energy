# Research Proposal 2 — Autoregression vs. Energy vs. Diffusion: A Controlled Comparison at Fixed Data, Compute, and Parameters

**Working name:** the AR-vs-EBM(-vs-diffusion) fair fight (a.k.a. "the autoregressiveness project")
**Co-leads:** Brando Miranda (more involved per 2026-06-08 scoping) · Elyas Obbad
**Repo:** https://github.com/brando90/free-energy · **Testbed:** VeriBench V1 + frozen synthetic Lean corpus (from Proposal 1)
**Target venue:** TMLR-first posture (rolling, scientific-paper-friendly); ICLR if the result set matures in time
**Public artifact:** biweekly blog-post series (posts 1–3 already drafted/published)
**Sources:** meetings 2026-06-08 ([recording](https://fathom.video/share/QPr1xMJsakH_9Sd6zxUqSiA2yQvwHs3Q)) and 2026-06-09 ([recording](https://fathom.video/share/GsBj2jyUo3Xo_pNaqWdvoZ4Ud4SXgdfv)), project slides. *Diffusion arm added per Brando's 2026-06-09 follow-up (extends meeting scope); the contract axis is deliberately open — per Kesh, the tradeoffs between architectures are a scientific paper in their own right.*
**Length:** main body ≈ 2 pages; compute matching, measurement protocol, dependencies, risks, and references in the Appendix.

---

## 1. Summary

The claim "energy-based models beat autoregressive models" conflates two independent critiques of AR: (i) **error compounding** — per-token commitment makes long generations collapse like $(1-\varepsilon)^{T_x}$ — and (ii) the **partition function** — softmax+MLE forces you to pay $Z$ and shapes what the model can represent. These are orthogonal axes, and no controlled experiment we know of separates them. We propose the experiment: AR LMs vs. masked/discrete **diffusion** LMs vs. EBMs/EBTs, all trained **from scratch** on identical data, identical parameter count, identical training FLOPs — the explicitly *un*-pretrained counterpart to Proposal 1 ("step 2"). Diffusion is the load-bearing third arm: it has **iterative refinement and error revision while still paying the per-token softmax**, so it dissociates "revision fights compounding" from "denormalization fights compounding." Pre-registered outcome contract from the planning meetings: either our EBM innovation beats the baselines at fixed resources, **or** the paper is a careful scientific characterization of the pros and cons of the three contracts. Both outcomes are deliverables; the second is the floor, not a failure.

## 2. The experiment's logic: three payment plans, two confounded mechanisms

All three contracts write $e^{\text{score}}/Z$ somewhere; they differ in *where and when* $Z$ is paid and *whether outputs can be revised*:

| Contract | Normalization | $Z$ payment plan | Revision? |
|---|---|---|---|
| AR LM | softmax per position, once | $O(V \cdot T_x)$ | none (blind rollout) |
| Masked/discrete diffusion LM | softmax per position **per denoising step** | $O(V \cdot T_x \cdot S)$ — *more* installments | built-in (iterative denoising) |
| Sequence EBM (EBT) | over $X^{T_x}$, deferred | $O(V^{T_x})$, ideally never | built-in (energy minimization) |

This table is the experiment's logic. If the EBM's hypothesized reasoning advantage comes from **revision/iterative refinement**, diffusion — which buys revision *while keeping normalization* — should capture most of it. If the advantage requires **denormalization itself** (margin/energy objectives, no softmax bottleneck at the head), the EBM should beat diffusion at matched inference compute. Without the diffusion arm these two stories are indistinguishable. The compounding side has the same structure: the recoverable-Markov alternative to the geometric error model (Miranda 2026a) predicts that *any* recovery mechanism — a verifier for AR/EBM, denoising steps for diffusion — flattens success-vs-length curves; Sean's standing objection (EBMs-as-normalization-choice do not inherently fix divergence) is the position this design takes seriously.

## 3. Hypotheses (pre-registered bets)

- **H1 (null ordering to beat).** At fixed data/params/FLOPs, likelihood-type metrics order AR ≥ diffusion ≥ naive EBM. *Falsified if* any cell breaks the ordering — localize which ingredient did it via the lattice.
- **H2 (revision, not normalization, fights compounding).** Success-vs-length curves: diffusion-blind already flattens relative to AR-blind (built-in revision, softmax intact); verifier columns flatten everything further; the normalization row explains little residual variance. *Falsified if* EBM-blind robustly beats diffusion-blind on length scaling at matched inference compute — that would mean denormalization carries independent weight.
- **H3 (independence of the axes — the decisive test).** Holding decoding completely fixed and swapping **only the head/objective** (softmax+MLE ↔ Z-free energy head with a margin/energy objective) changes calibration, robustness, and ranking quality but does **not** change the fitted compounding exponent. *Falsified if* the head swap alone moves the exponent — which would mean the partition-function and divergence critiques are *not* independent, itself a headline.
- **H4 (regime existence + attribution).** There exist regimes — long horizons, verifier-rich inference, high compositional depth (Dziri et al. 2023 splits) — where the EBM contract wins at matched inference compute. Attribution via the control: if the EBM wins only where diffusion also wins, the win is iterative refinement; if the EBM beats diffusion, denormalization matters. *Falsified if* no cell of the regime grid favors any EBM arm.

## 4. Design

**Controlled:** tokenizer, data (VeriBench-train + the Proposal-1 frozen synthetic Lean corpus), parameter count, training FLOPs (logged per run; Proposal-1 instrumentation), eval suite. From scratch — no pretrained initialization anywhere in this proposal, by construction.

**Factorial lattice** (affordable projection first; compute matching, metrics, and implementations: Appendix B):

| Axis | Levels |
|---|---|
| Contract | AR next-token · masked/discrete diffusion LM · sequence-level EBM (EBT) · *(axis open)* |
| Objective | MLE (softmax + CE) · diffusion ELBO / denoising score matching · energy/margin (Z-free) · score matching $D^{F}_{p^*}$ |
| Head (decisive arm) | softmax head · energy head, **decoding held fixed** |
| Inference | blind · verifier-in-loop (Lean kernel; backtrack/resample) |

**Decision rules (pre-registered).** If some EBM arm wins a regime → the innovation paper, with the lattice attributing the win to contract, objective, head, or inference — and the diffusion control settling refinement-vs-denormalization. If no arm wins → the pros-and-cons paper across the contracts, with the head-swap result settling whether the two anti-AR arguments are independent. Either way the paper ships. **This is also the program-level de-risk:** the architecture-tradeoff comparison on Proposal 1's frozen corpus is the standing fallback if "making EBMs work" fails — the EBM thesis failing does not zero either project.

## 5. Timeline & process

**July 2026:** kick off after VeriBench V1 ships (Brando switches focus here in earnest then, per 2026-06-08); freeze data snapshot + FLOP accounting; smallest grid = {AR, diffusion} × {blind, verifier} — mature codebases make these the cheap arms, so they run first. **Aug:** EBM arm + head swap (H3) + objective axis; first length-curve fits across all three contracts; blog the interim. **Sept–Oct:** regime grid (H4), optional attention axis, writing. Venue: TMLR whenever self-contained; ICLR only if H3/H4 land before the Sept 24 deadline (not assumed). Biweekly public posts regardless of paper state; Tuesday Lean AI Club = teaching/sync venue; coordinate the verifier-guided search/judge component with Sanmi's lab to divide rather than duplicate.

---

# Appendix

## A. Motivation detail: where $Z$ is paid, and the risk posture

$Z$ is paid at distinct sites — the attention axis, the output-head/vocabulary axis, and implicitly through MLE — and conflating them prevents causal attribution. The contribution is the disentanglement, at fixed resources, on a real task with a hard verifier (Lean) available as the recovery mechanism.

Risk posture, stated up front (2026-06-08): scientific-question papers are less reviewer-appreciated than SOTA papers and need more ML background. Mitigations: Brando co-leads with heavier involvement; TMLR-first; the biweekly blog series builds the audience and the writing in public; pre-registration converts "no winner" into a publishable answer.

## B. Design detail

**Inference-compute matching across contracts.** One AR pass + $k$ verifier resamples ↔ $S$ denoising steps ↔ $K$ energy-minimization steps, FLOP-matched; $S$ and $K$ are the test-time-compute knobs and get their own scaling curves.

**Optional fifth axis — attention normalization** (softmax attention · sigmoid attention): targets the third $Z$-payment site (attention is itself an EBM update — Ramsauer et al. 2020; sigmoid attention as the denormalized drop-in — Ramapuram et al. 2024). Include only if the four-axis grid lands early.

**Objective-axis note.** The diffusion ELBO / denoising-score-matching level includes SEDD score-entropy as the discrete-score bridge — ties to free-energy Issue 4 (where the score lives for discrete $x$).

**Diffusion implementations.** Off-the-shelf small discrete-diffusion codebases (D3PM / SEDD / MDLM-class), trained from scratch on the frozen corpus — the AR and diffusion arms are cheap (mature code) and run first; the EBM arm is the hard one.

**Measurement protocol.** Per arm: (a) test CE where defined — EBT's LM-style next-token energies normalize at $O(V)$ so CE is computable; diffusion reports an **ELBO-based NLL upper bound, not exact CE** (comparability caveat stated in-paper); pure sequence-EBMs report ranking/contrastive metrics; (b) verifier pass rate and compute-matched pass@k; (c) **success-vs-length curves with model fits** — geometric $(1-\varepsilon)^T$ vs. recoverable-Markov — reporting which error model fits which arm (the Miranda 2026a protocol, now run on all three contracts); (d) compositional-depth generalization splits; (e) calibration of energies/log-probs against verifier validity.

## C. What this is *not*

Not a leverage-pretraining project (that is Proposal 1). Not a SOTA chase in any of the three families — "it's very unlikely we beat models trained on frontier-scale data, and that is not the question." Not a diffusion-LM methods paper: the diffusion arm is a control, run with standard recipes. Model sizes are set by what the SNAP cluster trains comfortably from scratch, because the comparison, not the absolute number, is the product.

## D. Dependencies on Proposal 1 (sequencing: 1 → 2)

Consumed directly: VeriBench V1 splits; the frozen synthetic Lean corpus snapshot; EBT/IRED training code; optimizer-sweep results (free-energy Issues 2–3) so each objective starts from its best-known $F$; per-run FLOP logs for the fixed-compute constraint. New, small: a discrete-diffusion training pipeline (off-the-shelf). Nothing here blocks Proposal 1; everything here reuses it.

## E. Risks & mitigations

Reviewer appetite for "scientific" papers → TMLR-first, pre-registration, public series. Metric incomparability across contracts (CE vs ELBO bound vs energies) → shared metrics defined up front (App. B); likelihood-type numbers reported only with their caveats. From-scratch models too weak for signal → verifier pass and ranking metrics designed to be informative at small scale; lean on length-curve *shape*, not absolute accuracy. Diffusion arm scope creep → control-arm discipline: standard recipes only. Compute contention → smallest-grid-first; skampere2 iteration / skampere3 batched runs. Scope creep into Proposal 1 → the no-pretraining rule is the firewall.

## F. References

- Yann LeCun. *A Path Towards Autonomous Machine Intelligence*. OpenReview, 2022.
- Yann LeCun et al. *A Tutorial on Energy-Based Learning*. In *Predicting Structured Data*, MIT Press, 2006.
- Nouha Dziri et al. *Faith and Fate: Limits of Transformers on Compositionality*. NeurIPS 2023.
- Yang Song, Diederik P. Kingma. *How to Train Your Energy-Based Models*. arXiv:2101.03288, 2021.
- Aapo Hyvärinen. *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR 6, 2005.
- Alexi Gladstone et al. *Energy-Based Transformers are Scalable Learners and Thinkers*. arXiv:2507.02092, 2025.
- Yilun Du, Jiayuan Mao, Joshua B. Tenenbaum. *Learning Iterative Reasoning through Energy Diffusion*. ICML 2024. arXiv:2406.11179.
- Jacob Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)*. NeurIPS 2021. arXiv:2107.03006.
- Aaron Lou, Chenlin Meng, Stefano Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)*. ICML 2024. arXiv:2310.16834.
- Subham Sahoo et al. *Simple and Effective Masked Diffusion Language Models (MDLM)*. NeurIPS 2024. arXiv:2406.07524.
- Shen Nie et al. *Large Language Diffusion Models (LLaDA)*. arXiv:2502.09992, 2025.
- Yang Song, Stefano Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS 2019. arXiv:1907.05600.
- Pascal Vincent. *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation, 2011.
- Hubert Ramsauer et al. *Hopfield Networks is All You Need*. ICLR 2021. arXiv:2008.02217.
- Jason Ramapuram et al. *Theory, Analysis, and Best Practices for Sigmoid Self-Attention*. arXiv:2409.04431, 2024.
- Brando Miranda. *AR Error Compounding — Real or Fiction?* (2026a); *Why Energy-Based Models? The Toy AR-vs-EBM Argument* (2026b); *Score Matching: Training EBMs Without Ever Computing Z* (2026c). Blog series, cs.stanford.edu/people/brando9.
- Brando Miranda et al. *VeriBench V1*. In preparation, 2026.

```bibtex
@misc{lecun2022path, author={LeCun, Yann}, title={A Path Towards Autonomous Machine Intelligence},
  year={2022}, howpublished={\url{https://openreview.net/pdf?id=BZ5a1r-kVsf}}}
@incollection{lecun2006tutorial, author={LeCun, Yann and Chopra, Sumit and Hadsell, Raia and Ranzato, Marc'Aurelio and Huang, Fu Jie},
  title={A Tutorial on Energy-Based Learning}, booktitle={Predicting Structured Data}, publisher={MIT Press}, year={2006}}
@inproceedings{dziri2023faith, author={Dziri, Nouha and others},
  title={Faith and Fate: Limits of Transformers on Compositionality}, booktitle={NeurIPS}, year={2023}}
@misc{song2021how, author={Song, Yang and Kingma, Diederik P.},
  title={How to Train Your Energy-Based Models}, year={2021}, eprint={2101.03288}, archivePrefix={arXiv}}
@article{hyvarinen2005estimation, author={Hyv{\"a}rinen, Aapo},
  title={Estimation of Non-Normalized Statistical Models by Score Matching},
  journal={JMLR}, volume={6}, pages={695--709}, year={2005}}
@misc{gladstone2025ebt, author={Gladstone, Alexi and others},
  title={Energy-Based Transformers are Scalable Learners and Thinkers}, year={2025},
  eprint={2507.02092}, archivePrefix={arXiv}}
@inproceedings{du2024ired, author={Du, Yilun and Mao, Jiayuan and Tenenbaum, Joshua B.},
  title={Learning Iterative Reasoning through Energy Diffusion}, booktitle={ICML}, year={2024}}
@inproceedings{austin2021d3pm, author={Austin, Jacob and Johnson, Daniel D. and Ho, Jonathan and Tarlow, Daniel and van den Berg, Rianne},
  title={Structured Denoising Diffusion Models in Discrete State-Spaces}, booktitle={NeurIPS}, year={2021}}
@inproceedings{lou2024sedd, author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  title={Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution}, booktitle={ICML}, year={2024}}
@inproceedings{sahoo2024mdlm, author={Sahoo, Subham and others},
  title={Simple and Effective Masked Diffusion Language Models}, booktitle={NeurIPS}, year={2024}}
@misc{nie2025llada, author={Nie, Shen and others}, title={Large Language Diffusion Models},
  year={2025}, eprint={2502.09992}, archivePrefix={arXiv}}
@inproceedings{song2019generative, author={Song, Yang and Ermon, Stefano},
  title={Generative Modeling by Estimating Gradients of the Data Distribution}, booktitle={NeurIPS}, year={2019}}
@article{vincent2011connection, author={Vincent, Pascal},
  title={A Connection Between Score Matching and Denoising Autoencoders},
  journal={Neural Computation}, volume={23}, number={7}, year={2011}}
@inproceedings{ramsauer2021hopfield, author={Ramsauer, Hubert and others},
  title={Hopfield Networks is All You Need}, booktitle={ICLR}, year={2021}}
@misc{ramapuram2024sigmoid, author={Ramapuram, Jason and others},
  title={Theory, Analysis, and Best Practices for Sigmoid Self-Attention},
  year={2024}, eprint={2409.04431}, archivePrefix={arXiv}}
```
