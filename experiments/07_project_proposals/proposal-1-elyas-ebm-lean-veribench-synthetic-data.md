# Research Proposal 1 — Energy-Based Reasoning Beyond Toy Tasks: EBMs/EBTs for Lean (VeriBench) with Pretrained Initialization and Synthetic Data

**Working name:** Elyas-EBM
**Lead:** Elyas Obbad · **Co-lead/advisor:** Brando Miranda
**Repo:** https://github.com/brando90/free-energy · **Testbed:** VeriBench V1 (Lean 4)
**Target venue:** ICLR (full-paper deadline Sept 24, 2026); TMLR as faster fallback
**Sources:** meetings 2026-06-08 ([recording](https://fathom.video/share/QPr1xMJsakH_9Sd6zxUqSiA2yQvwHs3Q)) and 2026-06-09 ([recording](https://fathom.video/share/GsBj2jyUo3Xo_pNaqWdvoZ4Ud4SXgdfv)), project slides

---

## 1. Summary

Energy-based models are repeatedly hypothesized to be better suited than autoregressive (AR) LMs for system-2 reasoning, yet the strongest existing EBM-for-reasoning work — IRED (Du, Mao & Tenenbaum, 2024) — is validated only on toy tasks (Sudoku, mazes, matrix operations). We propose to extend the IRED paradigm to a *real* reasoning task: Lean 4 code generation and verification on VeriBench V1. The end-state system composes three ingredients, each solving a distinct EBM failure mode: **IRED-style training/inference** (EBMs are hard to train; IRED's annealed energy landscapes and inference-by-optimization fix training), **the EBT architecture** (the transformer is the best known energy parameterization — self-attention sees the whole input at once; Gladstone et al., 2025), and **grafting** (Chandrasegaran et al., 2025) to initialize the EBM from pretrained open-weight LLMs (gpt-oss-class models, Leanstral, DeepSeek), inheriting billions of dollars of pretraining instead of forfeiting it. A parallel **synthetic-data arm** auto-formalizes open-source software into Lean with LLM assistance to expand the training corpus. The deliverable is one paper: a single results table on VeriBench — {LM baseline, EBM-SOTA-R baseline, Elyas-EBM, all ablations} — i.e., the IRED paper's shape, executed on real data.

## 2. Motivation & background

**Why EBMs at all.** An AR LM pays the partition function $Z$ in per-token installments ($O(V \cdot T_x)$); a sequence-level EBM owes one intractable balloon payment ($O(V^{T_x})$) but judges whole objects holistically, with no per-step commitment to compound. The internal hypothesis — which we deliberately do *not* stake the paper on — is that EBMs are better at system-2 reasoning than AR models. The public framing is broader and safer: EBMs have unresolved fundamental problems in training, inference, and architecture; we advance them on a real task. (Background series: Miranda 2026a,b,c.)

**Why pretrained initialization.** Training an EBM from scratch forfeits the data already digested by open-weight LLMs, and we do not have the compute to compete in that regime regardless. Open-weight models exist; refusing to start from them is leaving billions of dollars of $\varepsilon$-suppression on the table. **Goal (step 1): convert a pretrained LLM into an EBM for Lean.** The from-scratch fair comparison is step 2 and lives in Proposal 2.

**Why Lean / VeriBench.** Lean is the testbed, not the thesis — "AGI works for anything." But it is the *right* testbed: the Lean kernel is a hand-built energy function (whole candidate proof → {valid, invalid}), so judging complete objects — the EBM's native mode — is also the verifier's native mode. VeriBench V1 supplies ~891 evaluation examples plus a train/val/test split (train is small, ~200 examples → treat training as post-training); code and a Colab already exist on Brando's site.

**Why the synthetic-data arm exists (three reasons, one de-risk).** The synthetic corpus is not garnish; it is load-bearing for three independent reasons:
(a) **We want to actually solve VeriBench.** The native train split (~200 examples) cannot get there; the auto-formalized corpus is on the critical path to the benchmark itself, not just to the EBM thesis.
(b) **Insurance.** If the search to make EBMs work fails, the project does not zero out: the synthetic-data corpus plus an **architecture comparison on that corpus** (the Proposal-2 lattice run as a tradeoff study) is a standalone publishable project. The fallback exists from day one, not as a pivot invented after failure.
(c) **Metrics are a deliverable.** Choosing VeriBench-and-model-development as the vehicle forces us to build good measurement instruments for VeriBench — e.g., **LLM judges for trusted equivalence**: theorem equivalence, proof completeness, spec faithfulness. These metrics ship with VeriBench V1 and are contributions regardless of how the EBM thesis resolves.
In short: (a) makes the data necessary, (b) and (c) make the whole program robust to "EBMs don't work."

**Why now.** IRED is from 2024 — geologic time in ML. Phase 0 (below) determines whether a successor exists before we commit to a baseline.

## 3. The paper we are writing

One paper, IRED-shaped: a main table on VeriBench V1 (plus one toy task for continuity with the IRED literature, and optionally one natural-language math task for reviewer coverage) reporting {EBM-SOTA-R, LM baseline, Elyas-EBM} and the full ablation lattice over the three ingredients. Lesson internalized from the ZIP-FIT review cycle (Obbad et al., 2024): pre-empt missing-baseline complaints — the ablation lattice *is* the baseline discipline.

## 4. Hypotheses (falsifiable bets, organized by stack layer)

- **H1 (trained behavior).** A SOTA reasoning EBM applied as-is to VeriBench underperforms a size-matched AR LM baseline. *Falsified if* the EBM matches/exceeds the LM out of the box — which would itself be a headline result.
- **H2 (initialization).** Grafting from a pretrained open-weight LM closes most of the H1 gap at small compute. *Falsified if* grafted-init ≈ random-init at matched fine-tuning budget.
- **H3 (composition).** IRED-training × EBT-architecture × grafted-init strictly dominates every ablation that removes one ingredient. *Falsified by* any ablation cell matching the full composition.
- **H4 (conversion authenticity).** Naive LLM→EBM conversions remain "LLMs in a trench coat" and inherit AR pathologies; a genuine conversion is detectable by the test battery in §7.4. *Falsified if* the naive conversion already passes the battery.
- **H5 (data, stretch).** Auto-formalized synthetic Lean data improves the best EBM monotonically over the accessible scale range.

## 5. Research questions

- **RQ1.** What is the 2026 SOTA EBM for reasoning? (Yilun Du has published at high rate since IRED — is there an IRED-v2?)
- **RQ2.** Which training principle works on real text at this scale: the EBT recipe, IRED's energy-diffusion training, or score matching ($D^{F}_{p^*}$, per the 2026-06-09 derivation)? Sub-question: does the optimizer matter — sweep $F \in \{$SGD, AdamW, Shampoo, Muon$\}$ on the chosen objective before designing anything new (free-energy Issues 2–3).
- **RQ3.** What is a concrete grafting recipe for LLM → EBT, and what test separates a real EBM from a renamed LLM?
- **RQ4.** Does verifier-in-the-loop inference (energy = $E_{\text{LM}} + E_{\text{verifier}}$, minimized iteratively) beat AR decoding at matched inference compute? (Optional/stretch — inference design is unconstrained per 2026-06-08 discussion.)
- **RQ5.** Does the synthetic Lean corpus move the needle (H5)?

## 6. Method: three ingredients, one system

1. **IRED — the training/inference fix.** IRED learns an energy function over candidate outputs conditioned on an input and solves tasks by iteratively optimizing the output to lower energy, with annealed landscapes stabilizing training. We use it as-is or extend it; it is also the *template* for how the paper is written.
2. **EBT — the architecture.** Parameterize $E_\theta$ with a transformer (self-attention over the entire candidate). EBT's LM-style variant keeps next-token energies normalizable at $O(V)$, so cross-entropy remains computable for evaluation.
3. **Grafting — the initialization.** Chandrasegaran et al.'s two-stage recipe transfers directly: (i) *activation distillation* — initialize the energy-head/operator replacements by regressing them onto the pretrained LM's activations; (ii) *lightweight fine-tuning* — repair error propagation with limited data. Their result (new architectures at <2% of pretraining compute) is exactly the budget profile we need. Consult Kesh (Keshigeyan Chandrasegaran) once a first recipe exists.

Score matching enters as the candidate objective wherever MLE's $\log Z_\theta$ is the obstruction; the running Fisher-divergence/optimizer questions are tracked as free-energy GitHub issues.

## 7. Experimental plan

**P0 — SOTA scan (week 1).** Lit review for IRED successors (Du's 2025–26 output first). Output: the named EBM-SOTA-R. Read IRED regardless (paper template). After the first baseline exists, contact Yilun Du (Rylan intro, else cold email) — arrive with "we ran your method on our task; here are the numbers and our directions."

**P1 — Baselines (weeks 1–3).** All on the VeriBench split; primary metric: cross-entropy on VB-test. Per the project slide:

| ID | System | Training | Eval |
|---|---|---|---|
| B1 | LM (small, open-weight) | SFT on VB-train | CE on VB-test (+ VeriBench eval, 891 ex.) |
| B2 | EBT | SFT-style on VB-train (if supported) | CE on VB-test |
| B3 | EBT | IRED training on VB-train | CE on VB-test |
| B4 | EBT | original EBT recipe | CE on VB-test |
| B5 | EBM-SOTA-R (from P0) | as published, then post-train on split | VeriBench eval + CE |

**P2 — Grafting (weeks 3–6).** Propose ≥1 concrete LLM→EBT grafting recipe (idea first; implementation can be delegated to Claude). Small-scale demo only. Run the H4 authenticity battery (§7.4).

**P3 — Composition + ablations (weeks 6–10).** Elyas-EBM = grafted-init × EBT × best training from P1/RQ2. Full ablation lattice → the paper's main table. Add one IRED toy task and (optionally) one NL-math task.

**P4 — Synthetic data + metrics (parallel; Lean AI Club).** LLM-driven auto-formalization of open-source software into Lean 4, generated on skampere3 (streamed working sessions). Deliverables: (i) the corpus + data-scaling curve for the best system (H5); (ii) the **VeriBench measurement suite** — LLM judges for trusted equivalence (theorem equivalence, proof completeness, spec faithfulness), validated against the Lean kernel where checkable — shipping with VeriBench V1; (iii) the corpus doubles as the substrate for the **fallback architecture-comparison paper** (§2, reason b) and for Proposal 2's fixed-data constraint. Role split: Brando+Elyas own the EBM side; a Lean-team collaborator owns auto-formalization.

**P5 — Verifier-in-the-loop (stretch).** $E = E_{\text{LM}} + E_{\text{verifier}}$ minimized at inference; compare to AR decoding at matched inference FLOPs.

### 7.4 The "real EBM vs. LLM in a trench coat" battery (H4)

A grafted model counts as a genuine EBM only if: (a) test-time energy minimization monotonically improves outputs (IRED/EBT-style "thinking" scaling); (b) verifier-valid completions receive lower energy than invalid ones *beyond* what the original AR log-prob already separates; (c) energies respond correctly to global-consistency corruptions that per-token log-probs miss; (d) (optional) any-order/infilling probes succeed where the frozen AR parent fails. Failing the battery is informative — fail fast, iterate on the recipe.

## 8. Data & compute

- **VeriBench V1:** ~891 eval examples; small train split (~200; post-training regime); code + Colab on Brando's site; not yet on arXiv (cite as in-prep).
- **Synthetic corpus:** P4 output; logged so Proposal 2 can reuse it under fixed-data constraints.
- **Compute:** SNAP cluster (skampere1–3). skampere2 for iteration; skampere3 (most capable, often contended) for synthetic-data generation and the largest runs. Track FLOPs per run from day one — Proposal 2 consumes these logs.

## 9. Team, roles, collaborations

Elyas: lead (system named accordingly — this is his PhD-application artifact); P0–P3 execution. Brando: co-lead; EBM theory, score-matching/objective design, blog series, Du/Kesh outreach, VeriBench. Lean AI Club collaborator: auto-formalization (P4). External: Kesh (grafting consult), Yilun Du (after first baseline). Cadence: meet as often as progress allows (daily is fine), Tuesday Lean AI Club as the teaching/sync venue; TODOs live in the Google Doc, not Discord.

## 10. Timeline (backplanned to ICLR, Sept 24 2026)

June: P0 + P1 complete. Early July: P2 grafting demo + authenticity battery. July–Aug: P3 composition/ablations; P4 running in parallel. Late Aug: writing (IRED as template), Du feedback round. Sept: polish, submit. TMLR at any earlier point the result set is self-contained.

## 11. Risks & mitigations

**The big one — the EBM thesis fails outright** → the project never zeroes out, by construction (§2): the synthetic corpus + architecture-tradeoff comparison on it is a standalone paper, and the VeriBench metric suite (LLM judges for theorem equivalence/completeness) ships regardless. EBM training instability → IRED recipe + score matching + the optimizer sweep (RQ2) before inventing anything. Grafting yields an LLM in disguise → §7.4 battery, pre-registered as the kill-test. SOTA moved since IRED → P0 exists precisely for this. Tiny VB-train → post-training framing + P4 synthetic arm. Reviewer demands non-formal benchmarks → toy + NL-math tasks in P3. Hardware contention on skampere3 → iterate on skampere2, batch big runs.

## 12. Relation to Proposal 2

This proposal *leverages* pretraining (step 1). Proposal 2 — AR vs. EBM vs. diffusion at fixed data/compute — deliberately does *not*, and consumes this project's infrastructure: the VeriBench splits, the synthetic corpus, the EBT/IRED training code, the optimizer-sweep results, and the FLOP logs. The two proposals also share the insurance policy: if EBMs fail, Proposal 2's architecture-tradeoff comparison on this proposal's frozen corpus *is* the fallback paper. Sequencing: 1 → 2.

## 13. References

- Yilun Du, Jiayuan Mao, Joshua B. Tenenbaum. *Learning Iterative Reasoning through Energy Diffusion*. ICML 2024. arXiv:2406.11179.
- Alexi Gladstone et al. *Energy-Based Transformers are Scalable Learners and Thinkers*. arXiv:2507.02092, 2025.
- Keshigeyan Chandrasegaran, Michael Poli, Daniel Y. Fu, et al. *Exploring Diffusion Transformer Designs via Grafting*. NeurIPS 2025 (oral). arXiv:2506.05340. https://grafting.stanford.edu
- Yang Song, Diederik P. Kingma. *How to Train Your Energy-Based Models*. arXiv:2101.03288, 2021.
- Aapo Hyvärinen. *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR 6:695–709, 2005.
- Yann LeCun. *A Path Towards Autonomous Machine Intelligence*. OpenReview, 2022.
- Nouha Dziri et al. *Faith and Fate: Limits of Transformers on Compositionality*. NeurIPS 2023.
- Elyas Obbad et al. *ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment*. arXiv:2410.18194, 2024.
- Brando Miranda. *AR Error Compounding — Real or Fiction?* (2026a); *Why Energy-Based Models? The Toy AR-vs-EBM Argument* (2026b); *Score Matching: Training EBMs Without Ever Computing Z* (2026c). Blog series, cs.stanford.edu/people/brando9.
- Brando Miranda et al. *VeriBench V1*. In preparation, 2026; code at cs.stanford.edu/people/brando9.

```bibtex
@inproceedings{du2024ired,  author={Du, Yilun and Mao, Jiayuan and Tenenbaum, Joshua B.},
  title={Learning Iterative Reasoning through Energy Diffusion}, booktitle={ICML}, year={2024}}
@misc{gladstone2025ebt, author={Gladstone, Alexi and others},
  title={Energy-Based Transformers are Scalable Learners and Thinkers}, year={2025},
  eprint={2507.02092}, archivePrefix={arXiv}}
@inproceedings{chandrasegaran2025grafting,
  author={Chandrasegaran, Keshigeyan and Poli, Michael and Fu, Daniel Y. and others},
  title={Exploring Diffusion Transformer Designs via Grafting}, booktitle={NeurIPS}, year={2025},
  note={arXiv:2506.05340}}
@misc{song2021how, author={Song, Yang and Kingma, Diederik P.},
  title={How to Train Your Energy-Based Models}, year={2021}, eprint={2101.03288}, archivePrefix={arXiv}}
@article{hyvarinen2005estimation, author={Hyv{\"a}rinen, Aapo},
  title={Estimation of Non-Normalized Statistical Models by Score Matching},
  journal={JMLR}, volume={6}, pages={695--709}, year={2005}}
@misc{lecun2022path, author={LeCun, Yann}, title={A Path Towards Autonomous Machine Intelligence},
  year={2022}, howpublished={\url{https://openreview.net/pdf?id=BZ5a1r-kVsf}}}
@inproceedings{dziri2023faith, author={Dziri, Nouha and others},
  title={Faith and Fate: Limits of Transformers on Compositionality}, booktitle={NeurIPS}, year={2023}}
@misc{obbad2024zipfit, author={Obbad, Elyas and others},
  title={ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment},
  year={2024}, eprint={2410.18194}, archivePrefix={arXiv}}
```
