# Research Proposal 3 — Verified Code, Measured Right: the Summer 2026 Program (Measurement Floor · Data-Centric Science · the EBM Bet)

**Working name:** VB-Summer-26 ("floor + upside")
**Owner of this document and of the floor:** Brando Miranda
**Prepared for:** Sanmi Koyejo (advisor) · collaborators: Elyas Obbad (EBM lead), Jerry (post-training), Eshaan Barkataki (grafting engineering), Kesh Chandrasegaran (grafting consult)
**Repos:** [veribench](https://github.com/brando90/veribench) · [cert-judge](https://github.com/brando90/cert-judge) · [free-energy](https://github.com/brando90/free-energy)
**Companion documents:** [Proposal 1 (Elyas-EBM)](proposal-1-elyas-ebm-lean-veribench-synthetic-data.md) · [Proposal 2 (AR-vs-EBM-vs-diffusion fair fight)](proposal-2-ar-vs-ebm-vs-diffusion-fair-comparison.md). This document is the program-level umbrella and the full specification of the *new* piece: the measurement paper (Paper A, the floor).
**Sources:** advising exchange with Sanmi (2026-06-10); planning meetings 2026-06-08 ([recording](https://fathom.video/share/QPr1xMJsakH_9Sd6zxUqSiA2yQvwHs3Q)) and 2026-06-09 ([recording](https://fathom.video/share/GsBj2jyUo3Xo_pNaqWdvoZ4Ud4SXgdfv)); the [VeriBench paper draft](https://cs.stanford.edu/people/brando9/professional_documents/papers/NeurIPS_2026_VeriBench.pdf); the cert-judge paper draft (`cert-judge/paper/neurips2026`).
**Versions:** this is the 8-page version; a [2-page executive version](proposal-3-scsc-measurement-floor-program-2page.md) exists alongside.

---

## 1. Summary

This summer the lab runs three papers in parallel plus one low-cost review track, all on one testbed — **VeriBench** (end-to-end Python→Lean 4 verified-code generation) — and all feeding Brando's thesis:

- **Paper A (the floor — Brando owns it outright).** A measurement paper: does **SCSC** (the Smooth Conjunctive Score for Code verification) measure what it was designed to measure — a smooth, partial-credit surrogate of the gold all-or-nothing metric "all code verifiably works" — and can we build **trustworthy, cheap-to-compute judges and proxies** for it? Validated the way the metric was always meant to be validated: by developing models against it in tandem and checking that climbing SCSC climbs the gold metric. Baselines Brando runs himself: a fine-tuned LM (done) and a discrete-diffusion LM (off-the-shelf recipe). **This paper depends on no EBM result and no intern deliverable. It ships regardless.**
- **Paper B (data-centric — shared).** Solve VeriBench by data: auto-formalize open-source software into Lean 4 at scale, then study data selection (ZIP-FIT), data scaling, and source composition at fixed architecture. The corpus is also Paper C's fuel and Proposal 2's fixed-data constraint, so the work is never wasted.
- **Paper C (the EBM bet — Elyas owns it; pure upside for the program).** Proposal 1 in full: IRED-style training × EBT architecture × grafting from pretrained LLMs, on VeriBench. If it works, it is the headline result *and* it plugs into Paper A as a third model family. If it does not, nothing in Papers A or B is waiting on it.
- **Track R (review/position — opportunistic).** "Are the alleged autoregressive pathologies real, and could alternative architectures (EBMs, diffusion) actually fix them?" — the literature work Papers A–C force us to do anyway, written up; Proposal 2 is its experimental counterpart and runs as the fall capstone on the program's accumulated infrastructure.

The program is explicitly designed around the advising constraint (§2): **the floor is Brando's and is separable from the EBM**; the ambitious project gets full support without putting the thesis clock in anyone else's hands.

## 2. The advising constraint, made structural

From the 2026-06-10 advising discussion, three design requirements this proposal must satisfy — stated up front so the reader can check them against the design:

- **R1 (ownership of the floor).** One chapter-grade paper must be owned outright by Brando and must not depend on the EBM working *or on the intern finishing anything*. Summers are short and undergrads are high-variance through no fault of their own. → Paper A is that paper: Brando can validate the SCSC metrics and judges against LM and diffusion baselines he runs himself (§4.6 separability table).
- **R2 (effort honesty).** "Something publishable either way" is true but underestimates effort: the original scoping stacked three full papers and framed two as safety nets under the third. → This proposal un-stacks them: A, B, C are three *independent* papers with one shared testbed and explicit, one-directional dependencies (§3); safety comes from A's independence, not from B and C catching A.
- **R3 (timeline protection).** A strong publication record protects against having no paper; it does not protect the graduation timeline — you can have papers and still lose a summer. → Hard allocation rule and milestone gates (§8): Brando's default hours go to A first every week; C consumes Brando's *advising* hours, not execution hours; gates that slip cut B's scope, never A's.

## 3. The program at a glance

| Paper | Question | Owner (execution) | Depends on | Risk role | Venue / when | Thesis |
|---|---|---|---|---|---|---|
| **A — Measurement** | Is SCSC a valid, discriminative, optimization-robust surrogate of "all code verifiably works"? Can certified judges/proxies track it cheaply? | **Brando** (+ Jerry: post-training) | VeriBench V1 (exists) | **Floor** | ICLR (Sept 24) or TMLR | Ch. core |
| **B — Data-centric** | Can we solve VeriBench by data: corpus construction, selection (ZIP-FIT), scaling, composition at fixed arch? | Brando + Lean AI Club collab. (+ Elyas: data selection) | A's eval harness | Second floor (shared) | TMLR rolling | Ch. |
| **C — EBM bet** | Do EBMs (IRED × EBT × grafting) reason better than AR on real verified-code generation? (Proposal 1) | **Elyas** (+ Eshaan: grafting impl.; Kesh: consult; Brando: advising) | A's splits & FLOP logging (exist) | **Upside** | ICLR if it lands; TMLR fallback | Ch. (if lands) |
| **R — Review/position** | Are AR pathologies (error compounding, partition-function costs) real, and can EBMs/diffusion fix them? | Brando (low-cost, rolling) | reading A–C already require | Opportunistic | TMLR / blog series | Framing ch. |

**Dependency direction (one-way, by construction):**

```
VeriBench V1 ──► Paper A (floor: metric + judges + LM & diffusion baselines)
                   │ harness, splits, FLOP logs        ▲ optional 3rd column
                   ▼                                   │ (results flow up only)
                 Paper B (corpus, data-centric) ──► Paper C (EBM bet, Elyas)
                   │ frozen corpus                      │
                   ▼                                    ▼
                 Proposal 2 (fall capstone: AR vs EBM vs diffusion, fixed data/FLOPs) ◄── Track R (reading)
```

Arrows point from producer to consumer. Nothing above an arrow waits on anything below it: **A is complete with zero input from B, C, or R.** C's artifacts (EBT runs, FLOP logs) enrich A's third column and Proposal 2's lattice if and only if they exist.

## 4. Paper A — the floor: "Does SCSC measure verified trustworthiness?"

### 4.1 Background: the metric and what's already in hand

VeriBench is an 884-task end-to-end Python→Lean 4 autoformalization benchmark (602-task canonical core incl. a 460-task security split adapted from MIT 6.858; 282-task high-assurance expansion across 14 domains). Its headline score is **SCSC**, a log-domain geometric mean over five per-task factors,

$$\mathrm{SCSC} = \exp\Big(\tfrac{1}{5}\sum_i \log f_i\Big),$$

combining agent-side factors — IC1: the generated Lean file typechecks; IC2: stated theorems verify without `sorry`; TC1: the theorems semantically cover the gold reference — and gold-side benchmark-validity gates (D1, D2). The agent-skill score is $\tilde S_{\text{skill}} = (\mathrm{IC1}\cdot \mathrm{IC2}\cdot \mathrm{TC1})^{1/3}$. The design intent: realize *conjunctive* (all-or-nothing) verification semantics in a smooth, partial-credit form, so a near-zero factor pulls the score toward zero while sub-frontier systems remain comparable.

Already measured (VeriBench paper draft): frontier agents under verifier feedback reach $\tilde S_{\text{skill}}$ of only 0.29 (Codex), 0.22 (Claude Code), 0.10 (Leanstral v2) — five-factor $\tilde S_5$: 0.42, 0.36, 0.23; gold-side $\tilde Q_{\text{gold}} \approx 0.72$. The striking regularity is the **theorem-coverage gap**: agents typecheck nearly everything, yet estimated theorem–gold coverage stalls at ≤ 0.156 (≤ 0.105 for the headline agents). Coverage is estimated by an **LLM coverage judge** whose held-out task-split agreement with five human raters reaches Pearson r = 0.70 under leakage-safe isotonic calibration (zero-shot r = 0.52).

Separately, **cert-judge** proposes *property certification* for LLM judges: humans specify falsifiable behavioral properties once — identity (P1), bug-monotonicity (P2), spec-monotonicity (P3), repeat-stability (P4) — and the trust index $\mathrm{JTI\text{-}v1} = (P_1 P_2 P_3 P_4)^{1/4}$ then grades future judges without recollecting human labels. On an 8-judge calibration panel, JTI-v1 predicts validation-set human Spearman at ρ = 0.881; P3 is the strongest single predictor (ρ = 0.747).

So the two ingredients exist but have not been put under load: **SCSC has never been validated under optimization pressure** (no one has trained models against it and checked that the smooth score tracks the gold one), and **the judges have never been certified the cert-judge way on VeriBench's own artifacts**. Paper A is exactly that load test — which is why developing models in tandem was part of the metric's design plan from the start.

### 4.2 Hypotheses (falsifiable bets)

- **HA1 (surrogate validity under optimization).** Improving a model against SCSC (via SFT/post-training on VB-train) monotonically improves the gold metric ("all code verifiably works") on VB-test; the smooth score does not Goodhart away from the conjunctive one over the accessible improvement range. *Falsified if* a model family climbs SCSC while gold stays flat or drops (reward-overoptimization regime, cf. Gao et al. 2023) — itself a headline result with a concrete fix mandate.
- **HA2 (judge trust transfers).** Property certification (JTI-style, adapted from cert-judge) predicts held-out human agreement for VeriBench's coverage judge and its variants; certified judges keep r ≥ 0.7 on artifacts from *new* model families (diffusion, EBM) they were never calibrated on. *Falsified if* certification fails to transfer across model families — which would mean per-family human re-validation is unavoidable (important negative result for scalable oversight).
- **HA3 (a proxy ladder exists).** There is a hierarchy of ever-cheaper proxies — completion-CE on VB-test ≺ single factors (IC1) ≺ judge-scored TC1 ≺ full SCSC ≺ gold — such that each level usefully predicts the next (rank correlation high enough for model selection) at a fraction of the compute. *Falsified if* cheap signals (CE, typecheck rate) carry no ranking information about gold — which would say "you must pay for full verification every time," also worth knowing.
- **HA4 (discriminative stability across families).** SCSC ranks model families (AR LM, diffusion LM, EBM) and checkpoints within a family stably across training, and its factors decorrelate in interpretable ways (e.g., typecheck saturates early, coverage moves late). *Falsified if* rankings churn with seed/checkpoint noise, i.e., the smooth score is not actually discriminative where it claims to be.

### 4.3 Method: models developed in tandem, metric under load

The validation instrument is a small grid of models that Brando trains and evaluates himself, instrumented end-to-end with the VeriBench evaluation chain:

| ID | System | Training | Status | Role |
|---|---|---|---|---|
| M0 | Open-weight LM, zero-shot | — | **done** (test completion-CE 1.1485) | anchor |
| M1 | Qwen2.5-0.5B | SFT on VB-train | **done** (test completion-CE 0.3124 ± 0.0008, ppl 1.37; free-energy PR #43, end-to-end on SNAP) | AR baseline |
| M2 | Discrete-diffusion LM (MDLM/SEDD-class, off-the-shelf recipe) | from scratch / fine-tune on VB-train | June | revision-without-denormalization control |
| M3 | Stronger open LM + post-training (SFT → RL-style loop against SCSC factors) | post-train on VB-train (+B's corpus when it exists) | July, with **Jerry** | the optimization-pressure probe for HA1 |
| M4 | EBT/EBM (from Paper C) | per Proposal 1 | **optional** — plugs in if C delivers | third family for HA2/HA4; *not on A's critical path* |

(The from-scratch EBT control already ran without instabilities — test completion-CE 2.7794, ppl 16.1 — so the harness demonstrably handles a non-AR family end-to-end today, independent of C's success.)

Experiments, mapped to hypotheses:

1. **Optimization-pressure curves (HA1).** For M1 and M3 checkpoints across training: plot SCSC, $\tilde S_{\text{skill}}$, each factor, and the gold all-or-nothing rate on VB-test. Deliverable: the "does smooth track gold?" figure — the paper's centerpiece — plus the first public characterization of *where* the surrogate bends (which factor saturates first, where partial credit stops being informative).
2. **Judge certification (HA2).** Port cert-judge's P1–P4 battery to VeriBench's coverage judge (and 2–3 prompt/model variants); compute JTI; validate against the existing five-rater human labels on a held-out task split; then stress-test transfer by scoring M2/M4 artifacts and re-checking human agreement on a small fresh sample (~100 items, the only new human-label spend in the paper).
3. **Proxy ladder (HA3).** Across all checkpoints of all families: rank-correlation matrix between {completion-CE, typecheck rate IC1, sorry-free rate IC2, judged coverage TC1, SCSC, gold}. Deliverable: a "what can you afford to measure?" table with compute cost per proxy (CE is ~free; full chain is ~hours/model) — the practical artifact other groups will cite.
4. **Stability & factor anatomy (HA4).** Seed/checkpoint repeats on M1–M2; factor-decomposition plots over training; flag any factor whose noise dominates its signal.

### 4.4 Deliverables

(i) The validity study (HA1 figure + HA4 anatomy); (ii) certified-judge protocol + JTI numbers for VeriBench judges (HA2), with the transfer result; (iii) the proxy-ladder table (HA3); (iv) released harness: splits, FLOP-logged training/eval scripts, judge-certification code — which is precisely the infrastructure Papers B/C and Proposal 2 consume. One paper, one chapter, Brando's name on the critical path and no one else's.

### 4.5 Venue & timing

Built back from ICLR (full-paper deadline Sept 24, 2026): results table frozen late Aug, writing Sept. TMLR at any earlier self-contained point if ICLR timing gets tight — the paper is scientific-question-shaped, so TMLR is a natural fit, and the floor does not gamble on a deadline.

### 4.6 Separability table (the R1 check, explicit)

| Does Paper A need… | Answer |
|---|---|
| …the EBM to work? | **No.** M4 is an optional third column; HA1–HA3 close on M1–M3. |
| …Elyas to finish anything? | **No.** All of A's critical-path models are Brando-run; Elyas's outputs only *add* a column. |
| …Eshaan's grafting to land? | **No.** Grafting feeds C only. |
| …Jerry's post-training? | **Soft no.** M3 strengthens HA1's pressure probe; if it slips, M1's SFT curves already answer HA1 at smaller dynamic range. |
| …the synthetic corpus (B)? | **No.** A runs on VB-train/val/test as released; B's corpus only extends M3's data if ready. |
| …new human labels? | Mostly no — reuses the five-rater coverage dataset; one small (~100-item) fresh sample for HA2 transfer. |

## 5. Paper B — data-centric: solve VeriBench by data

**Question.** Holding architecture fixed (the best AR recipe from A's grid), how far does *data* go on VeriBench — and which data? **Hypotheses:** HB1: an auto-formalized corpus of real open-source software (LLM-assisted Python→Lean translation, verifier-filtered) beats size-matched generic Lean data; HB2: targeted selection (ZIP-FIT-style compression alignment — Elyas's method) beats random selection at equal token budget; HB3: data scaling on the corpus is monotone over the accessible range and its slope is steeper for verification-specific factors (TC1) than for typechecking (IC1), i.e., data buys *semantics* not just *syntax*.

**Plan.** June–July: corpus pipeline on skampere3 (streamed working sessions; Lean-verifier filtering; frozen snapshot with hash + manifest — the same snapshot Proposal 2's fixed-data constraint requires). July–Aug: selection & scaling experiments using A's harness and metrics. Deliverables: the corpus (released), the data-centric results, and a data-quality section that doubles as SCSC-in-practice evidence for A. **Ownership:** Brando + Lean AI Club collaborator on the pipeline; Elyas advises on ZIP-FIT; none of A blocks on it (one-way dependency, §3).

## 6. Paper C — the EBM bet (Elyas's project; Proposal 1 in full)

The ambitious project, fully supported and fully separable: extend the IRED paradigm beyond toy tasks to real verified-code generation, composing IRED-style training, the EBT parameterization, and grafting from pretrained open-weight LLMs (full spec, baselines B1–B5, kill-tests, and timeline in [Proposal 1](proposal-1-elyas-ebm-lean-veribench-synthetic-data.md)). **Roles:** Elyas — lead (his PhD-application artifact); **Eshaan — grafting engineering** (the LLM→EBT conversion recipe: activation distillation + light fine-tune), with **Kesh** consulting once a concrete recipe exists; **Brando — advising** (theory, objectives, weekly deep-dives) — by design his *advising* hours, not execution hours. **Program-level accounting:** C inherits A's splits, FLOP logging, and eval harness on day one (they exist — PR #43), so Elyas starts from a running system rather than building infrastructure. Everything C produces flows up as upside: a working EBM gives A its third model family (HA2/HA4 transfer tests) and gives Proposal 2 its hardest arm. If C stalls, the program loses nothing it was counting on — and Elyas still graduates the summer with the B2-class EBT baselines, the grafting battery, and co-authorship on whatever his artifacts fed.

## 7. Track R — the review/position paper, and Proposal 2 as the fall capstone

Papers A–C force a structured read of the same literature: AR error-compounding claims, partition-function critiques, EBM training/inference fixes (IRED, EBT, score matching), diffusion LMs as revision-with-normalization. Track R writes that reading up as a review/position paper — "the case for and against autoregression for verified generation, and what would settle it" — built on the existing blog series (Miranda 2026a–c), at low marginal cost, venue TMLR or a strong workshop, absorbing useful effort even in weeks when experiments stall. Its experimental counterpart is **Proposal 2** (AR vs. EBM vs. diffusion from scratch at fixed data/params/FLOPs), which kicks off ~July after VeriBench V1 ships and consumes the program's accumulated assets (A's harness and FLOP logs, B's frozen corpus, C's EBT code). R states the questions sharply; Proposal 2 answers them; either order of completion is publishable.

## 8. Timeline, gates, and the allocation rule

| When | Paper A (floor) | Paper B | Paper C (Elyas et al.) | R / P2 |
|---|---|---|---|---|
| June wk 1–2 | M2 diffusion baseline runs; judge-certification port begins | corpus pipeline v0 | P0 SOTA scan; P1 baselines (B1/B2 done) | reading notes |
| June wk 3–4 | **Gate G1:** HA1 curves for M1 + M2 exist end-to-end | first verifier-filtered shard | P1 complete | — |
| July | M3 post-training with Jerry; HA2 certification + transfer | **Gate G2:** frozen corpus snapshot; HB2 selection runs | P2 grafting (Eshaan ramps, Kesh consult) | P2 kickoff (smallest grid) |
| Aug | full grid; **Gate G3:** results table frozen ~Aug 24 | HB3 scaling | P3 composition + ablations | blog interim |
| Sept | write; ICLR Sept 24 (or TMLR) | TMLR when self-contained | ICLR if it lands; else TMLR later | R drafting |

**Allocation rule (R3, enforced weekly):** Brando's default execution hours go to A until the current gate is met; B gets the remainder; C gets fixed advising slots (deep but bounded — e.g., the Tuesday Lean AI Club sync plus ad-hoc theory sessions). **Gate-slip policy:** a slipped gate cuts B's scope (corpus size, number of selection methods) and never A's; two consecutive slipped gates trigger a scope conversation with Sanmi, not silent stretching.

## 9. Team & proposed ownership

| Person | Role | Critical path of |
|---|---|---|
| Brando Miranda | Program owner; Paper A end-to-end; B co-lead; C advisor; R author | **A** (only A) |
| Elyas Obbad | Paper C lead (Proposal 1); ZIP-FIT advice on B | C |
| Eshaan Barkataki | Grafting engineering for C (LLM→EBT recipe implementation) | C (grafting arm) |
| Jerry | Post-training for A's M3 (and B's best-recipe runs) | enhances A, blocks nothing |
| Kesh Chandrasegaran | Grafting consult (method author) | — (consult) |
| Lean AI Club collaborator(s) | B's auto-formalization pipeline | B |
| Sanmi Koyejo | Advisor | — |

Compute: SNAP cluster — iterate on skampere2, batch big runs and corpus generation on skampere3; FLOPs logged per run from day one (Proposal 2 consumes the logs).

## 10. Risks & mitigations

- **The stacked-projects trap (R2).** *Risk:* three papers quietly become one over-coupled mega-project again. *Mitigation:* the one-way dependency graph (§3) is the contract; any new cross-dependency needs to be written into this doc explicitly (and justified) before anyone builds on it.
- **Losing the summer with papers in hand (R3).** *Risk:* progress everywhere, completion nowhere. *Mitigation:* gates G1–G3 with the gate-slip policy; A's results table freeze (~Aug 24) is the hard line.
- **HA1 comes out negative (SCSC Goodharts).** Not a program failure: a documented overoptimization regime for conjunctive verification metrics, with factor-level diagnosis, is a strong measurement paper on its own (and directly actionable for VeriBench V2).
- **Judge certification doesn't transfer (HA2 negative).** Publishable negative for scalable oversight; bounds what JTI-style certification can promise; the per-family re-validation cost gets quantified rather than assumed away.
- **M2 diffusion baseline is weak at this scale.** Use it for HA2/HA4 (artifact diversity) regardless; HA1 closes on M1/M3 alone. Lean on curve *shape*, not absolute accuracy.
- **EBM thesis fails outright (C).** By construction nothing of Brando's is on the line (R1); Elyas keeps the baselines, the battery, and co-authorship; A and B are unaffected; Proposal 2 still runs as the pros-and-cons paper.
- **Corpus pipeline slower than hoped (B).** B's scope is the designated shock absorber (§8); the frozen-snapshot requirement is minimal (any size snapshot freezes).
- **Compute contention on skampere3.** Iterate on skampere2; batch corpus generation; A's grid is small-model by design.

## 11. Thesis mapping

- **Paper A** → core measurement chapter: *trustworthy evaluation of verified code generation* (metric validity, certified judges, proxy ladder).
- **Paper B** → data-centric chapter: *what data buys verification* (corpus, selection, scaling).
- **Paper C / Proposal 2** → architectures chapter: *beyond autoregression for verified generation* (the bet if it lands; the controlled comparison either way).
- **Track R** → framing material for the introduction/related-work spine.

Any single row already advances the thesis; A alone is a complete summer outcome (R1); the modal outcome is A + most of B + C's baselines; the upside outcome adds a working EBM and a headline.

## References

- Brando Miranda et al. *VeriBench: End-to-End Formal Verification Benchmark for AI Coding Agents in Lean 4*. Preprint, 2026. [PDF](https://cs.stanford.edu/people/brando9/professional_documents/papers/NeurIPS_2026_VeriBench.pdf).
- Brando Miranda et al. *Property Certification for Trustworthy LLM Judges (JTI-v1)*. In preparation, 2026. Repo: [cert-judge](https://github.com/brando90/cert-judge).
- Leo Gao, John Schulman, Jacob Hilton. *Scaling Laws for Reward Model Overoptimization*. ICML 2023. arXiv:2210.10760.
- Yilun Du, Jiayuan Mao, Joshua B. Tenenbaum. *Learning Iterative Reasoning through Energy Diffusion*. ICML 2024. arXiv:2406.11179.
- Alexi Gladstone et al. *Energy-Based Transformers are Scalable Learners and Thinkers*. arXiv:2507.02092, 2025.
- Keshigeyan Chandrasegaran, Michael Poli, Daniel Y. Fu, et al. *Exploring Diffusion Transformer Designs via Grafting*. NeurIPS 2025 (oral). arXiv:2506.05340.
- Elyas Obbad et al. *ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment*. arXiv:2410.18194, 2024.
- Subham Sahoo et al. *Simple and Effective Masked Diffusion Language Models (MDLM)*. NeurIPS 2024. arXiv:2406.07524.
- Aaron Lou, Chenlin Meng, Stefano Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)*. ICML 2024. arXiv:2310.16834.
- Nouha Dziri et al. *Faith and Fate: Limits of Transformers on Compositionality*. NeurIPS 2023.
- Yang Song, Diederik P. Kingma. *How to Train Your Energy-Based Models*. arXiv:2101.03288, 2021.
- Brando Miranda. *AR Error Compounding — Real or Fiction?* (2026a); *Why Energy-Based Models? The Toy AR-vs-EBM Argument* (2026b); *Score Matching: Training EBMs Without Ever Computing Z* (2026c). Blog series, cs.stanford.edu/people/brando9.

```bibtex
@misc{miranda2026veribench, author={Miranda, Brando and others},
  title={VeriBench: End-to-End Formal Verification Benchmark for AI Coding Agents in Lean 4},
  year={2026}, note={Preprint}}
@misc{miranda2026certjudge, author={Miranda, Brando and others},
  title={Property Certification for Trustworthy LLM Judges (JTI-v1)},
  year={2026}, note={In preparation}}
@inproceedings{gao2023scaling, author={Gao, Leo and Schulman, John and Hilton, Jacob},
  title={Scaling Laws for Reward Model Overoptimization}, booktitle={ICML}, year={2023},
  note={arXiv:2210.10760}}
@inproceedings{du2024ired, author={Du, Yilun and Mao, Jiayuan and Tenenbaum, Joshua B.},
  title={Learning Iterative Reasoning through Energy Diffusion}, booktitle={ICML}, year={2024}}
@misc{gladstone2025ebt, author={Gladstone, Alexi and others},
  title={Energy-Based Transformers are Scalable Learners and Thinkers}, year={2025},
  eprint={2507.02092}, archivePrefix={arXiv}}
@inproceedings{chandrasegaran2025grafting,
  author={Chandrasegaran, Keshigeyan and Poli, Michael and Fu, Daniel Y. and others},
  title={Exploring Diffusion Transformer Designs via Grafting}, booktitle={NeurIPS}, year={2025},
  note={arXiv:2506.05340}}
@misc{obbad2024zipfit, author={Obbad, Elyas and others},
  title={ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignment},
  year={2024}, eprint={2410.18194}, archivePrefix={arXiv}}
@inproceedings{sahoo2024mdlm, author={Sahoo, Subham and others},
  title={Simple and Effective Masked Diffusion Language Models}, booktitle={NeurIPS}, year={2024}}
@inproceedings{lou2024sedd, author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  title={Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution},
  booktitle={ICML}, year={2024}}
@inproceedings{dziri2023faith, author={Dziri, Nouha and others},
  title={Faith and Fate: Limits of Transformers on Compositionality}, booktitle={NeurIPS}, year={2023}}
@misc{song2021how, author={Song, Yang and Kingma, Diederik P.},
  title={How to Train Your Energy-Based Models}, year={2021}, eprint={2101.03288}, archivePrefix={arXiv}}
```
