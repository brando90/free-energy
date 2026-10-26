# Research Proposal 3 (2-page) — Verified Code, Measured Right: the Summer 2026 Program

**Working name:** VB-Summer-26 ("floor + upside") · **Owner:** Brando Miranda · **Prepared for:** Sanmi Koyejo (advisor); collaborators Elyas Obbad, Jerry (post-training), Eshaan Barkataki, Kesh Chandrasegaran
**Repos:** [veribench](https://github.com/brando90/veribench) · [cert-judge](https://github.com/brando90/cert-judge) · [free-energy](https://github.com/brando90/free-energy)
**Full version:** [8-page](proposal-3-scsc-measurement-floor-program-8page.md) · Companions: [Proposal 1 (Elyas-EBM)](proposal-1-elyas-ebm-lean-veribench-synthetic-data.md), [Proposal 2 (fair fight)](proposal-2-ar-vs-ebm-vs-diffusion-fair-comparison.md)

---

## 1. The design constraint

From the 2026-06-10 advising discussion: (R1) one chapter-grade paper must be owned outright by Brando, depending on **no EBM result and no intern deliverable**; (R2) the original scoping stacked three full papers as mutual safety nets — un-stack them into independent papers with one-way dependencies; (R3) a publication record protects against "no papers," not against a lost summer — so the floor gets hard gates and a weekly allocation rule. This document is the check that the program satisfies all three.

## 2. The program: three parallel papers + one cheap track, one testbed (VeriBench)

| Paper | Question | Owner | Risk role |
|---|---|---|---|
| **A — Measurement (floor)** | Is **SCSC** a valid, discriminative, optimization-robust surrogate of "all code verifiably works"? Can property-certified judges and cheap proxies track it? | **Brando** (+ Jerry post-training) | **Ships regardless** |
| **B — Data-centric** | Solve VeriBench by data: auto-formalized open-software corpus; selection (ZIP-FIT), scaling, composition at fixed arch | Brando + Lean AI Club (+ Elyas advice) | Second floor; scope = shock absorber |
| **C — EBM bet** | IRED × EBT × grafting on real verified-code generation (Proposal 1) | **Elyas** (+ Eshaan grafting, Kesh consult; Brando advises) | **Pure upside** |
| **R — Review/position** | Are AR pathologies real; can EBMs/diffusion fix them? (Proposal 2 = its experimental capstone, ~July kickoff) | Brando, rolling | Opportunistic |

**Dependencies are one-way:** VeriBench → A → (harness, splits, FLOP logs) → B → (frozen corpus) → C and Proposal 2. Results flow back *up* only as optional extra columns. **A is complete with zero input from B, C, or R.**

## 3. Paper A — the floor, in one column

**Why it's needed.** VeriBench's SCSC ($\exp\frac{1}{5}\sum_i \log f_i$ over typecheck IC1, sorry-free IC2, judged coverage TC1, gold-side D1–D2) was designed as a smooth surrogate of the gold conjunctive metric, and its coverage judge is human-calibrated (held-out Pearson r = 0.70). But the metric has never been validated **under optimization pressure**, and the judges have never been **certified** (cert-judge JTI-v1 style: properties P1–P4, panel ρ = 0.881 predicting human agreement). Developing models in tandem was the design plan from the start.

**Hypotheses.** HA1: climbing SCSC climbs the gold metric (no Goodhart over the accessible range; cf. reward overoptimization). HA2: JTI-style certification predicts held-out human agreement for VB judges *and transfers* to artifacts from new model families. HA3: a proxy ladder exists — completion-CE ≺ factors ≺ SCSC ≺ gold — each level predicting the next at a fraction of the compute. HA4: SCSC ranks families/checkpoints stably, with interpretable factor anatomy. Every negative outcome is itself a publishable measurement result.

**Instrument (all Brando-run).** M0 zero-shot LM (done: completion-CE 1.1485) · M1 Qwen2.5-0.5B SFT (done: 0.3124 ± 0.0008, ppl 1.37; free-energy PR #43, end-to-end on SNAP) · M2 diffusion LM (MDLM/SEDD-class, June) · M3 stronger LM post-trained against SCSC factors (July, with Jerry) · M4 EBM **optional** third family from Paper C — not on the critical path (a from-scratch EBT control already ran cleanly: CE 2.7794). Deliverables: the smooth-vs-gold curves, certified-judge protocol + transfer result, the proxy-cost table, and the released harness that B/C/Proposal 2 consume. Venue: ICLR (Sept 24) or TMLR; thesis core chapter.

**Separability check (R1):** needs the EBM to work? **No.** Needs Elyas/Eshaan to finish anything? **No.** Needs Jerry? Soft no (M3 widens HA1's range; M1 already answers it). Needs B's corpus? **No.** New human labels? ~100 items only (reuses the five-rater coverage dataset).

## 4. Timeline, gates, allocation

**June:** M2 runs; judge-certification port; **G1 (end June): HA1 curves exist end-to-end for M1+M2.** **July:** M3 with Jerry; HA2 transfer; **G2: frozen corpus snapshot (B).** **Aug:** full grid; **G3 (~Aug 24): A's results table frozen.** **Sept:** write; ICLR Sept 24 or TMLR. *Allocation rule:* Brando's default execution hours go to A until the current gate is met; B takes the remainder; C gets fixed, bounded advising slots (Tuesday Lean AI Club + theory sessions). *Gate-slip policy:* slips cut B's scope, never A's; two consecutive slips → scope conversation with Sanmi.

## 5. Team

Brando — program owner, A end-to-end, B co-lead, C advisor, R author (critical path of A **only**). Elyas — C lead (his PhD-application artifact). Eshaan — grafting engineering for C (Kesh consulting). Jerry — post-training (A's M3, B's best-recipe runs). Lean AI Club collaborator — B's pipeline. Compute: SNAP (skampere2 iterate / skampere3 batch), FLOPs logged per run.

## 6. The ask

Does this satisfy the constraint — the floor clearly mine, clearly separable from the EBM, with the timeline protected by gates rather than by optimism? Full hypotheses, experiment grid, risk table, and thesis mapping in the [8-page version](proposal-3-scsc-measurement-floor-program-8page.md).
