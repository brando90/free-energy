# Latent Recovery Pilot → Workshop Paper — Status Update (2026-06-10)

**TL;DR:** The full experimental program ran end-to-end today on skampere2. The workshop
paper is drafted and compiles: `paper_latex/papers/latent_recovery/main.pdf` (4pp + table
+ figure, zero placeholder numbers). Headline: pretrained transformers behaviorally
recover from planted reasoning errors at high, formally-validated rates that scale with
model size; latently they re-enter the vicinity of the gold trajectory within 2–3
sentences but carry a persistent, decaying functional trace. Recovery is silent.

Every number below is read from a results file on disk (path given). Nothing in this
update is from memory or extrapolation.

## What ran (all on skampere2, GPUs 0–5, ~6 hours wall-clock)

| Stage | Scale | Output |
|---|---|---|
| Gold rollouts | 3 models × ≤500 instances | `results*/gold/rollouts.jsonl` |
| Perturbed rollouts | 7B: 3 families × 3 points × 161 inst; 1.5B/32B: 1 family | `results*/perturbed*/runs.jsonl` |
| Null ensembles | n=8 per injection site, all models | `results*/null/` |
| Formal validation | all of the above | `results*/validated_summary*.json` |
| Functional ρ (JSD) | all models | `results*/functional_rho.json` |
| Probe suite + sweep | 7B | `results/probes/{probe_accuracy,sweep}.json` |
| Controls | shuffled-gold, LM-head verification, bootstrap CIs | `results/shuffle_control.json`, in-log assert, in-JSON CIs |

## Headline results

1. **Behavioral recovery, formally validated** (`results/validated_summary*.json`):
   72–86% of perturbed continuations at 7B re-derive a *formally valid* proof (entailment
   checking under rule closure), across three perturbation families (wrong-category /
   distractor / contradiction) and three injection points. n=130 runs/cell.
2. **The parroting correction**: naive final-answer "recovery" is 94–97%
   (`results/stats.json`) — inflated 10–20 points by goal parroting, since the prompt
   states the target. This gap is itself a finding: final-answer robustness metrics
   overstate recovery.
3. **Silent recovery / detection dissociation**: doubt language in ≤3% of wrong-step and
   distractor continuations vs 18–27% for direct contradictions, at similar recovery
   rates. The model notices contradictions, not plausible lies — and verbalizes neither reliably.
4. **Latent (functional) re-entry** (`results/functional_rho.json`, fig
   `results/fig_rho_func_log.png`): ρ_func spikes at the first post-injection sentence
   (median 4–140× null spread), collapses to 1.2–3× within 2–3 sentences, leaves a
   persistent decaying residual for early/mid injections (h=8 CIs exclude 1, e.g.
   wrong/early 1.59 [1.25, 2.05]), and fully re-enters (ρ→1.00) for late injections.
   **Behavioral recovery does not require full latent re-entry.**
5. **Size trend** (`results_1p5b|results_32b/validated_summary.json`): valid
   re-derivation 60–67% (1.5B) → 75–86% (7B) → 88–96% (32B); parroting falls 11–20% →
   7–12% → 1–2%; solve rates 52/91/98%. Recovery — validated, not parroted — scales.

## Honest caveats (also in the paper's Limitations)

- **Target-in-prompt**: ProofsOnly format states the goal; the validator addresses the
  behavioral consequence, latent precommitment confounds remain partially open.
- **One regime**: PrOntoQA is semantically redundant by design. The regime map
  (brittle/Lean arm) is NOT run — it's the next experiment and the ICLR-scale story.
- **Cross-size ρ comparison is confounded** (each model's null spread differs); only
  within-size ρ shapes are interpretable. Stated explicitly in §4.3.
- **Probe metric demoted**: linear probes hit a lexical-echo confound (layer-0 probes
  ≈ deep-layer probes, `results/probes/sweep.json`) and pre-registered kill-threshold
  (67–78% < 80%). Functional JSD replaced it — this followed the PLAN.md kill-condition,
  not preference.
- **1.5B statistical weight**: 53–55 runs/cell (short chains often lack 3 injection
  points); h≥4 latent horizons largely truncate.
- Excluded-from-cohort runs (gold not formally valid): 93/483 (7B), 87 (1.5B), 66 (32B) —
  counted, not hidden.
- Doubt detection is lexical (regex), greedy gold only, one model family.

## What was NOT done (so nobody assumes it was)

- No Lean/VeriBench (brittle-regime) arm. No path-patching/mechanism localization.
- No horizon-warped trajectory alignment (sentence-index matching only).
- No precommitment-timing analysis (perturb-before-decodable conditioning).
- Paper not pushed to git; nothing committed — repo state is local to skampere2.
- Blog post not drafted.

## Verified citations only

von Recum 2602.07470, Lad/Gurnee/Tegmark 2406.19384, McGrath 2307.15771 (web-verified
today); Saparov & He 2210.01240 (repo); Dziri 2305.18654, LeCun 2022 (established).
The deep-research briefing's other citations were NOT verified and are not in the paper.

## Suggested next steps (in order of leverage)

1. Read the PDF; red-pen the framing before showing Brando.
2. The brittle-regime arm (Lean via the lab's VeriBench harness) — turns this from a
   workshop paper into the ICLR regime-map paper.
3. Precommitment-timing control; mechanism localization via path patching.
4. git add experiments/09_latent_recovery paper_latex/papers/latent_recovery + commit.

---

# ADDENDUM (2026-06-11) — The inversion: v1's headline was a construct artifact

**Read this before the section above.** Elyas's question "how did you guarantee that you
corrupted?" exposed that the v1 "wrong-category" injections were off the gold *proof's*
path but **entailed-TRUE in the world 92% of the time** (truth audit:
`audit run, see results/perturbed*/runs.jsonl + validator closure check`). v1's
"recovery from lies" was recovery from true statements. Corrected findings (paper v2,
`paper_latex/papers/latent_recovery/main.pdf`, on Elyas's Desktop as
`latent_recovery_paper_v2.pdf`):

- Truth-status cross-tab (wrong family): entailed-true injections .868 valid / .000
  poisoned vs genuinely-false .257 valid / **.457 poisoned** (n=355/35).
- Two new guaranteed-false families (per-injection falsity verified vs rule closure):
  - **negated-step** (1-hop checkable, n=130/cell): .65/.64/.75 valid, doubt .39/.25/.18,
    significantly worse than benign at all positions (p=.029/.0014/.0003).
  - **off-path category** (globally checkable only, n=35/cell): .34/.23/.29 valid,
    poisoned .29/.60/.43, doubt ~0 — **silent absorption**.
- New headline: **local falsifiability governs the response** — cheap-to-check lies get
  flagged and survived; hard-to-check lies get silently absorbed into the derivation.
- Latent metric demoted to descriptive: benign paraphrase shows the LARGEST early
  divergence (rho_4=2.32) — style propagation dominates; metric can't separate style
  from semantics as instrumented.
- Scale section reinterpreted: the size sweep used pre-audit (mostly-true) injections →
  reads as "robustness to true interruptions scales"; falsehood-family size sweep NOT run.
- Validator v2 (discourse-marker stripping; bug caught by the benign control). Poisoning
  note: mechanically measurable only for positive category injections.

Caveats that remain open: category-falsehood n=35/cell; matched-decoding latent
calibration not run; single task/model-family for truth-audited results; nothing
committed to git.

---

# ADDENDUM 2 (2026-06-11 PM) — Batch-2: dose-response, regime map, scale, family replication

Paper v4 on Desktop (`latent_recovery_paper_v4.pdf`). All pre-registered in PLAN.md
before launch; outcomes vs predictions:

- **Dose-response (CONFIRMED, the centerpiece)**: doubt falls monotonically with hops-to-
  falsifying-evidence: .392 (k=1) → .323 (k=2) → .231 (k=3) → .000 (global); recovery
  .646 → .562 → .531 → .26-.34. n=130/cell early. `results/validated_summary_neghop*.json`
- **Regime map (CONFIRMED, stronger than predicted)**: GSM8K recovery .088, absorption
  .877 (n=57; format-filtered cohort 57/250) — realistic math lands at the brittle end.
  Refinement: corrupted GSM8K values ARE 1-hop checkable; models re-check stated facts,
  not recomputations. `results/gsm8k/summary.json`
- **Scale (PARTIALLY FALSIFIED pre-registration)**: absorption attenuates at 32B
  (valid .60-.73 vs 7B .23-.34; poisoned .10-.37 vs .29-.60) — we predicted persistence.
  Silence persists at all scales (ack ≤.07). `results_32b/validated_summary_*.json`
- **OLMo-2 family (PARTIAL)**: recovery gradient replicates; flagging does NOT —
  OLMo verbalizes ~nothing anywhere. Verbalization is training-dependent; recovery
  isn't. `results_olmo/validated_summary_*.json`
- **Matched decoding (CONFIRMED)**: benign ≈ falsehood persists with decoding matched
  (h4 1.36 vs 1.58) — latent demotion is final. `results/matched_rho.json`
- **Stability**: rho unstable across null configs (medians 1.9-7.9, IQRs 2 orders of
  magnitude). `results/rho_stability.json`
- 32B judge validated the doubt regex (76-100% agreement, contrast sharpened).
  `results/doubt_judge.json`

Still running: R1-distill (gold 147/500 solved, 51% think-truncation; perturb stage in
progress), false-ontology generation. Not started: Lean arm, mechanism slice.
