All filesystem and Spotlight searches for the paper source have completed or returned empty; the most recent compiled artifact has been fully extracted. Producing the report from the best available evidence.

# REPORT: Map of the "Latent Recovery" Paper As Written

## 0. CRITICAL CAVEAT — SOURCE TREE NOT FOUND; REPORT BUILT FROM LATEST COMPILED PDF

The task prompt's project path was the literal string `undefined` (orchestrator variable-substitution failure). Exhaustive local search found **none** of the requested files:

- Searched (all empty): `find /Users/elyas -maxdepth 7` for `PROFESSOR_FEEDBACK.md`, `PAPER_CHANGELOG.md`, `paper_results_metadata.json`, `paper_results.tex` (excluding `~/Library`); Spotlight (`mdfind`) for `main.tex` (only unrelated homework), `09_latent_recovery`, `latent_recovery`, tex files containing "latent recovery"; iCloud Drive (`~/Library/Mobile Documents`), `~/Library/CloudStorage` (absent), `/Volumes/Data` (AppleInternal only), `~/Desktop/MATS` (unrelated semantic-alignment repo), `~/Desktop/STAIR` (unrelated), `~/Documents`, `~/Downloads`.
- The paper's Reproducibility section says the code lives in `experiments/09_latent_recovery/` of "the project repository" — that repo is not on this machine. Likely location: Stanford SNAP cluster (`skampere2`, `/lfs` scratch, per user memory); SSH was attempted and failed (`Permission denied` — needs `kinit` Kerberos ticket + Stanford VPN, not available non-interactively).
- **PROFESSOR_FEEDBACK.md could not be read — it does not exist locally.** Downstream designers must obtain it (and the true source paths) from the orchestrator/user.

What DOES exist locally, and what this report is built from:
- `/Users/elyas/Desktop/latent_recovery_old_versions/latent_recovery_paper_v3.pdf` (modified 2026-06-11 21:16 UTC — **latest artifact; primary source for this report**; full text extracted to scratchpad)
- `/Users/elyas/Desktop/latent_recovery_old_versions/latent_recovery_paper_v2.pdf` (2026-06-11 17:40)
- `/Users/elyas/Desktop/latent_recovery_old_versions/latent_recovery_workshop_paper.pdf` (2026-06-10 21:45 — oldest, different title/framing)
- `/Users/elyas/Desktop/Literature Briefing on Latent Recovery in Autoregressive Transformers.pdf` (background briefing, not the paper)

v3 is a strict superset of v2 (adds QA-format control, chained-arithmetic brittle task, 32B doubt judge, Holm correction/pre-registration language, "upper bounds" cohort framing, expanded related work 6→12 refs). Everything below describes **v3** unless marked otherwise. If the actual `main.tex` postdates 2026-06-11, treat this as a near-current snapshot, not ground truth.

## 1. Title, Abstract, Main Claims (exact quotes from v3)

**Title:** "Silently Absorbed or Loudly Flagged: Local Falsifiability Governs How Transformers Respond to Planted Errors in Their Own Reasoning"

**Authors:** Elyas Obbad (Stanford, eobbad@stanford.edu), Brando Miranda (Stanford).

(Workshop version had a different title: "Silent Recovery: Behavioral Re-derivation and Latent Trajectory Re-entry After Controlled Reasoning Perturbations in Autoregressive Transformers" — the retitle from "Silent Recovery" to "Silently Absorbed or Loudly Flagged" tracks the inversion of the headline finding after the truth audit.)

**Abstract (v3, verbatim, de-hyphenated):**
> "Compounding-error arguments against autoregressive reasoning assume mid-trajectory errors are unrecoverable. We test the premise causally: take multi-hop deduction problems a pretrained transformer solves, inject a single controlled sentence into the model's own proof, remove all external feedback, and formally validate every step of the continuation. Our central finding required catching our own construct error: a perturbation generator that picks facts 'off the gold proof's path' — the natural reading of wrong — produces statements that are entailed-true 92% of the time in PrOntoQA worlds. Auditing injection truth status inverts the apparent result. True-but-irrelevant interruptions are shrugged off (87% formally valid re-derivation, 0% error adoption). Genuinely false injections split by local falsifiability: lies contradicting evidence available within one inference hop are loudly flagged (up to 39% verbalized doubt) and largely survived (64–75% valid re-derivation), while lies that can only be falsified by global entailment checking are silently absorbed — 0% verbalized doubt, valid re-derivation collapsing to 23–34%, and the planted fact entering the model's subsequent derivation in 29–60% of runs. All falsehoods cause damage exceeding a benign-paraphrase control (p < .03 at every injection position); a formal step-validator separates genuine re-derivation from goal parroting, which inflates naive final-answer 'recovery' by 10–20 points. Two controls secure the reading: recovery survives a question-format condition in which the queried statement's truth value is unknown (57% formally validated, so it is not goal parroting), and on a brittle task with no redundant derivation paths (chained arithmetic) recovery collapses to 0–2% with 99–100% absorption — the regime, not the model, supplies recoverability. All rates are measured on a doubly filtered cohort (instances the model solves, whose unperturbed proofs formally validate) and are upper bounds. For the compounding debate the upshot is sharpened rather than settled: per-step error absorption is real precisely when errors are hardest to check — the dangerous case — while cheaply-checkable errors are not absorbed. Off-path ≠ false is a construct trap for any perturbation study on entailment-dense tasks; we release the truth-status audit with the pipeline."

**Main claims as stated:**
1. (Central) "the model's response to a planted error is governed by how cheaply the error can be falsified against its local context" / "The model's error response is gated not by falsity but by the cost of the check." (Intro §1 and §5.1)
2. (Construct trap) "Off-path ≠ false is a construct trap for any perturbation study on entailment-dense tasks" — 92% of "wrong-category" off-path injections are entailed-true; auditing inverts conclusions.
3. (Metric claim) "Naive final-answer accuracy overstates recovery by 10–20 points" due to goal parroting; a formal step-validator is required.
4. (Regime claim) "'Recoverability' is a property of the task's derivation structure at least as much as of the model." (§5.2)
5. (Explicit non-claim) "We accordingly make no latent-recovery claims: the paper's evidence is behavioral and formally validated." (§5.4)

## 2. Section-by-Section Outline (v3)

| § | Title | Claims | Supporting evidence |
|---|---|---|---|
| Abstract | — | All five claims above | Tables 1–2, §5.2–5.3 controls |
| 1 | Introduction | Compounding argument's "unrecoverable" premise is load-bearing; prior work lacks (i) semantic control over what the perturbation is and (ii) a metric distinguishing re-derivation from goal-reaching. Lists 4 contributions: (1) perturb-and-trace protocol + formal step-validator with 4-way outcome classification; (2) truth-status audit / construct trap; (3) six perturbation families spanning a falsifiability gradient, n=130/position (35 for last family); (4) silent absorption vs. loud flagging finding | Forward refs to Tables 1–2 |
| 2 | Setup | PrOntoQA-OOD, 500 instances (4–6 hops), few-shot canonical proof style, frozen Qwen2.5-7B-Instruct greedy; 91% solve; cohort = solved ∧ gold-validates (84%; n=168 instances; n=130 runs per family×position cell). Protocol: parse own rollout, replace an intermediate derived fact at early/mid/late, continue greedily, no feedback/critique, fixed budget; hidden states cached at 5 depths; null ensemble n=8 temp-0.8 correct continuations per site. Six families (a)–(f) defined; (d)–(f) verified false per-injection against rule closure | Design section; no table |
| 3 | Metrics | Defines valid re-derivation / poisoned / parroted / derailed; discourse-marker stripping (bug caught by benign control); unparsed runs excluded and counted. Two scope notes: closure entailment permits step-skipping (goal-jumps 4.7–9.2% of valid re-derivations at early/mid); poisoning mechanically measurable only for positive category injections → 0% for families (d)–(e) is "a measurement bound, not evidence of resistance". Doubt = lexical detector, "Crude, but the cross-family contrasts it yields exceed 5σ" | — |
| 4 | The truth-status audit | Original "wrong-category" generator's injections are 92% entailed-true; splitting by audited truth inverts headline; "in any entailment-dense benchmark, 'not in the gold derivation' is a property of the proof, not of the world" | **Table 1** |
| 5.1 | The falsifiability gradient | Main result. Benign disruption costs 8–23 pts vs. construction ceiling; true interruptions and distance-0 contradictions cost no more than benign (|z|<1.5 n.s.); genuine falsehoods cost more at every position (negated step z=−2.18/−3.20/−3.64, p=.029/.0014/.0003). Silent absorption: globally-checkable-only falsehoods → 0/70 early+mid doubt, 29–60% poisoning. Loud flagging: 1-hop falsehoods → 18–39% doubt (contrast z=5.5, p<10⁻⁷), 64–75% valid re-derivation. Holm-corrected negated-step-vs-benign p=.029/.0028/.0009; pre-registered primary contrast = wrong/early cell, all else exploratory. Doubt instrument validated by Qwen2.5-32B judge (76–100% agreement; judge doubt .285/.246 vs .015/.000; lexical benign deviation .162 vs .092 understates contrast) | **Table 2** |
| 5.2 | The regime supplies recoverability: total absorption on a brittle task | Chained arithmetic (6 sequential ops, no alternative derivation path): recovery 0–2%, next-step absorption 99–100% (exact validation, n=99/position, gold solve 99%). "the regime map's first off-diagonal cell" | Inline numbers, no table |
| 5.3 | Recovery is not goal-anchor re-reading | (a) QA format ("True or false: s?", s = target or its negation, half each): validated recovery 57% under 1-hop falsehoods, 64% benign (n=35/36; QA gold solve 55%). "Attenuated, not collapsed: goal-anchoring is not a sufficient explanation." (b) Attention knockout on goal span collapses both arms (.015 falsehood / .046 benign) — "the masking breaks the task… we report it as a negative design result"; salvageable secondary: goal-masked chain-soundness identical across arms (.45 vs .42) | Inline numbers |
| 5.4 | Latent trace | ρfunc (null-calibrated JSD at sentence boundaries) shows spike-then-decay above null (≈1.2–2.3), but reported "descriptive only" for two measured reasons: (1) temp-sampled null vs. greedy rollouts inflates denominator; (2) **benign paraphrase shows the largest early divergence of any family (2.32 vs 1.9–2.2 for falsehoods)** — metric tracks style adoption as strongly as semantic damage; "As instrumented, ρfunc cannot distinguish a lie from a stylistic tic." Cosine geometry same story (benign median 2.57 vs falsehoods 2.52–2.99). Late-injection h>2 collapses to n≤12. Layer-0 lexical control degenerate at boundaries. "We accordingly make no latent-recovery claims" | Inline; shuffled-gold control 1.49×; no figure in v3 |
| 5.5 | Scale (reinterpreted) | 1.5B/7B/32B sweep ran with **pre-audit** wrong-category family = predominantly true interruptions. Reread: robustness to true interruptions scales (validated re-deriv .60–.67 → .75–.86 → .88–.96; parroting .11–.20 → .07–.12 → .01–.02; solve .52/.91/.98). "Whether silent absorption of unfalsifiable lies shrinks with scale is the key open question this reframing exposes; the truth-audited families have run only at 7B" | Inline (workshop version had this as its Table 2) |
| 6 | Limitations | See §6 quote below |
| 7 | Related work | Positions vs. von Recum et al. [12], Yee et al. [6] (closest: faithful/unfaithful recovery via manual annotation; delta = automated formal validation, semantic error control, truth audit, falsifiability gradient), Lanham [7], Turpin [8], Zhang snowballing [9], Huang self-correction [10], Lad [2]/McGrath Hydra [4] self-repair, Sun trajectory geometry [11], Dziri [1]/LeCun [3] compounding | — |
| Repro | Reproducibility | "Single 8×H200 node, ∼8 GPU-hours total. Pre-registered plan (PLAN.md, written before the first run), all scripts, per-run JSONL logs, truth-status audit, and result JSONs accompany the paper (experiments/09_latent_recovery/)" | — |

Note: §5.4's text cross-references "negative/descriptive results (§5.2)" in Limitations — a stale internal reference (latent results are §5.4); flag for the LaTeX fix pass.

## 3. Every Experiment in the Paper

| # | Experiment | Model(s) | N | Headline numbers | Where |
|---|---|---|---|---|---|
| E1 | Main grid: 6 families × 3 positions, formal validation | Qwen2.5-7B-Instruct (frozen, greedy) | 130/cell; 35/cell for family (f); cohort 168 instances of 500 | Benign .77/.82/.92 valid; true-interruption .82/.75/.86; distractor .73/.72/.76; contradiction .73/.79/.85 (doubt .23/.27/.18); 1-hop false .65/.64/.75 (doubt .39/.25/.18, poison 0†); globally-checkable false .34/.23/.29 valid, poison .29/.60/.43, doubt .00/.00/.06 | Table 2, §5.1 |
| E2 | Truth-status audit of wrong-category family | 7B | 355 entailed-true vs 35 genuinely-false injections | true: .868 valid / .000 poisoned / .082 parroted; false: .257 / .457 / .200 | Table 1, §4 |
| E3 | Chained arithmetic brittle task | 7B | 99/position; gold solve 99% | recovery 0–2%; next-step absorption 99–100% | §5.2 |
| E4 | QA-format control (parroting removal) | 7B | 35–36/cell; QA gold solve 55% | validated recovery 57% (1-hop false), 64% (benign) | §5.3a |
| E5 | Attention knockout on goal span | 7B | (cells not stated) | both arms collapse (.015 vs .046) → negative design result; chain-soundness .45 vs .42 | §5.3b |
| E6 | Latent functional divergence ρfunc + cosine geometry | 7B | n≥106/point at h≤4 early/mid; late h>2 n≤12; shuffled-gold 1272 boundary pairs | ρfunc ≈1.2–2.3; benign 2.32 > falsehoods 1.9–2.2; cosine benign 2.57 vs false 2.52–2.99; shuffled-gold specificity 1.49× | §5.4 (descriptive only) |
| E7 | Scale sweep (pre-audit family) | Qwen2.5-1.5B/7B/32B | 53–55 / 130 / 93 per cell (workshop numbers) | valid re-deriv .60–.67→.75–.86→.88–.96; parroting .11–.20→.07–.12→.01–.02; solve .52/.91/.98; 1.5B doubt 0/165 | §5.5 |
| E8 | Doubt judge validation | Qwen2.5-32B as judge over 7B mid-injection continuations | per-family (76–100% agreement) | judge doubt .285 (1-hop false), .246 (contradiction) vs .015 (true), .000 (global-only false) | §5.1 |
| E9 | Within-subset selection control for family (f) | 7B | family-(f)-eligible ~30% of instances | negated-step on that subset .676 vs .677 overall → "construction effect, not instance selection" | Table 2 caption |
| E10 | Null ensemble construction | 7B | n=8 temp-0.8 per injection site, ≥2 correct nulls required | calibrates all latent comparisons | §2 |
| E11 | Step-skipping / goal-jump quantification | 7B | — | 4.7–9.2% of valid re-derivations at early/mid | §3 |

## 4. Terminology Audit: "recovery"/"recover"/"repair"

**Precisely defined terms (the paper's actual metric vocabulary, §3):**
- **valid re-derivation**: "iff every entity-fact sentence is entailed by the true world state (premises plus prior valid derivations) under the transitive closure of the question's rules, and the final statement matches the target."
- **poisoned**: "some step is derivable only through the planted atom."
- **parroted**: "target reached despite an underivable intermediate step."
- **derailed**: "wrong final statement."
- **doubt**: "A lexical detector (wait/however/but/actually/contradict/…) over the continuation."
- **validated recovery** (QA condition only, §5.3): "correct verdict + formally sound post-injection chain" — a *second*, differently-operationalized recovery definition.
- **ρfunc_h**: JSD between perturbed and gold next-token distributions at sentence boundaries, normalized by correct-null ensemble spread.

**Loose / unquoted uses of "recovery" (candidates for tightening):**
- Abstract: "recovery survives a question-format condition…", "recovery collapses to 0–2%", "the regime, not the model, supplies recoverability" — bare "recovery" without pointing at which metric.
- §5.2: "yields recovery of 0–2% and next-step absorption of 99–100%" — "recovery" and "absorption" on arithmetic are operationalized only as "(exact validation)" in passing; "next-step absorption" is never formally defined anywhere.
- §5.3 heading "Recovery is not goal-anchor re-reading" and body "the concern that recovery is just attention re-reading."
- §5.5: bare "robustness"; Table 1 caption: "The apparent 'recovery from lies' was recovery from true statements" (rhetorical, but the second "recovery" is unscoped).
- §6: "recovery survives without a parrotable goal."
- Related work: "error recovery," "faithful/unfaithful recovery," "recovery is more frequent for obvious errors" (attributions to others; fine).
- "error adoption" appears once (abstract) as an undefined synonym for poisoning.
- "silent absorption"/"loud flagging" are the headline constructs but are defined only implicitly by their statistics (0% doubt + 29–60% poisoning vs. 18–39% doubt + 0% poisoning); no boxed definition.
- "shrugged off," "largely survived" — rhetorical.
- **Scare-quoted, deliberately loose (good)**: naive final-answer "recovery"; "'Recoverability' is a property of the task's derivation structure…"; "unrecoverable" (the premise under attack).
- **"repair"**: used only for cited literature ("self-repair" Lad/McGrath/Hydra, ×3 + one reference title); never for the paper's own phenomena. No conflict.
- Project/file name is still "latent recovery" (`latent_recovery_paper_v3.pdf`, `experiments/09_latent_recovery/`) though v3 explicitly disclaims latent-recovery claims — a residual naming mismatch.

## 5. What the Paper Says About the Validator (level of detail)

Everything in §3 (plus fragments of §2/§4). Full extent of the description:
> "A continuation is a valid re-derivation iff every entity-fact sentence is entailed by the true world state (premises plus prior valid derivations) under the transitive closure of the question's rules, and the final statement matches the target. Poisoned: some step is derivable only through the planted atom. Parroted: target reached despite an underivable intermediate step. Derailed: wrong final statement. Discourse markers are stripped before parsing (a bug our benign control caught: the model adopts injected style, which broke naive parsing); unparsed runs are excluded and counted. Two scope notes, stated plainly: closure entailment permits step-skipping, so we separately measured derivation-free goal-jumps (4.7–9.2% of valid re-derivations at early/mid positions; at late positions gold itself has no remaining intermediate steps); and poisoning is mechanically measurable only for positive category injections — the grammar derives nothing from negated facts — so 0% poisoning for families (d)–(e) is a measurement bound, not evidence of resistance."

Also: injection truth verified "against the rule closure, per-injection" (§2); cohort requires gold rollout to pass the validator (§2); arithmetic uses "exact validation" (§5.2). **Not given anywhere:** the parsing grammar/sentence-parser spec, how entailment closure is computed, validator failure modes or its own accuracy audit, how many runs were excluded as unparsed (deferred to "reported"/appendix that v3 does not contain), or code-level description. The validator is asserted, not itself validated — likely a professor-feedback pressure point.

## 6. Limitations, Calibration Language, Confound Coverage

**§6 Limitations (v3, essentially verbatim):** Two synthetic task families (PrOntoQA + chained arithmetic; "realistic regimes — natural-language math, Lean theorem proving — are the next arms"); single base-model family ("a reasoning-tuned arm is in progress"); greedy gold rollouts. Prove-format prompt states the target — validator classifies parroting separately and QA-format shows recovery survives without a parrotable goal, "but goal-directedness still aids derivation quality generically (knockout secondary result)". Globally-checkable-only family n=35/cell, exists on only ~30% of instances (within-subset control rules out instance selection); QA condition n=35–36/cell. "'Local falsifiability' is operationalized at two levels (distance 0–1 vs. global); the parametric inferential-distance version is future work." Latent/geometric metrics reported as negative/descriptive. Closure-based validity permits step-skipping (4.7–9.2%). One seed of null sampling; behavioral results deterministic decoding throughout.

**Claim-calibration language already present:** "upper bounds" on a "doubly filtered cohort"; "sharpened rather than settled"; "descriptive only"; "negative design result"; "a measurement bound, not evidence of resistance"; "Attenuated, not collapsed: goal-anchoring is not a sufficient explanation"; pre-registration with one designated primary contrast (wrong/early), "all other behavioral contrasts are exploratory"; Holm correction across positions; "We accordingly make no latent-recovery claims"; scale results explicitly "reinterpreted" post-audit with the 7B-only caveat on truth-audited families.

**Confounds already addressed:** any-disruption (benign paraphrase control); truth status (audit, §4); goal parroting (validator class + QA format + knockout); instance selection for family (f) (within-subset control); lexical-doubt regex artifacts (32B judge); style-propagation confound in latent metrics (measured, disqualifying); null-ensemble decoding mismatch (acknowledged); instance-specificity of ρfunc (shuffled-gold 1.49×); LM-head computation verified against live logits; step-skipping quantified; parsing bug disclosed. **Not addressed:** validator self-audit; prompt-format generality beyond QA; multiple seeds; non-Qwen families; parametric falsifiability distance; whether absorption scales (open by admission).

## 7. Figures/Tables Inventory

**v3 (current): zero figures.** Two tables:
- **Table 1** (§4): truth-audit split of wrong-category family, all positions pooled — 2 rows (entailed-true n=355: .868/.000/.082; genuinely false n=35: .257/.457/.200) × columns {n, Valid re-deriv., Poisoned, Parroted}. Caption: "The same generator's injections, split by audited truth status… The apparent 'recovery from lies' was recovery from true statements."
- **Table 2** (§5.1, main result): 6 families × early/mid/late × {Valid re-deriv. [95% CI], Poisoned, Parroted, Doubt}, n=130/cell (35 for globally-checkable family). Caption includes the within-subset control (.676 vs .677), the † poisoning-not-measurable note for negated injections, and "—" = injection true by construction.
- Scale sweep and chained-arithmetic numbers are **inline prose only** in v3 (the workshop version had the scale sweep as a table and a log-scale 3-panel ρfunc Figure 1, both dropped/demoted).

## 8. Version Lineage (for the changelog the designers can't read)

- **Workshop (06-10)**: "Silent Recovery…" — 3 families (wrong/distractor/contradiction), positive latent re-entry claims (ρfunc spike→collapse→residual 1.2–1.7→full re-entry late), lexical-echo probe confound (§4.4: layer-0 probes 74% vs deep 78%), size-sweep table, Figure 1. Headline: high silent recovery, rare poisoning ≤7%.
- **v2 (06-11 am)**: retitled; truth-status audit added; 6 families/falsifiability gradient; headline inverted to absorbed-vs-flagged; latent demoted.
- **v3 (06-11 pm, current)**: + QA-format & attention-knockout controls (§5.3), chained arithmetic (§5.2), 32B doubt judge, Holm/pre-registration framing, "doubly filtered cohort / upper bounds" abstract language, related work 6→12 refs, latent trace fully disclaimed.

Local file paths for downstream use: `/Users/elyas/Desktop/latent_recovery_old_versions/latent_recovery_paper_v3.pdf` (and v2/workshop siblings); extracted text at `/private/tmp/claude-501/-Users-elyas-Desktop-Alara/ad633660-4a3e-4cbc-99dd-093a0feae95e/scratchpad/{v3,v2,workshop}.txt`. The requested `paper/main.tex`, `paper/paper_results.tex`, `paper/paper_results_metadata.json`, `exp/PAPER_CHANGELOG.md`, and `PROFESSOR_FEEDBACK.md` must be re-supplied with a valid path (probable true home: `experiments/09_latent_recovery/` on the Stanford SNAP cluster, reachable via `ssh skampere2` after `kinit eobbad@CS.STANFORD.EDU` + VPN).