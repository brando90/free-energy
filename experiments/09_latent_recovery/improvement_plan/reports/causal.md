# EXPERIMENT PROGRAM: Isolating Local Refutability from Its Confounds
## (Weakness #1 remediation plan — TMLR primary target, ICLR causal-isolation variant)

**Evidence base & caveats (read first).** `PROFESSOR_FEEDBACK.md` does not exist on this machine (confirmed by fresh `mdfind` + three prior exhaustive searches); this design is keyed to the professor's confound list as quoted in the task (negation presence, lexical contradiction/overlap, statement type, on/off-path relevance, derivational usability, stylistic change, category frequency) plus the gaps the paper itself admits. The design is grounded in the **newest draft** (`/Users/elyas/Desktop/main.pdf`, Jun 23, TMLR double-blind format, extracted to `/private/tmp/claude-501/-Users-elyas-Desktop-Alara/ad633660-4a3e-4cbc-99dd-093a0feae95e/scratchpad/main_jun23.txt`), whose Tables 1–8 and Figure 1 I verified directly. The experiment source tree is **not local** (lives in `experiments/09_latent_recovery/` on the SNAP cluster, `ssh skampere2` after `kinit eobbad@CS.STANFORD.EDU` + VPN); all file-level integration points below are named from the paper's repro section and the inventory reports and **must be verified against the repo before implementation**. Grammar-feasibility claims below are anchored to what the paper proves is feasible: expc's local-positive-**attribute** cell exists (Table 5), PrOntoQA-OOD has negative rules in-grammar ("Every dumpus is not opaque", "Every impus is not red" appear verbatim in Table 2's example world), and neghop-k depth-targeting machinery exists (Figure 1).

---

## 1. The causal variable, formalized — and the two design moves that isolate it

**Definition (pre-register verbatim).** For a planted claim *P* injected at proof state *S* (stated premises + trace prefix at the injection point), the **refutation distance** d(P,S) is the length of the shortest derivation, under the world's rule closure, of a statement contradicting *P* starting from facts in *S*. d=0: a stated sentence contradicts *P* directly. d=k: k rule applications needed. d=∞: no refutation exists in the closure (*P* is merely unentailed — closed-world false). This replaces the informal "local falsifiability"; it is computable by BFS over the rule graph and should be **emitted as a per-injection integer by the audit**, not a family label.

**Why the current gradient is confounded (verified in the draft).** Every locally-refutable falsehood in Table 3 is a *negation* ("Stella is not mean."); the only d=∞ falsehood is an *affirmative categorical* ("Stella is an impus.") that is also off-path, derivationally usable (the example shows the model consuming it via "Every impus is not red"), a different falsity type (unentailed vs contradicted), and the only family whose absorption the validator can mechanically measure (Table 3 footnote †). So distance is currently perfectly collinear with polarity, usability, path status, falsity type, and measurability. The expc attempt to break this (Table 5) died at 7 problems because it **filtered pre-existing worlds** instead of generating worlds to spec.

**Move 1 — generate worlds to spec (the attribute route).** PrOntoQA attribute rules ("Every X is F" / "Every X is not F") attach to any category in the chain and take both polarities natively, and attributes have no outgoing rules (nothing chains *from* "Stella is mean"). Therefore, by generating ontologies where a single attribute rule attaches at a chosen depth beyond the injection point, we get a **fully symmetric polarity × distance factorial in which falsity type (provably contradicted), usability (inert), statement type (attribute fact), style (identical sentence frame), and subject (active entity) are all held constant by construction** — the exact cross expc could not fill. Bonus balance property: the AFF-false cell pairs a *positive claim* with a *negative rule*, the NEG-false cell a *negative claim* with a *positive rule*, so total negation count in {context ∪ claim} is constant across polarity cells; and each world hosts a mirrored true/false claim pair (§2.1).

**Move 2 — move the evidence, not the sentence (the evidence-mover).** The strongest possible isolation: hold the planted sentence **byte-identical** across conditions and vary only the surrounding evidence, generalizing the existing artifact-locked cert-flip (Table 4). All seven professor axes are properties of the planted sentence or its relation to context; if the sentence never changes and only the inferential distance of inserted (true, subject-matched, frame-matched) evidence varies, every sentence-property confound is dead by construction.

**Move 3 — fix the dependent variable.** Injection-dependence is only mechanically measurable for usable positive-category claims (draft, §2.2), so the DV is currently confounded with the design. Add a **belief probe**: a separate post-hoc forward pass appending `Question: True or false: "<planted claim>"? Answer:` to the full continuation, recording the logprob margin log P(True) − log P(False) and the greedy verdict, with a per-problem unperturbed-trace baseline (report Δbelief). Uniform across polarity/usability/statement type; costs ~20 tokens per run. Secondary universal absorption measures: **echo rate** (planted string re-asserted later) and, for negations, **lie-consistency** (continuation never derives the fact the lie denies). Also: **unparsed must become a first-class outcome class** — the cert-flip's local-certificate arm has 0.317 unparsed (Table 4), so its surviving-run rates are computed on a 68% selected subset; a referee will find this.

---

## 2. The condition catalog

Naming follows the repo's existing conventions (`neghop2..5`, `EXPA/B/C`): new suites are **EXPD** (`expd_matched_gradient.py`) and **EXPE** (`expe_cert_distance.py`). All cells: Qwen2.5-7B primary, greedy, same interception protocol, same token limits, n=150 problems per cell per position (justified in §4), worlds freshly generated to spec (never re-runs locked artifacts; new namespaces `results_expd/`, `results_expe/`).

### 2.1 EXPD-CORE: polarity × distance, everything else pinned (attribute route)

World template (using the draft's own example vocabulary): chain `Stella is a lorpus → shumpus → wumpus → …`; the generator attaches exactly one governing attribute rule for nonce attribute F at category C_{front+d}, where "front" = last category derived before the injection point.

| Cell | Planted sentence | Rule placed in premises | Truth | d | Positions |
|---|---|---|---|---|---|
| `aff_false_d{1,2,3,5}` | "Stella is mean." | "Every ⟨C_{front+d}⟩ is not mean." | false, provable | 1/2/3/5 | early, mid (+late for d1,d3) |
| `neg_false_d{1,2,3,5}` | "Stella is not mean." | "Every ⟨C_{front+d}⟩ is mean." | false, provable | 1/2/3/5 | same |
| `neg_true_d{1,3}` | "Stella is not mean." | "Every ⟨C_{front+d}⟩ is not mean." | **true**, entailed | 1/3 | mid |
| `aff_true_d{1,3}` | "Stella is mean." | "Every ⟨C_{front+d}⟩ is mean." | **true**, entailed | 1/3 | mid |
| `aff_false_unent` | "Stella is mean." | no rule about "mean" on Stella's chain | unentailed (closed-world false) | ∞ | early, mid, late |

Held constant across all rows: statement type (attribute fact), subject (active entity), usability (inert — attributes have no outgoing rules), style (identical frame; ±1 token for "not", logged as covariate), on-path status of the claim itself. Varies: d; polarity; truth. Design gems to state in the paper: (i) `aff_false_dk` and `neg_true_dk` share the **same world** and differ by one token in the planted sentence; `neg_false_dk`/`aff_true_dk` share the mirrored world — a within-world truth×polarity 2×2 at each d. (ii) `neg_true` is the missing "negation without falsity" cell: if verbalized doubt is a negation detector (professor's alternative), it fires here; if doubt tracks refutable falsity, it stays at benign-paraphrase levels. (iii) `aff_false_unent` bridges to the locked global-falsehood family but with statement type and usability now matched to the gradient — the confounded 0.393-closure-valid/0.342-inj-dep cell (Table 1) gets a clean twin. Also regenerate `benign_paraphrase` and `true_interruption` on the new worlds (3 positions each) as the anchor controls.

### 2.2 EXPD-SAT satellites (mid position only, n=150 each)

| Cell | Construction | Axis closed |
|---|---|---|
| `cat_false_d{1,3}_inert` | Plant "Stella is a grimpus."; premise "Every ⟨C_{front+d}⟩ is not a grimpus."; grimpus has **no outgoing rules** | statement type (categorical vs attribute) at matched d, usability off |
| `cat_false_d{1,3}_usable` | Same + "Every grimpus is red." in premises | **usability × distance** cross; also makes strict inj-dep measurable in a *local* cell for the first time |
| `cat_false_unent_{inert,usable}` | "Stella is a grimpus.", no refuting rule; ± outgoing rule | usability at d=∞; twin of locked global family (which is usable — verify via retro-audit §5.1) |
| `aff_false_offpath_d3` / `neg_false_offpath_d3` | Refuting rule attaches to the **side chain** ("Stella is a dumpus → …") not the goal chain, at matched d=3 | evidence path-relevance vs distance (note honestly: for false claims, the *claim* can never be on the gold path; the crossable axis is which chain carries the refuting evidence) |
| `rule_false_d0` | Plant "Every shumpus is not a wumpus." where premises state "Every shumpus is a wumpus." | statement type = rule, locally contradicted |
| `rule_false_far` | Plant "Every ⟨C4⟩ is not a ⟨C5⟩." injected early (refuting instance derivable only ≥3 hops later) | rule × distance |
| `lex_decoy` | Premises contain "Max is not mean." (inert side entity); plant "Stella is mean." with no rule governing mean on Stella's chain (status = unentailed, identical to `aff_false_unent`) | **lexical contradiction/overlap**: doubt above the `aff_false_unent` baseline ⇒ surface predicate-polarity matching, not inference |
| `freq_high_d3` | `aff_false_d3` + 2 extra rules mentioning "mean" only as consequent on non-Stella chains ("Every jompus is not mean.") | **category/attribute frequency** decoupled from usability and from refutation paths; base cell = f=1 |

Plus always-on **measured covariates** logged per injection (all cells, and retro-fitted to locked artifacts): in-context frequency of planted predicate/category token, token distance to its nearest prior occurrence, # outgoing rules of the planted category (usability degree), planted-sentence token length, claim-subject on-goal-chain flag, position index.

### 2.3 EXPE: the evidence-mover (same-sentence designs; paired within problem; mid position)

Base: a fixed planted falsehood per problem, **byte-identical across all arms** (use `cat_false_unent_usable` as base to also connect to the cert-flip and the locked family).

- **EXPE-inf (`certd_inf{1,2,3}` + `certd_irrel{1,2,3}`)** — insert immediately before the planted step a true, subject-matched, frame-matched sentence "Stella is a ⟨C⟩." chosen so the refutation of the planted claim follows from it in j = 1, 2, or 3 hops via rules already in premises; irrelevant arms insert a true side-chain fact at the same offset that creates no refutation path. **Only j varies; the planted sentence, its polarity, type, usability, frequency, style, and path status are frozen.** n=150 paired problems × 6 arms + shared no-cert baseline.
- **EXPE-pos (`certd_pos{0,2,4,8,prem}` + matched irrelevant arms)** — insert the *directly refuting* fact (post-insertion d=0) at 0/2/4/8 sentences before the injection or in the premise block. Separates **positional/token distance** from **inferential distance** — the draft's title says "nearby evidence" without ever distinguishing these; this is the experiment that defines "nearby." n=150 × 10 arms.
- **EXPE must fix the cert-flip's holes:** (a) 0.317 unparsed in the certificate arm → report a 6-class multinomial outcome (closure-valid / inj-dep / parroted / derailed / **unparsed** / other) + best/worst-case imputation bounds + a doubt-judge pass over unparsed runs (they plausibly contain the doubt the parser chokes on); (b) the irrelevant certificate arm currently *beats* the local certificate on closure-valid (0.500 vs 0.467) and *raises* inj-dep over baseline (0.240 vs 0.153) — the re-analysis must model arms jointly rather than as Δs vs baseline, or a referee will do it for you.

### 2.4 Cross-model and transfer arms

- Core replication (all `aff/neg_false_d*` + `unent`, early+mid; and EXPE-inf) on **Qwen2.5-32B**; core-mid cells on **Llama-3.1-8B-Instruct**, **OLMo-2-7B-Instruct**, **DeepSeek-R1-Distill-Qwen-7B** (the doubt/recovery dissociation of Fig 2 predicts R1 shows the gradient in doubt but not closure-valid — test it on deconfounded cells).
- ICLR arms: **Qwen2.5-72B** core-6 (the paper's stated open question: does silent absorption shrink with scale, now on deconfounded cells); **verification-prompt × distance** interaction on `aff_false_d{1,5}` (does the existing "prompting doesn't fix far errors" claim survive deconfounding); **GSM8K evidence-mover transfer** — insert a true nearby check-fact vs an irrelevant fact before a corrupted intermediate value (n=150 × 3 arms; turns the finding into an actionable defense and escapes the toy-grammar objection).

---

## 3. What exists vs what must be written (file-level; verify names on skampere2)

**Exists (reuse):** vLLM greedy rollout + interception protocol; grammar parser & closure validator (do **not** touch closure logic — headline metric continuity); truth-status audit vs rule closure (`audit`-side); depth-targeted injection selection (neghop-k machinery behind Fig 1); paired-insertion machinery (`expb_local_cert_flip.py` — EXPE is a parameterization of it); matched-cohort pairing (`matched_pert.py`); strict replay validator (stored by `expc_polarity_control.py`, Table 6 — promote to optional flag suite-wide); lexical doubt detector + 32B doubt judge; problem-cluster bootstrap + fixed-effect logistic scaffolding (draft §3.1 already fits one); PLAN.md pre-registration convention; RESULTS_LOCK discipline.

**Write new:**
1. `exp/src/build_worlds_to_spec.py` — ontology generator emitting worlds with (i) attribute rules of chosen polarity attached at chosen depth relative to a planned injection front, (ii) negative category rules for `cat_*` cells, (iii) frequency-filler consequent rules, (iv) side-chain attachment for offpath cells; emits a per-world spec JSON (chain, attach depth, polarity, mirrored-pair id).
2. `exp/src/audit.py` extension — three-valued truth status {entailed-true, provably-false, unentailed} + `refutation_distance(claim, stated_facts, rules)` BFS + covariate emitter; **also runnable over locked JSONLs** (§5.1 retro-audit).
3. New family templates in `perturb.py` (`aff_false_dk`, `neg_false_dk`, `neg_true_dk`, `cat_*`, `rule_*`, `lex_decoy`, `freq_*`).
4. `exp/src/expd_matched_gradient.py`, `exp/src/expe_cert_distance.py` (clone of expb with offset + inferential-depth parameters).
5. `exp/src/probe_belief.py` — post-hoc probe pass over any results dir (planted-claim True/False logprob margin + unperturbed baseline).
6. `exp/analysis/mixed_models.py` (pymer4/lme4 or statsmodels GLMM; keep cluster bootstrap side-by-side for continuity) + `power_sim.py` + **`PLAN2.md` written and committed before the first run**, per house convention.

Engineering is the binding cost (~3–5 focused days); GPU is trivial (§6).

---

## 4. Statistical analysis plan (pre-register in PLAN2.md)

**Units & structure.** Greedy decoding ⇒ one continuation per problem × cell; variance lives across problems; worlds are shared across cells by design (mirrored pairs, EXPE arms), so problems are crossed random effects.

**Primary models (per DV — doubt-judge and Δbelief-probe are the two pre-registered primary DVs; lexical doubt and closure-valid secondary for continuity):**
`glmer(DV ~ polarity * log2(d) + position + claim_len + freq_cov + overlap_cov + (1|world) + (1|template), binomial)` over the false cells d∈{1,2,3,5}; d=∞/unent cells enter a companion model with categorical distance. EXPE: conditional logistic / GLMM with (1|problem), arms jointly. Multinomial model over the 5-class validator outcome including unparsed; sensitivity bounds by worst/best-case imputation of unparsed. Report problem-cluster bootstrap CIs alongside every GLMM estimate (continuity with Tables 1–5).

**Pre-registered confirmatory family (Holm-corrected, α=.05; everything else exploratory with BH-FDR flags):**
- P1: doubt-judge declines in log2(d) **within affirmative** false cells (slope < 0).
- P2: Δbelief-probe absorption increases in log2(d) within affirmative false cells.
- P3: polarity × distance interaction bounded — TOST equivalence on the (d1 − d5) gap difference between polarities, bounds ±10pp.
- P4: EXPE-inf doubt(j=1) > doubt(j=3) with the planted sentence byte-identical.
- P5: usability main effect on absorption at matched d (`cat_false_d1_usable` vs `inert`), testing whether "silent absorption" is partly "usable absorption."
- P6: doubt(`neg_true_d1`) < doubt(`neg_false_d1`) — doubt requires falsity, not negation.

**Decision rules.** *Local-refutability hypothesis CONFIRMED* iff P1 ∧ P2 ∧ P4 significant in the covariate-adjusted model and P3 equivalence holds. *Professor's confound account SUPPORTED* iff the affirmative gradient is flat (95% CI excluding OR ≥ 1.5 per doubling of d) while the negated gradient persists, or P6 fails (doubt fires on true negations at false-negation rates), or P4 is null with tight CI. *Usability moderation* (P5 large, distance effect shrinking under usability control) re-scopes the absorption claim rather than killing it. All three outcomes are publishable at TMLR; write PLAN2.md to say so.

**Power (target n=150/cell/position, pooling to 450).** Doubt contrasts are enormous in existing data (0.336 vs 0.013, Table 1) — trivially powered. Sizing is driven by mid-range absorption contrasts: two proportions 0.20 vs 0.35 at Holm-adjusted α=.005 (z=2.807), power .90 (z=1.282): h=2(asin√.35−asin√.20)=0.339, n=(4.089)²/h²≈146 ⇒ **n=150/cell/position** (exactly the existing convention); pooled 450 detects 10pp (0.25 vs 0.35, h=0.219, n≈349). TOST P3 at pooled n: SE of gap-difference ≈ √(4·0.24/450) ≈ 0.046 ⇒ 90% CI ±0.076 < ±0.10 bound. EXPE paired McNemar: ~35% discordant pairs at n=150 ⇒ ~52 discordant, ample. Verify all by simulation in `power_sim.py` (simr-style, using Table 3 rates as priors) and pre-register the simulated curves.

---

## 5. Zero-GPU and re-presentation work

### 5.1 Retro-audit of locked artifacts (do this first; 0 GPU-hours; never re-runs anything)
Run the extended audit over the **locked** per-run JSONLs: emit refutation distance, planted-token frequency, usability degree (# outgoing rules of the planted category — the draft's own example shows "Stella is an impus" consumed via "Every impus is not red"), and lexical-overlap covariates for every existing injection; then re-fit the draft's §3.1 logistic model with these covariates. If inj-dep in the locked global family tracks usability degree, that single free analysis both concedes and quantifies the professor's point and motivates the new cells — put it in the paper as the hinge between old and new results.

### 5.2 The decisive figures
- **Fig A (replaces current Fig 1) — "The gradient survives its confounds."** Two panels: verbalized doubt (judge) and Δbelief absorption. x = refutation distance {1, 2, 3, 5, ∞/unentailed}; solid = affirmative false, dashed = negated false (both new, everything else matched); grey = the legacy negated-only curve (locked Fig 1 data) and ghost points for the locked 1-hop/global cells; second row or dashed-color overlay = 32B. Claim in one glance: doubt falls and absorption rises with d **within each polarity**, slopes statistically indistinguishable (print the P3 equivalence bound on the plot).
- **Fig B — "Same sentence, moving evidence" (EXPE).** Left panel x = inferential distance j of the inserted true certificate {1,2,3, irrelevant, none}; right panel x = positional offset {0,2,4,8,premises} at j fixed; y = doubt and inj-dep with paired 95% CIs. Caption's first sentence: *"The planted sentence is byte-identical in every condition; only the surrounding evidence moves."* This is the "with force" version of the cert-flip and the strongest causal exhibit for an ICLR variant.
- **Fig C — cert-flip re-presented honestly.** Per-problem paired-flip plot (baseline → certificate arrows in doubt/inj-dep space; McNemar counts printed) plus a stacked 6-class outcome bar per arm **with unparsed visible**, replacing the current Δ-vs-baseline prose that hides the 0.317 unparsed and the irrelevant-certificate anomaly.
- **Table (main text) — the design matrix**: rows = all conditions (old + new), columns = the seven professor axes, entries ✓/✗/pinned. This table *is* the answer to weakness #1; make the reviewer read it before the results.

---

## 6. Models and honest compute (8×H200, vLLM, greedy; original full suite was ~8 GPU-hours per the repro section — this program is the same order)

| Suite | Runs | Gen tokens (≈400/run; R1 ≈3k) | Wall-clock (padded ×3 over pure-gen) |
|---|---|---|---|
| World gen + gold collection (~9k rollouts, CPU+GPU) | 9,000 | ~4M | ~1 h |
| EXPD core+satellites, 7B (+probe pass) | ~7,650 | ~3.1M | ~1 h |
| EXPE inf+pos, 7B | ~2,400 | ~1M | ~0.5 h |
| 32B core replication (TP=2 ×4 replicas) | ~3,600 | ~1.5M | 1.5–3 h |
| Llama-8B / OLMo-7B core-mid (each) | 1,800 | ~0.7M | ~0.5 h each |
| R1-Distill-7B core-mid (long CoT) | 1,800 | ~5.4M | 1.5–3 h |
| 32B doubt-judge pass over ~25k continuations | — | prefill-heavy | 1–2 h |
| 72B core-6 (TP=4 ×2) [ICLR] | 2,700 | ~1.1M | 2–4 h |
| Verification-prompt × distance; GSM8K evidence-mover [ICLR] | ~1,350 | ~0.9M | ~1 h |

**Total: REQUIRED set ≈ 6–10 node-hours; full ICLR program ≤ 1.5 node-days.** Per-condition-cell cost ≈ 150 runs × ~1.6k total tokens ≈ 240k tokens — compute is never the constraint; generator/audit engineering is. Optional sensitivity arm (limitations currently admit deterministic-decoding-only): 3 seeds at T=0.7 on the 8 core cells, +~1 h.

---

## 7. Ranking by evidential value per GPU-hour

| Rank | Experiment | GPU cost | Value | Tier |
|---|---|---|---|---|
| 1 | Retro-audit + covariate re-analysis of locked artifacts (§5.1) | ~0 | Quantifies every confound in the existing headline result without re-running anything | **REQUIRED (TMLR)** |
| 2 | EXPE-inf same-sentence inferential dose-response | ~0.5 h | Kills all sentence-property confounds in one design; the causal exhibit | **REQUIRED (TMLR)**; centerpiece for ICLR |
| 3 | EXPD-CORE polarity × distance (aff/neg × d1–d5 + unent bridge) | ~1 h | The powered 2×2(+parametric) that expc failed at n=84; directly answers weakness #1 | **REQUIRED (TMLR)** |
| 4 | Belief-probe DV + unparsed-as-class multinomial re-reporting (incl. cert-flip Fig C) | ~0.2 h | Fixes the measurement confound (†-cells) and the 0.317-unparsed selection hole | **REQUIRED (TMLR)** |
| 5 | `neg_true`/`aff_true` truth×polarity mirror cells | ~0.1 h | One-cell refutation of the "doubt = negation detector" alternative (P6) | **REQUIRED (TMLR)** |
| 6 | `cat_*` usability × distance | ~0.3 h | Separates "absorbed because usable" from "absorbed because unrefuted"; explains the locked global cell | **REQUIRED (TMLR)** |
| 7 | 32B + Llama/OLMo/R1 core replication | ~4 h | Upgrades the deconfounded gradient from a 7B fact to the paper's cross-model claim (Fig 2 continuity) | REQUIRED-lite (TMLR: 32B + one lineage); full grid ICLR |
| 8 | EXPE-pos (token distance vs hop distance) | ~0.3 h | Defines "nearby" — title-level clarification | Strongly recommended TMLR; **REQUIRED (ICLR)** |
| 9 | `rule_*`, `lex_decoy`, `freq_*` satellites | ~0.3 h | Closes statement-type, lexical-overlap, frequency axes (appendix table) | Recommended TMLR appendix |
| 10 | Verification-prompt × distance; GSM8K evidence-mover transfer | ~1 h | Actionable-defense narrative; escapes toy grammar | **ICLR unlock** |
| 11 | 72B scale point on deconfounded cells; 3-seed sensitivity | ~4 h | Answers the paper's own open question (does silent absorption shrink with scale) | **ICLR unlock** |

**Execution order:** PLAN2.md (pre-registered, committed) → retro-audit → generator/audit extensions + 200-world pilot per core cell (validate eligibility yields and probe calibration; pilot excluded from confirmatory stats) → freeze PLAN2 → full EXPD/EXPE on 7B → probe + judge passes → mixed models → cross-model arms → ICLR arms. Lock each new artifact set under the existing RESULTS_LOCK convention on completion; locked Tables 1–8 are never re-run — new results land as the deconfounded companion table + Figs A–C, with the retro-audit as the bridge paragraph.