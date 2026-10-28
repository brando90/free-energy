# REPORT: Perturbation machinery — latent recovery project

## 0. INFRASTRUCTURE FAILURE (read first)

The project path in the task prompt was the literal string `undefined` (variable substitution failed upstream). An exhaustive search of this machine (`find` over `/Users/elyas` excluding Library/caches, plus Spotlight) found **no copy of the project source**: `perturb.py`, `build_dataset.py`, `matched_pert.py`, `expa_global_expansion.py`, `expb_local_cert_flip.py`, `expc_polarity_control.py`, `PROFESSOR_FEEDBACK.md`, and the `exp/` and `paper/` LaTeX trees **do not exist anywhere on this machine**. What DOES exist locally:

- `/Users/elyas/Desktop/main.pdf` (Jun 23, 11 pp — newest compiled paper, "Nearby Evidence Shapes How Transformers Respond to Planted Errors in Their Own Reasoning")
- `/Users/elyas/Desktop/latent_recovery_paper_latest.pdf`, `latent_recovery_paper_v4.pdf`, `/Users/elyas/Desktop/latent_recovery_old_versions/` (v2, v3, workshop)
- `/Users/elyas/Desktop/Literature Briefing on Latent Recovery in Autoregressive Transformers.pdf`

Everything below is reverse-engineered from the compiled paper (`main.pdf`), **not** from source code. Items answerable only from code are explicitly flagged `[CODE-ONLY]`. Re-dispatch with the correct project path to fill those gaps.

---

## 1. Perturbation families — construction as documented

**Protocol frame (all families):** model first produces a *correct, validator-passing* proof on PrOntoQA-OOD (fictional ontologies, e.g. "every wumpus is a tumpus"; rigid grammar → per-sentence formal parse). One intermediate step at a controlled position (early / middle / late) is **overwritten** with the planted sentence; generation resumes with same token limits, no feedback, no self-correct prompt. Continuation is formally validated (closure-valid / injection-dependent / parroted / derailed / unparsed). Every planted atom is audited against the rule closure so off-path-but-entailed facts aren't miscounted as errors. Latent comparisons additionally sample 8 temperature-0.8 continuations from the same interception point with the correct step intact; hidden states cached at 5 depths.

Families, ordered by ease of checking against surrounding proof:

| Family | Construction | Truth status | Polarity | Injection position |
|---|---|---|---|---|
| (a) **Benign paraphrase** | The correct gold step, reworded (e.g. "Thus, Stella is not opaque.") | TRUE | matches gold step | early/mid/late |
| (b) **True interruption** (off-path category) | Fact absent from gold proof path but logically entailed by world premises | TRUE | affirmative | early/mid/late |
| (c) **Distractor rule** | Fabricated rule about an entity entirely irrelevant to the active entity | unentailed/irrelevant (not refutable) | affirmative rule | early/mid/late (n=130, older auxiliary sweep) |
| (d) **Contradiction (dist. 0)** | Negation of a fact **already stated** in the proof — lexical conflict with adjacent context at distance zero | FALSE | negated | early/mid/late (n=130) |
| (e) **Negated step (negstep ≈ k=1)** | Negation of the step **about to be derived**; conflicts with no prior statement but refutable via single-hop inference | FALSE | negated | early/mid/late |
| **1-hop falsehood** (expanded audited family, n=150/pos) | False claim about active entity refutable in one hop; examples all negations: "Stella is not mean.", "Stella is not a dumpus." | FALSE | **negated** | early/mid/late |
| (f) **Globally-checkable falsehood** | Categorical claim strictly unentailed, with **no nearby conflicting evidence**; detectable only by searching entire rule closure. Example: "Stella is an impus." Rarer under constrained generator → `expa_global_expansion.py` expands to 150/position and pairs with matched controls | FALSE | **affirmative categorical** | early/mid/late |
| **neghop2–5** (dose-response) | Negate the fact the model will derive exactly *k* hops downstream → contradicting evidence placed exactly k inferences away; k∈{1..5}, plus k=∞ anchor = globally-checkable off-path lie | FALSE | negated (k=1–5); affirmative at k=∞ | early (per Fig. 1) |
| **Local certificate** (expb, not a falsehood family) | TRUE statement directly contradicting the planted global claim, inserted immediately before it | TRUE | (form) [CODE-ONLY] | before planted step |
| **Irrelevant certificate** (expb control) | Length-matched TRUE statement that does not falsify the claim | TRUE | — | before planted step |
| **expc cells** | local-positive / local-negative / global-positive / global-negative categorical claims; strict local-positive categorical cell infeasible in grammar — generated local-positive uses affirmative **attributes** instead of categories | FALSE | 2×2 by design | matched cohort, 7 problems / 84 rows |

`[CODE-ONLY]`: exact sentence templates, entity/attribute vocab selection, how "early/middle/late" indices are computed, tie-breaking when multiple candidate steps exist, and whether negations use "is not a X" vs "is not X" surface forms.

## 2. Confound matrix (professor's axes)

Legend: ✓ = present, ✗ = absent, ~ = partial/ambiguous, ? = [CODE-ONLY].

| Family | False | Negation present | Lexical contradiction w/ context | Statement type | On-path | Derivationally usable toward goal | Stylistic change |
|---|---|---|---|---|---|---|---|
| benign paraphrase | ✗ | matches gold (can be neg.) | ✗ | fact (categorical/attr) | ✓ | ✓ | **✓** (only family that is pure style) |
| true interruption | ✗ | ✗ | ✗ | fact, categorical | ✗ | ~ (entailed, but off gold path) | ✗ |
| distractor rule | ~ (unentailed, unrefutable) | ✗ | ✗ | **rule** (only rule family) | ✗ | ✗ | ✗ |
| contradiction (d=0) | ✓ | **✓** | **✓** (token-level clash w/ prior line) | fact | ✓ (active entity) | ✗ (grammar derives nothing from a negation) | ✗ |
| negstep / 1-hop falsehood | ✓ | **✓** | ✗ | fact, negated categorical | ✓ | ✗ (negation → underivable) | ✗ |
| neghop2–5 | ✓ | **✓** | ✗ | fact, negated categorical | ✓ (future derived fact) | ✗ | ✗ |
| global falsehood | ✓ | **✗ (affirmative)** | ✗ | fact, **positive categorical** | **✗ (off-path)** | **✓** (feeds later derivation steps — hence high inj.-dep. 0.233–0.420) | ✗ |

**Axes currently confounded with inferential distance to refuting evidence (the paper's independent variable):**

1. **Negation/polarity — fully confounded.** Every locally-refutable falsehood (d=0 contradiction, negstep, 1-hop, neghop2–5) is a *negated* sentence; the only distance-∞ falsehood is *affirmative*. The lexical-polarity control (Table 5) tried to break this and failed (local-positive categorical infeasible; n=84 underpowered).
2. **Derivational usability — fully confounded, and measurement-confounded.** Affirmative false claims can be consumed by forward-chaining (usable → absorbed → inj.-dep.); negations derive nothing in the grammar. Worse, the validator itself can only mechanically detect strict poisoning for *positive* category claims (Table 3 footnote †) — so the dependent variable is partly unmeasurable exactly in the local/negated families.
3. **On/off-path relevance — confounded at the distance extremes.** k=1..5 negations sit on the derivation path; the k=∞ anchor is off-path. So "far from refuting evidence" co-varies with "off the gold path."
4. **Statement type — partially confounded.** All falsehoods are facts; the sole rule perturbation (distractor) is simultaneously irrelevant, off-path, and unrefutable, so "rule-ness" is never crossed with falsity or locality. Within facts, local families are negated-category, global is positive-category.
5. **Lexical contradiction — confounded at d=0 only.** The d=0 contradiction is the only cell with token-level clash; it is also the most locally-refutable, so surface string-matching vs. 1-hop logical inference are not separated between d=0 and d=1.
6. **Stylistic change** — NOT confounded with distance (benign paraphrase isolates it), and §3.6 shows the latent JS-divergence metric registers style as strongly as lies — a stated negative result, not a design hole.

## 3. What each experiment varies / holds constant

**Distance-k (dose-response, Fig. 1 / expa territory).** *Varies:* k = number of inference hops between planted claim and its refuting evidence (1–5, plus ∞ = global-only off-path lie), and model scale (7B vs 32B). *Holds constant:* the planted sentence is always a negation of a fact the model would itself derive; same active entity; same injection position (early); same world/proof per k presumably via the same problem instance `[CODE-ONLY: whether the k-conditions are within-problem paired]`. *Does NOT hold constant:* polarity, usability, and on-path status at the k=∞ endpoint (see §2). Result: doubt and closure-valid decline with k then plateau at k≥3; 32B shifts curve up without removing distance dependence.

**expa_global_expansion.py.** Expands the generation-constrained globally-checkable-falsehood family to 150 examples per injection position and **matches** each with true-interruption, benign-paraphrase, and 1-hop-falsehood controls (Table 1: n=450 pooled per condition). So the expanded primary sweep is a 4-condition matched design over the same problem set. `[CODE-ONLY: matching keys]`.

**expb_local_cert_flip.py (Local-Certificate Flip).** *Holds constant:* the planted step itself — always the same globally-checkable falsehood — plus injection position; strictly **paired triples** at each position, 183 problem clusters, 300 continuations per condition (Table 6: 900 total). *Varies only:* what is inserted immediately before the falsehood — (i) nothing (global baseline), (ii) a true local certificate that directly contradicts the planted claim, (iii) an **irrelevant certificate**, length-matched and true but non-falsifying (controls for mere interruption/added text). Result: certificate → inj.-dep. −0.127, doubt +0.370, closure-valid only +0.090 with elevated unparsed (0.317) — certificates buy flagging, not repair.

**expc_polarity_control.py (Lexical-Polarity Control).** Attempted 2×2: locality (refuting evidence local vs global-only) × polarity (affirmative vs negated surface form), on a **fully matched cohort** (same 7 problems, 84 validated rows across 4 cells: local-positive, local-negative, global-positive, global-negative). *Holds constant:* problem, entity, position. *Boundary condition:* the rigid ontology grammar cannot produce a **local, affirmative, false categorical** claim without changing the category rules — that cell was filled with affirmative *attribute* claims instead, so the design is reported as a boundary-check, not a decisive control. It also carries the only strict-replay (gold-suffix) validator (Table 6). Doubt aligns with locality not polarity; closure-valid too noisy.

**matched_pert.py `[CODE-ONLY]`.** From paper evidence it must implement at least: same-problem pairing across the 4 expanded families, same injection position, per-injection-position example counts (150), and the length-matching used for the irrelevant certificate; plus the perturbed-vs-gold latent comparison pairing (8 temp-0.8 gold continuations from the identical interception point). Whether it also matches token length, entity, sentence frame, or parse depth of the planted step is **unverifiable without source**.

## 4. Minimal NEW conditions to complete the confound matrix

Goal: fully cross **local-refutability {1-hop, global-only}** × **polarity {affirmative, negated}** × **statement type {fact-category, fact-attribute, rule}** × **path status {on, off}** × **usability {usable, inert}** where the grammar permits. Ranked minimal set (7 conditions; the first four are the load-bearing ones):

1. **Local affirmative false categorical (the expc missing cell), at scale.** Extend the generator grammar with disjoint-category rules ("every wumpus is not a numpus" / "wumpuses and numpuses are distinct") so an affirmative claim ("Stella is a numpus") is refutable in one hop. This is the single cell blocking the polarity×locality cross; needs n≈150/position, not 21. Bonus: affirmative → strict poisoning becomes mechanically measurable in the local cell, fixing the measurement confound.
2. **Global-only negated falsehood, powered.** A negation ("Stella is not an impus") whose refutation requires full closure search (no derivation chain touching impus near the injection). Exists embryonically as expc "global negative" (n=21); scale to match. Completes polarity×locality with (1).
3. **Usability-inert global affirmative falsehood.** An affirmative false categorical about the active entity that *cannot* feed any rule toward the goal (no rule has its category as antecedent). Separates "absorbed because usable" from "absorbed because not locally refuted" — currently perfectly confounded in the global family.
4. **Affirmative distance-k sweep.** Re-run neghop2–5 with disjoint-category affirmative lies at matched k, so the dose-response in Fig. 1 is not a dose-response in "negation nearby." If the curve replicates, the headline claim survives its worst confound.
5. **On-path false rule (local and global variants).** Corrupt a rule the proof actually uses ("every wumpus is a jompus" → "every wumpus is a vumpus"), with the refuting evidence 1 hop vs closure-only away. Crosses statement-type with locality; currently rules exist only as the irrelevant distractor.
6. **True negated interruption.** A TRUE, entailed, *negated* off-path fact ("Stella is not liquid" where true). Unconfounds negation-presence from falsity — currently every negated planted sentence is false, so any negation-triggered doubt heuristic mimics falsity detection. (Cheap: the audit machinery already verifies entailment of negations.)
7. **Lexical-clash-without-logical-conflict control.** A true statement sharing predicate vocabulary with a nearby line but consistent with it (e.g. restating a prior fact about a *different* entity with negation). Separates surface string-contradiction matching (d=0 contradiction family) from genuine 1-hop inference.

Priority justification: (1)+(2) complete the 2×2 the professor's polarity axis demands; (3)+(4) de-confound the paper's two headline results (global-absorption and the distance dose-response); (5)–(7) close the remaining statement-type and lexical axes.

---
**Files consulted:** `/Users/elyas/Desktop/main.pdf` (primary), `/Users/elyas/Desktop/latent_recovery_paper_latest.pdf`, `/Users/elyas/Desktop/latent_recovery_paper_v4.pdf`. **Action needed from orchestrator:** re-run this task with the real project root substituted for `undefined` to verify §1 templates, matched_pert.py matching keys, and all `[CODE-ONLY]` items.