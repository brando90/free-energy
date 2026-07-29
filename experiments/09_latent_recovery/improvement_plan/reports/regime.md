# DESIGN: Second Task Regime for Weakness #5 (Generality) — Latent Recovery / "Nearby Evidence" Paper

**Provenance note:** `PROFESSOR_FEEDBACK.md` and the project source tree do not exist on this machine (all task paths were literal `undefined`; prior subagents confirmed exhaustively). This design is grounded in the newest compiled draft `/Users/elyas/Desktop/main.pdf` (Jun 23, "Nearby Evidence Shapes How Transformers Respond to Planted Errors in Their Own Reasoning"; text extracted at `/private/tmp/claude-501/-Users-elyas-Desktop-Alara/ad633660-4a3e-4cbc-99dd-093a0feae95e/scratchpad/main.txt`) plus the four analyst reports. Verified draft facts load-bearing for this design: metric vocabulary = {closure-valid completion, injection-dependent, parroted, derailed, unparsed, doubt}; Table 7 = PrOntoQA 0.360 closure-valid / 0.373 inj-dep (n=150), GSM8K 0.088 / 0.877 (n=57), chained arithmetic 0.020 / 0.990 (n=99); §3.4 already conjectures **"Models appear more likely to re-check information they can re-read than information they must recompute"**; Limitations currently promises **"A formal-verification task such as Lean is the natural next regime."** That promise must be edited if the recommendation below is adopted.

---

## 1. Candidate regime evaluation

Criteria: (a) machine-auditable truth status of planted step; (b) formal validation of continuations; (c) clean refutation-distance / local-counterevidence manipulation; (d) implementation cost given existing vLLM + injection + validator + stats pipeline; (e) reviewer impressiveness.

| Regime | (a) Truth audit | (b) Continuation validation | (c) Distance manipulation | (d) Cost | (e) Impressiveness | Verdict |
|---|---|---|---|---|---|---|
| **Lean / miniF2F proof traces** | Partial: kernel checks proofs, it does not *decide* truth of an arbitrary planted `have` proposition. Auditing "false vs. true-but-off-path" (the paper's own construct trap) requires proving P or ¬P — only decidable for `decide`/`norm_num`-able numeric propositions | Gold standard (Lean kernel) — best-in-class on this axis alone | Poor: no natural parametric "k hops to refuting evidence"; goal-state context ≠ fact list; proofs the cohort would admit are 1–3 tactics, killing early/mid/late positions | Very high: Lean4+mathlib+REPL/LeanDojo, per-step kernel checking with sorry-stubs, sentence↔tactic alignment. **Fatal: cohort collapse** — Qwen2.5-7B-Instruct greedy pass@1 on miniF2F ≈ 0–3%; the protocol requires the model's *own* correct trace, frozen model, greedy decoding. A specialized prover (DeepSeek-Prover, Kimina) breaks model comparability and relies on sampling/search, violating the frozen-greedy protocol | Highest prestige | **Reject for this cycle.** Convert the vague Limitations promise into a concrete future-work sketch (decidable-proposition intermediates audited by `norm_num`, prover-specialized model, whole-proof `have`-chain injection) |
| Informal theorem-proving traces (NL math, NaturalProofs-style) | No | **No** — validation would be LLM-judge, reintroducing exactly the black-box-validator criticism | Undefined | Low | Low | Reject |
| Richer synthetic logic (FOL, ProofWriter, disjointness grammar) | Yes (trivially) | Yes | Yes | Low | **Worst** — it *is* another synthetic ontology grammar; answers nothing about weakness #5. Belongs to the EXPC polarity-cell fix (grammar extension), a different weakness | Reject as regime; keep as EXPC follow-up |
| **Program-execution traces (Python, interpreter-validated)** | **Exact**: CPython gives ground truth for every claimed variable value; planted-claim truth is decided by execution, not by the authors' logic | **Exact, external, two-world**: reference interpreter σ_true + counterfactual interpreter σ_cf (state forced to planted value) classify every post-injection claim. Crucially this makes injection-dependence mechanically measurable for **all** families — removing the paper's own † asymmetry (poisoning measurable only for positive-category injections in PrOntoQA) | **Clean and decomposable**: refutation cost splits into *re-read distance* (how far back the refuting operand values were stated) and *recompute depth* (how many unstated evaluations falsification requires) — a factorial that directly operationalizes the draft's §3.4 re-read-vs-recompute conjecture, which PrOntoQA cannot separate | **Low–moderate**: strict generalization of the existing (locked) chained-arithmetic module; reuses rollout harness, injection-at-boundary machinery, cohort double-filter, doubt lexicon + 32B judge, cluster-bootstrap/Holm stats, artifact-lock conventions. New code ≈ generator + interpreter + parser (~1–1.5k lines) | Strong for TMLR: ground truth is a real interpreter (not author-defined semantics), stateful imperative semantics vs. monotone Horn logic, canonical lineage (Nye et al. 2021 scratchpad tracing; CRUXEval), direct relevance to agentic code execution. Add an externally-authored CRUXEval-derived arm and the "authors generated the distribution" objection dies too | **RECOMMEND — PRIMARY** |
| Controlled symbolic math + CAS (SymPy) | Yes within a restricted grammar (polynomial/linear identities decidable via canonical forms; avoid ops needing zero-testing) | Yes with care (equivalence vs. one-way implication under squaring/division; restrict to invertible ops) | **Weak** — every algebraic step is refutable from the immediately preceding line alone; distance ≈ 1 everywhere; parametric manipulation is contrived | Moderate; the real tax is robustly parsing model-emitted math (LaTeX/ASCII variance) — worse than a rigid trace format | Moderate (third symbol system: equational rewriting) | **Optional cheap secondary**, pilot scale only, contributing to "gradient exists in a third semantics" but not to the parametric curve |

**Recommendation: PRIMARY = interpreter-validated program-execution traces** (controlled generator + a CRUXEval-derived external-provenance arm). **Optional secondary = SymPy-validated simplification chains** at n=35–50/cell, 3 families, mid position only — cut it without regret if time-boxed. **Explicit reasoning for primary:** it is the only candidate that scores exact on (a) and (b), *improves* on the PrOntoQA measurement design ((†-fix, polarity and usability held constant by construction, no goal anchor in prompt so the parroting confound is structurally absent), gives the paper's central independent variable a cleaner parametric handle than PrOntoQA itself, is feasible for the frozen Qwen2.5-7B-Instruct greedy protocol (difficulty is a generator knob), and turns Table 7's three incommensurate tasks into one structured regime map. Lean loses on the binding constraint (cohort collapse under the frozen-greedy protocol) despite winning prestige; TMLR-first favors the airtight option, and the Lean prestige play is preserved as a concrete future-work paragraph (or a later ICLR arm with a prover model, framed as a protocol variant).

---

## 2. Full experimental design: the execution-trace regime

### 2.1 Task and trace generation
- **Programs:** randomly generated straight-line Python with bounded loops. Grammar knobs: length L ∈ [12, 20] executable steps; ops {+, −, ×} (no division — parse/float hazards); integer values, all pairwise-distinct within a run, |v| ≤ 999 (≤3 digits, tokenizer-uniform across model families); loops of 3–5 iterations; variable names from a fixed pool disjoint from Python builtins; final `output: <value>`.
- **Controlled dataflow (the design's core):** the generator controls (i) *operand staleness* — how many trace lines back each operand of a given line was last written/stated; (ii) *expression depth* — how many primitive evaluations a line's result requires, with intermediates never stated; (iii) *forward-use gap* — how many lines until a variable is next read; (iv) *redundancy* — density of re-reads/asserts/redundant restatements of a variable between its corruption and its final use.
- **Generator guarantees (rejection sampling):** planted counterfactual diverges (σ_cf ≠ σ_true on ≥2 downstream reads — guarantees discriminating claims); no value collisions between σ_true and σ_cf on discriminating lines; planted value never becomes accidentally true later; **forward-use gap held constant (2 lines) across all distance conditions** — a causal-isolation improvement over PrOntoQA's neghop-k, where evidence distance co-varies with use distance.
- **External-provenance arm:** ~100–150 loop-light functions filtered from CRUXEval (externally authored, executable), traced under the same format; 1-hop and global cells only. This is the direct answer to "artifact of a grammar the authors wrote."
- **Prompting:** 3-shot canonical trace format ("Line 4: x = a + b → x = 7"), numbered program listing in prompt (the true world is fully visible in-context, mirroring PrOntoQA's visible premises), greedy decoding, frozen Qwen2.5-7B-Instruct, fixed budget, no feedback.

### 2.2 Cohort filter (mirrors the paper's double filter)
Keep instances where (1) the unperturbed greedy trace exactly matches the interpreter on every claim and the final output, and (2) the trace parses. Tune grammar until greedy solve ≥ 75–85% and parse ≥ 95% (Stage 0 pilot); report cohort rates; retain the "rates are upper bounds on the solved cohort" language.

### 2.3 Injection procedure
Identical to PrOntoQA: parse the model's own rollout at line boundaries, overwrite the stated result of the step at early (~3), mid (~8), late (~14) positions with the planted sentence, resume greedy generation, same token budget, no self-correct prompt. Discourse-marker stripping reused.

### 2.4 Perturbation families — what transfers

| PrOntoQA family | Execution-trace analogue | Transfers? |
|---|---|---|
| (a) Benign paraphrase | Correct result restated in altered style ("So now v holds 7.") | Yes — isolates style/interruption cost |
| (b) True interruption (off-path entailed) | True but non-canonical state fact ("note y is still 4"; "a + b equals 11") | Yes |
| (c) Distractor | Claim about a dead variable / uncalled helper, off every dataflow path to output | Yes |
| (d) Contradiction, distance 0 | Wrong restatement of a value stated on the immediately preceding line (adjacent conflict, zero inference) | Yes — exactly |
| (e) 1-hop falsehood | Wrong result of the current line, operands stated adjacently, single op (one recomputation refutes) | Yes — the core injection |
| (f) Globally-checkable falsehood | Wrong loop-accumulator/deep-expression summary claim: no stated intermediates, no adjacent conflicting line; falsification requires re-executing ≥4 steps | Yes |
| neghop-k dose-response | **Two orthogonal parametric knobs** — re-read distance k_r ∈ {1,2,4,8} (operand staleness) and recompute depth k_c ∈ {1,2,3,5} (unstated evaluations); run both axes + 3 crossing cells (pure-retrieval vs. pure-compute corners) | Yes, upgraded: decomposes the gradient |
| EXPB certificate flip | Insert true operand restatement ("recall a = 3 and b = 4") immediately before the planted wrong result vs. length-matched irrelevant true statement ("recall q = 2"); strictly paired triples | Yes — exactly |
| Polarity families (EXPC) | **Does not transfer — by design a gain**: every planted claim is an affirmative value-equality, so surface form, statement type, and derivational usability are *constant* across the whole gradient. The polarity confound that EXPC failed to break in PrOntoQA does not exist here. Say this explicitly in the paper |
| QA-format / anchor-knockout controls | **Not needed**: no target statement appears in the prompt, so goal-parroting is structurally absent | Confound removed by construction |

### 2.5 Validator (exact tool) and metrics mapping
**Tool: CPython.** Reference interpreter replays the program (σ_true per line); counterfactual interpreter forces the injected variable to the planted value at the injection point and executes forward (σ_cf). Each parsed post-injection claim "u = d" is classified: true-match (d = σ_true(u)), cf-match (d = σ_cf(u)), both (non-discriminating; excluded from the discriminating set), neither.

| Paper metric | Execution-trace metric |
|---|---|
| Closure-valid completion | **Trace-valid completion**: every post-injection claim true-matches and final output correct. (Note in paper: this validator is *stricter* than closure entailment — per-step exact, no author-defined closure. Step-skipping measured separately, mirroring the goal-jump accounting.) |
| Injection-dependent | ≥1 discriminating claim cf-matches (sub-metrics: next-read absorption = first discriminating read cf-matches, the exact analogue of arithmetic "propagation"; final-answer absorption = output equals σ_cf output). **Measurable for every family** — fixes the † footnote asymmetry |
| Parroted | Final output correct with ≥1 discriminating claim not true-matching ("lucky/inconsistent completion"). Expected ≈ 0 absent a goal anchor — itself an informative contrast confirming parroting is goal-anchor-driven |
| Derailed | Final output wrong and not cf-matching |
| Unparsed | Same convention |
| Doubt | Same lexical detector with a code-domain extension of the lexicon ("bug", "mistake", "should be", "that's wrong") **frozen pre-registration**, plus the existing Qwen2.5-32B judge with domain-adapted prompt |
| (New, free) Repair event | Post-injection claim about the planted variable that true-matches — an *observable overwrite*, crossed with doubt to give silent-repair vs. flagged-repair directly |

### 2.6 Statistical plan
Mirror the paper: n = 150/cell (matches expanded audited sweep); problem-cluster bootstrap CIs; Holm across positions; pre-registered `PLAN_PROGTRACE.md` written before the first full run (the repo's existing convention — TMLR-friendly). Primary contrasts (pre-registered): doubt(k_r=1) > doubt(k_r=8) and trace-valid(k_r=1) > trace-valid(k_r=8) at mid position; k_c contrasts secondary; k_r×k_c interaction and redundancy moderation exploratory. Power: n=150 gives CI half-widths ≈ ±0.08; detectable Δ ≈ 0.12 under Holm.

---

## 3. Replication criteria, failure, partial replication

**Counts as replication (directional/ordering claims, not point values — pre-register signs):**
1. Benign paraphrase ≈ true interruption ≈ unperturbed ceiling (control sanity).
2. Adjacent/1-hop falsehood cells: elevated doubt (judge-validated), trace-valid moderately reduced, low absorption — the "loudly flagged" arm.
3. Global/deep cells: doubt ≈ benign baseline or below, trace-valid strongly reduced, injection-dependence high (≥ 0.3) — the "silently absorbed" arm.
4. Monotone-then-plateau decline of doubt and trace-valid in k_r (and/or k_c) — the dose-response signature of Fig. 1.
5. Operand-certificate raises doubt vs. irrelevant certificate (EXPB signature, Δdoubt ≥ ~0.15).

**Counts as failure:** flat or inverted distance curves at full power, or a floor (absorption ≥ 0.95 in every cell including k_r=1 with high redundancy). A floor is not paper-killing: it places program traces in the arithmetic regime and *sharpens* the claim's scope — "the gradient operates in tasks whose checks are re-reads of visible declarative facts; recomputation-checked tasks absorb regardless" — which Table 7 + §3.4 already gesture at. The paper handles it by scoping the headline, not retracting it.

**Partial replication — pre-commit this decision tree in PLAN_PROGTRACE.md:**
- Gradient in absorption but not doubt (or vice versa) → split the claim by channel using the existing doubt/repair dissociation frame (Fig. 2 / Llama already establishes the channels separate); headline follows the channel that generalizes.
- k_r gradient without k_c gradient → the most theoretically interesting outcome: "local falsifiability" refines to **re-readable counterevidence availability**, confirming §3.4's conjecture causally; rename the construct accordingly ("evidence-availability gradient").
- k_c gradient without k_r → verification-compute story; same move, opposite direction.
- Gradient only under high redundancy → claim becomes "local falsifiability governs the response wherever task structure affords repair" — a refinement folding cleanly into the existing Table 7 narrative.
In all cases: report every cell, keep the pre-registered primary contrast honest, and let the abstract state the scope the data supports. The paper has survived one headline inversion by auditing itself; the same posture is the correct handling here.

---

## 4. Implementation plan

**New modules (`exp/src/progtrace/`):**
- `gen_programs.py` — grammar + knobs (L, k_r, k_c, forward-use gap, redundancy, loop config) + rejection-sampling guarantees; emits program, injection sites, family metadata.
- `interp.py` — reference + counterfactual executors; per-line σ_true/σ_cf; discriminating-line marking.
- `trace_format.py` — canonical trace prompt (3-shot) + strict parser (reuse discourse-marker stripping; unparsed accounting).
- `inject_prog.py` — thin adapter over the existing sentence-boundary injection machinery; families (a)–(f), k_r/k_c sweeps, certificate triples.
- `validate_prog.py` — claim classification, 5-way run outcome, repair events; **strict-replay artifact** (reuse the EXPC replay-validator pattern) so the validator is auditable — this doubles as evidence against the validator-black-box weakness.
- `cruxeval_arm.py` — filter/format externally-authored functions into the same pipeline.

**Reused as-is:** vLLM rollout runner, cohort double-filter, doubt lexicon + `doubt_judge` (32B) prompt scaffolding, cluster-bootstrap + Holm stats, matched-pairing logic (`matched_pert.py` conventions), results-JSON → paper-table regeneration pipeline, RESULTS_LOCK/artifact-lock conventions. **Locked results (arith n=99, GSM8K n=57, all PrOntoQA tables) are not re-run.**

**N and compute (8×H200, Qwen2.5-7B primary):** gold collection 2,500 programs (→ ~1,700–2,000 cohort); core grid 6 families × 3 positions × 150 = 2,700; dose-response (4 k_r + 4 k_c + 3 crossing cells) × 150 × 2 positions ≈ 3,300; certificate triples 3 × 300 = 900; redundancy moderator 2 × 150 = 300; CRUXEval arm ≈ 400. Total ≈ 10,000 rollouts × ~1k tokens ≈ 10M tokens — under 2 H200-hours at vLLM batch throughput. 32B dose-response replication (+ its own gold cohort) ≈ 3,200 rollouts ≈ 4–6 H200-hours (TP=4). 32B judge pass over ~7,600 continuations ≈ 1 hour. **End-to-end < 25 H200-hours; wall-clock one weekend.** Human time dominates: ~1.5–2 weeks.

**Staged path:**
- **Stage 0 (days 1–2):** generator + interpreter + parser; 100-program solve-rate pilot; tune grammar to greedy solve ≥ 75% / parse ≥ 95%; 25 hand-audited validator classifications (publishable spot-check table).
- **Stage 1 pilot (days 3–4):** mid position, {benign, d0-contradiction, 1-hop, global-loop}, n=35/cell; direction check; freeze doubt lexicon; write and commit `PLAN_PROGTRACE.md` **before** Stage 2.
- **Stage 2 (week 2):** full 7B grid + both dose-response axes + certificates + CRUXEval arm; artifact-lock; regenerate paper tables from summary artifacts.
- **Stage 3 (optional; feeds the ICLR variant):** 32B + Llama-3.1-8B + OLMo-2 dose-response (cross-lineage generality of the gradient), full k_r×k_c factorial, redundancy interaction. Skip latent caching by default (latent metrics are disclaimed negatives in the current draft).

**Key risks / mitigations:** low solve rate → grammar knob, report final params; "absorption is faithful-to-context, arguably correct" objection → pre-empt in text (identical tension exists in PrOntoQA and *is the phenomenon*; program listing in-context = true world fully visible); both-match saturation → generator guarantee of ≥2 discriminating reads; value collisions → distinct-values constraint; contamination → none for random programs, and the CRUXEval arm is perturbation-novel even if functions are seen.

---

## 5. GSM8K / chained arithmetic disposition

- **Chained arithmetic: promote by absorption, do not re-run.** It is the degenerate cell of the new regime (a straight-line program with zero redundancy, k_r=1 mandatory-use dataflow). Keep the locked numbers (0.020 / 0.990, n=99) as the pre-registered anchor; retire it as a standalone "brittle task" section and re-present Table 7 as a **regime map** with the new program-trace cells populating the axis between PrOntoQA (redundant, re-readable checks) and arithmetic (no redundancy, recompute-only checks). Three incommensurate tasks become one structured comparison — this is the single biggest framing upgrade the new regime buys.
- **GSM8K: keep, demote, do not promote.** It cannot be the audited second regime: truth status of an arbitrary planted natural-language step is not machine-auditable (the paper's own off-path≠false lesson — promoting GSM8K re-opens the audit attack it survived), continuations have no formal per-step validator (its absorption is propagation-based, n=57). Reframe as *naturalistic corroboration*: its 0.088 / 0.877 pattern is the observational data point that motivated the re-read-vs-recompute conjecture (§3.4), which the program-trace k_r×k_c factorial then tests causally. Add one caveat sentence to the Table 7 caption distinguishing audited (PrOntoQA, program traces, arithmetic) from non-audited (GSM8K) rows.
- **Required text edits when this lands:** (1) Limitations "two synthetic families plus GSM8K" → "three formally validated regimes (one with external interpreter ground truth and an externally-authored program arm) plus GSM8K corroboration"; (2) replace "A formal-verification task such as Lean is the natural next regime" with the concrete Lean future-work sketch from §1 (decidable-proposition intermediates, prover-model protocol variant) — do not leave a promise the paper doesn't keep; (3) abstract gains one clause: the falsifiability/nearby-evidence gradient replicates in an interpreter-validated execution regime, and "local falsifiability" decomposes into evidence re-readability vs. recomputation cost (whichever the data supports per §3's decision tree).