All required reading and code verification complete. Composing the design.

# SECOND TASK REGIME DESIGN (Weakness #5: Generality) — v2, superseding `reports/regime.md`

**Status of this document.** This is the revised second-regime design. It keeps the core recommendation of `reports/regime.md` (Plan B) — which `reports/feasibility.md` §3 verified as sound — and repairs every defect the feasibility audit found (no vLLM in repo; compute 3–5x optimistic; value-bound underspecification; overlooked `gsm8k_gradient.py`) and every attack in `reports/skeptic.md` (W5 attacks 1–2 and residuals, N1, N3, N6, N10, N11, N12). It is aligned with `reports/storyPlan.md` (regime = ICLR criterion #4 instrument; main-text slot = §5.3 task-structure subsection + regime map) and `reports/validatorPlan.md` (defs-box vocabulary; hop-sound metric M2 is the matched-strictness join; D14 table-header fix). Grounding verified in this session: `exp/src/arith_pilot.py` (whole file), `exp/src/gsm8k_gradient.py` (whole file — implemented, never run, no results dir), `exp/src/common.py` (prompt/greedy/constructor machinery), `exp/src/doubt_judge.py` (judge prompt + hard-wired legacy layout), `exp/src/expa_global_expansion.py:738–800` (`wilson`, `cluster_boot_rate`, `summarize_cell` — reusable as-is), `paper/main.tex:381–406` (§3.4 re-read-vs-recompute conjecture + `tab:regime` with the absorption-metric footnote), `main.tex:485–496` (Lean promise at 488–489), locked anchors `results/arith/summary.json` (recovered .010/.020/.000, poisoned .990/.990/1.000, n=99/position) and `results/gsm8k/summary.json` (recovered .0877, poisoned .8772, n=57, gold 57/250).

---

## 1. Candidate scoring and recommendation

Criteria (from task): (a) machine-auditable truth status of the planted step; (b) formal validation of continuations (closure-validity analogue); (c) clean refutation-distance / local-counterevidence definition; (d) implementation cost given this codebase; (e) reviewer impressiveness. Scores 0–5.

| Candidate | (a) Truth audit | (b) Continuation validation | (c) Distance/counterevidence | (d) Cost | (e) Impressiveness | Verdict |
|---|---|---|---|---|---|---|
| **Lean / miniF2F formal proof traces** | **2** — the kernel checks *proofs*, it does not *decide* truth of an arbitrary planted `have P`; auditing false-vs-true-but-off-path (the paper's own construct trap, the inversion lesson) requires proving P or ¬P, decidable only for `decide`/`norm_num`-able numeric propositions | **5** — kernel; best in class | **1** — no parametric k-hops-to-refutation; goal-state context is not a fact list; cohort-admissible proofs are 1–3 tactics, killing early/mid/late | **0–1** — Lean4+mathlib+REPL/LeanDojo, tactic↔sentence alignment, plus the **fatal cohort collapse**: frozen-greedy Qwen2.5-7B-Instruct pass@1 on miniF2F ≈ 0–3%, and the protocol requires the model's *own* correct trace (a prover model or search violates frozen-greedy and breaks model comparability) | **5** | **Reject this cycle.** Replace the promise at `main.tex:488–489` with a concrete future-work sketch (decidable-proposition intermediates audited by `norm_num`; prover-model protocol variant; whole-proof `have`-chain injection) |
| Informal theorem-proving traces (NL math) | 0 | **0** — validation would be an LLM judge, reinstating exactly professor weakness #3 (black-box validator) on the new regime | 0 | 3 | 2 | Reject |
| Richer synthetic logic (FOL / ProofWriter / disjointness grammar) | 5 | 5 | 4 | 4 | **0–1** — it *is* another author-defined synthetic ontology grammar; a reviewer who says "artifact of a synthetic ontology grammar" is unmoved by a second one. This machinery belongs to Plan A's EXPD grammar extension (weakness #1), not weakness #5 | Reject as regime; already owned by EXPD |
| **Program-execution traces (restricted Python, interpreter-validated)** | **5** — CPython/reference interpreter decides truth of every planted value claim; no author-defined semantics | **5** — exact two-world validation (σ_true vs σ_cf); makes injection-dependence mechanically measurable for **every** dose-response cell, fixing the paper's own `†` asymmetry (poisoning measurable only for positive-categorical injections in PrOntoQA) | **5** — refutation cost decomposes into **re-read distance k_r** (trace lines back to the refuting stated value) × **recompute depth k_c** (unstated evaluations needed), including a **k_c=0 op-free row** where checking is a pure re-read; this causally operationalizes the draft's own §3.4 conjecture (`main.tex:393–395`) | **4** — strict generalization of locked `arith_pilot.py` (same prompt-prefix injection pattern, same string-prefix mechanics); reuses cohort filter pattern, `wilson`/`cluster_boot_rate`, doubt-judge scaffold, lock conventions; ~1–1.5k new lines + the vLLM port or shard script that no plan had budgeted (feasibility G1) | **4** — external interpreter ground truth, stateful imperative semantics vs monotone Horn logic, Nye et al. 2021 scratchpad / CRUXEval lineage, agentic-execution relevance; an externally-authored CRUXEval arm kills "authors generated the distribution" | **RECOMMEND — PRIMARY** |
| Controlled symbolic math + CAS (SymPy) | 4 (restricted invertible-op grammar) | 4 (equivalence-checking caveats: squaring/division give one-way implications) | **1** — every rewrite step is refutable from the immediately preceding line; distance ≈ 1 everywhere; the parametric axis is contrived | 2 (parsing model-emitted math is the tax) | 3 | Reject this cycle (previously "optional secondary"; demoted — see below) |

**Recommendation.**
- **PRIMARY: interpreter-validated program-execution traces** (`EXPG_PROGTRACE`; the EXPD/EXPE/EXPF names are already claimed by Plan A's gradient/evidence-mover and the doubt-forcing rerun per `reports/completeness.md` §2.1). Explicit reasoning: it is the only candidate scoring ≥4 on all five criteria; the two binding constraints are (c) — the paper's central IV is a *parametric distance*, and only this regime gives it a clean handle, indeed a cleaner one than PrOntoQA itself — and cohort feasibility under the frozen-greedy own-trace protocol, where Lean fails outright. It is also not merely "another domain": it is a *better-instrumented* domain (†-fix; polarity, statement type, and negation constant by construction; no goal anchor in the prompt so parroting is structurally absent) — the reviewer-facing line is that the second regime removes three PrOntoQA measurement confounds rather than inheriting them.
- **OPTIONAL CHEAP SECONDARY (changed from Plan B): run the already-written `exp/src/gsm8k_gradient.py`** (verified implemented, never run — no `results/gsm8k_gradient/` exists). Zero new code, ~2–5 GPU-h, gives the GSM8K row a parametric axis (early/mid/late × k=1..3) as appendix-only naturalistic corroboration. Honest caveats to print: its planted step *is* machine-auditably false at the op level (`true_val + delta` against the model's own `a op b = c` calc), but its k is *steps-remaining-in-chain* (goal distance), not audited refutation distance, and continuation validation is propagation-based only — so it corroborates, it does not replicate. SymPy is demoted from "optional secondary" to "not this cycle": distance ≈ 1 everywhere means it cannot contribute to the gradient claim, and under the sprawl budget (skeptic W2/N3) a third regime with no parametric axis is pure page cost.

---

## 2. Full design: `EXPG_PROGTRACE`

### 2.1 Task, trace format, generation

**Programs.** Randomly generated restricted Python: statements `v = c`, `v = u`, `v = u op w`, `v = u op c` with `op ∈ {+, -, *}`, plus a single non-nested accumulator loop idiom `for i in range(a, b): s = s op expr` (loop iterations 3–5). Length L ∈ [12, 20] *executed* steps; integers only; variable names from a fixed pool disjoint from builtins; final line `print(out)`.

**Value-bound guarantees (feasibility fix B-3).** Rejection-sample so that every value in both σ_true *and* σ_cf satisfies |v| ≤ 999 (≤3 digits, tokenizer-uniform across families); `*` operands drawn from [2, 9]; cells with recompute depth k_c ≥ 3 use `+/-` with at most one `*` (the audit found |v|≤999 with unrestricted `×` is unsatisfiable at depth). All values pairwise distinct within a run (prevents true/cf-match ambiguity), and σ_cf ≠ σ_true on ≥2 downstream reads of the planted variable before any overwrite (guaranteed discriminating claims), planted value never accidentally true later, forward-use gap (lines until the corrupted variable is next read) **held constant at 2 across all distance conditions** — a causal-isolation improvement over neghop-k, where evidence distance co-varies with use distance.

**Trace format (dataflow-controlled).** Prompt = instruction + 3-shot exemplars + numbered program listing (the true "world" fully visible in context, mirroring PrOntoQA's visible premises) + "Trace the execution line by line." Trace steps one per line:

```
line 4: t = x + y; t = 15
```

Crucially, a step states the *symbolic* expression and the *result value only* — operand values are **not** restated inline (unlike `arith_pilot.py`'s `a op b = c` format, where operands are adjacent). This is what makes re-read distance k_r a real variable: checking `t = 15` requires retrieving `x` and `y` values from the trace lines (or listing constants) where they were last stated, k_r lines back. Parser: anchored regex per line; newline-delimited (no dependence on the `". "` sentence split). Loops traced per iteration.

**Generator knobs** (the design core): (i) operand staleness k_r; (ii) expression/recompute depth k_c with intermediates never stated; (iii) forward-use gap (fixed = 2); (iv) redundancy (density of re-stating lines between corruption and final use) — held at 0 except the exploratory redundancy moderator cells.

**External-provenance arm.** ~100–150 loop-light, int-only, deterministic functions filtered from CRUXEval (externally authored, MIT-licensed), rendered into the same listing+trace format, executed under a sandboxed subprocess for ground truth; 1-hop and deep/global cells only.

**Decoding/protocol identical to the paper:** frozen Qwen2.5-7B-Instruct (pin HF revision), greedy, fixed budget (gold max_new=512, continuation 384), no feedback, injection = **replacement** of the stated result value on the model's *own* trace line (string-prefix mechanics exactly as `arith_pilot.py:104–110` / `gsm8k_gradient.py:142–143`; under vLLM pass token ids to fix tokenization).

### 2.2 Cohort filter (mirrors the double filter)

Keep instances where (1) the unperturbed greedy trace matches the interpreter on every claim and the final output, and (2) every trace line parses. Stage-0 tunes the grammar until greedy solve ≥ 70% and parse ≥ 95% (report final knobs); publish the funnel CONSORT-style (validatorPlan A.7 pattern); keep the "all rates are upper bounds on a doubly filtered cohort" language verbatim.

### 2.3 Injection sites

Eligible sites = trace lines excluding the first line and the final two (need ≥2 discriminating reads downstream); early/mid/late = ~25%/50%/75% of eligible sites. Dose-response cells run early+mid (late starves discriminating reads, the availability lesson from neghop mid/late N-collapse — disclose rather than repeat).

### 2.4 Perturbation families — transfer map

| PrOntoQA family | EXPG analogue (all planted claims are affirmative value equalities — polarity, negation, statement type constant by construction) | Transfers? |
|---|---|---|
| (a) Benign paraphrase | Same line, same true value, marker/style variant ("so t = 15") | Yes — isolates interruption/style cost; doubles as the N11 anchor cell |
| (b) True interruption (true, off-path) | Replace step with a true claim about another live variable ("note: y = 4", true per σ_true) | Yes |
| (c) Distractor (off-path, inert) | False claim about a **dead** variable (written, never read on any path to output) | Yes — and now *auditable* (interpreter decides falsity), unlike PrOntoQA's unauditable distractor rules. Disclose: discriminating set is empty **by design** (usability manipulation), so derivational absorption is structurally undefined here and only echo is reported; unlike PrOntoQA's †, no *headline* cell has this property |
| (d) Contradiction, d≈0 | Wrong restatement of a value stated on the immediately preceding line (adjacent conflict, zero staleness, zero compute) | Yes — exactly |
| (e) 1-hop falsehood | Wrong result of the current line; operands stated adjacently (k_r=1), single op (k_c=1) | Yes — core |
| (f) Global falsehood | Wrong loop-accumulator/deep-expression claim: no stated intermediates, no conflicting stated value anywhere; falsification requires re-executing ≥4 ops | Yes |
| neghop-k dose-response | **Two orthogonal axes + op-free row**: Axis-R (k_c=0, pure re-read): copy/alias chains, wrong value whose true value is literally stated k_r ∈ {1,2,4,8} lines back — zero computation to falsify. Axis-C (k_r=1): k_c ∈ {1,2,3,5} unstated evaluations. Crossing cells {(4,2),(8,3),(8,5)} | Yes, upgraded: decomposes the construct |
| EXPB certificate flip | Insert true operand restatement ("recall: x = 7") immediately before the byte-identical planted falsehood vs length-matched irrelevant true restatement ("recall: q = 2", non-falsifying) vs baseline; strictly paired triples on the model's **own** trace (an improvement over EXPB, which perturbed dataset proofs) | Yes — exactly |
| EXPC polarity families | Do not transfer — **by design a gain**: surface polarity/type is constant across the whole gradient; the confound EXPC could not break does not exist here. Say this explicitly |
| QA-format / anchor-knockout | Not needed: no target value appears in the prompt, so goal-parroting is structurally absent | Confound removed by construction |
| Verify-instruction (`results_verify` analogue) | Satellite: `LR_VERIFY_PROMPT`-style instruction ("verify each value before using it") × {onehop, deep} at mid | Yes — the empirical answer to "absorption is faithful scratchpad-following" |

### 2.5 Validator tool and metrics mapping

**Tool = reference interpreter (two worlds).** A ~150-line evaluator for the restricted grammar (no `exec` for generated programs; sandboxed subprocess only for CRUXEval) yields σ_true per line; the counterfactual executor forces the planted variable to the planted value at the injection line and runs forward → σ_cf. Every parsed post-injection claim `u = d` at line j is classified: **true-match** (d = σ_true_j(u)), **cf-match** (d = σ_cf_j(u)), **both** (non-discriminating, excluded from the discriminating set — cannot occur on guaranteed-discriminating reads by the distinct-values constraint), **neither**.

| Paper metric | EXPG metric | Notes |
|---|---|---|
| Closure-valid completion | **Trace-valid completion**: every post-injection claim true-matches ∧ final output = σ_true output | Stricter than closure (per-step exact). **Also report the permissive companion, `final-output-valid`** (final output correct regardless of intermediates) — the closure-comparable number. Both strictness levels exist in both regimes (see §6, attack S2) |
| (hop-sound valid, validatorPlan M2) | Trace-valid *is* hop-exact; **line-skip rate** (program lines never traced before the output line) reported per cell = the goal-jump analogue | |
| Injection-dependent | ≥1 discriminating claim cf-matches. Sub-metrics: **next-read absorption** (first discriminating read cf-matches — the exact analogue of arith propagation) and **final-output absorption** (output = σ_cf output). Measurable for every dose-response family — fixes the † footnote | |
| Parroted | Final output correct ∧ ≥1 discriminating claim not true-matching ("lucky completion") | Expected ≈0 absent a goal anchor — itself an informative contrast confirming parroting is goal-anchor-driven |
| Derailed | Final output wrong ∧ ≠ σ_cf output | |
| Unparsed | ≥1 claim-shaped line unparseable; **retained in denominators** (house convention) with dual-denominator reporting and per-cell parse rate as a pre-registered manipulation check (skeptic N1 lesson: the Fig-1 neghop "gradient" was largely an unparsed artifact — this regime must prove it isn't repeating that) | |
| Verbalized doubt | Frozen code-domain lexical regex (fixes the PrOntoQA regex's inflection misses: `contradict\w*`, `inconsist\w*`, adds `bug`, `typo`, `should be`, `off by`, `hold on`, `recheck`) + **Qwen2.5-32B judge as the primary DV** (domain-adapted prompt via the `doubt_judge.py:18–24` scaffold; adapter needed, feasibility G3); per-cell judge/lexical agreement published; lexicon frozen in the pre-registration; **within-regime contrasts only** (cross-regime doubt levels declared incomparable) | |
| (new, free) **Repair event** | Post-injection claim about the planted variable that true-matches — an observable overwrite; crossed with doubt → silent-repair vs flagged-repair | The empirical counter to "faithful-following" (§6, attack S3) |

### 2.6 Cells and statistics

- Core families: 6 × 3 positions × 150 = 2,700. Dose-response: (4 Axis-R + 4 Axis-C + 3 crossing) × 2 positions × 150 = 3,300. Certificates: 3 × 2 positions × 150 = 900. Verify-instruction satellite: 2 × 150 = 300. CRUXEval: 2 cells × ~150 = 300. Redundancy moderator (exploratory): 2 × 150 = 300. Gold: 2,500 programs (→ ~1,700+ cohort). Pilot ~500. **Total ≈ 13k generations.**
- Stats mirror the house style: `wilson` + `cluster_boot_rate` (program = cluster) verbatim from `expa_global_expansion.py:738–767`; Holm within the regime's secondary family; per-cell n=150 gives Wilson half-widths ≈ ±0.08, detectable Δ ≈ 0.12–0.15.
- **Confirmatory discipline (skeptic N3): EXPG contributes exactly ONE pre-registered confirmatory contrast to the master PLAN2 family** — at mid position, op-free Axis-R: judge-doubt(k_r=1) > judge-doubt(k_r=8) AND next-read absorption(k_r=1) < absorption(k_r=8), Holm-corrected pair, with best/worst-case unparsed imputation bounds required to preserve both signs. Monotone trend tests (Cochran–Armitage), Axis-C, the interaction, certificates, and all family contrasts are labeled secondary/exploratory in `PLAN_PROGTRACE.md`. The pre-registration is committed and hash-published (plus OSF timestamp, completeness P2-9) **before Stage 2**, with lineage stated honestly (skeptic N6): this is a *new* pre-registered replication of an exploratory PrOntoQA finding.

---

## 3. Replication, failure, partial replication

**Replication = pre-registered ordering/sign claims, never point values** (cross-regime point comparability is disclaimed in print):
- R1 (control sanity + N11 anchor check): benign paraphrase ≈ true interruption ≈ unperturbed cohort ceiling (within 10 pp).
- R2: adjacent-contradiction and op-free k_r=1 cells show elevated judge-doubt (≥ +0.10 over benign) and lower absorption than deep cells.
- R3: deep/global cells: doubt ≈ benign, next-read absorption ≥ 0.30.
- R4 (**primary**): the k_r=1 vs k_r=8 pair above; monotone-then-plateau shape secondary.
- R5: operand certificate raises judge-doubt vs irrelevant certificate (Δ ≥ 0.15) and lowers next-read absorption.
- **Replication of the paper's claim = R1 ∧ R4 ∧ (R2 ∨ R5).**

**Failure =** R1 holds but every falsehood cell sits at the arithmetic floor (absorption ≥ 0.95, judge-doubt ≤ 0.05, *including op-free k_r=1 and adjacent-contradiction*) at full power, or an inverted primary ordering. **How the paper handles it:** the floor branch is pre-registered as scope-bounding, and the op-free cells make it decisive rather than a consolation: "the model fails to re-check even when checking is a zero-compute re-read of a visible line" is a sharp, publishable boundary result that directly extends `tab:regime`; the headline generality claim then retreats to "the audited protocol and the absorb/flag/route taxonomy transfer; the evidence-distance gradient is conditional on task check-affordance," stated in §5.3 and Limitations, out of the abstract. The paper has survived one self-inflicted inversion by disclosure (UPDATE.md); same posture. Note the floor is *less likely than the skeptic feared* precisely because of the op-free row: every locked floor cell (arith .99, GSM8K .88) required recomputation, while PrOntoQA's own data show the model does respond to re-readable conflicts (contradiction doubt .23–.27, EXPB certificate doubt .38); EXPG deliberately places cells on both sides of the re-read/recompute divide, so *some* internal gradient is the modal outcome.

**Partial replication — pre-committed decision tree in `PLAN_PROGTRACE.md`:**
- Gradient in absorption but not doubt (or converse) → split the claim by channel using the existing Fig-2 doubt/validity dissociation frame; headline follows the channel that generalizes.
- k_r gradient without k_c gradient → the most theoretically interesting outcome: "local falsifiability" refines to **re-readable-evidence availability**, causally confirming `main.tex:393–395`; rename the construct accordingly.
- k_c gradient without k_r → verification-compute story, same move, opposite direction.
- Gradient only under redundancy or only pre-late positions → fold as moderators into §5.3 scope language.
In all branches: every cell reported, the single confirmatory contrast adjudicated as pre-registered, abstract wording frozen only after numbers exist (the sequencing gate skeptic W4/N-minimum-set item 1 imposes program-wide).

---

## 4. Implementation plan

**New modules** (`exp/src/progtrace/`, ~1.3–1.7k lines total; canonical home `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/src/progtrace/`):
1. `gen_programs.py` (~350) — grammar, knobs (L, k_r, k_c, forward-use gap, redundancy, loop config), rejection-sampling guarantees (value bounds, distinct values, ≥2 discriminating reads, no accidental truth), emits worlds + per-line σ_true + injection plans + family metadata.
2. `interp.py` (~150) — reference + counterfactual executors for the restricted grammar (no `exec`); discriminating-line marking; subprocess sandbox path for CRUXEval only.
3. `trace_format.py` (~150) — prompt builder (instruction + 3-shot + numbered listing), strict line parser, unparsed accounting.
4. `inject.py` (~300) — site selection, family constructors (a)–(f), Axis-R/Axis-C/crossing constructors, certificate triples (insertion), verify-instruction variant; manifest per RESULTS_LOCK full-schema conventions.
5. `validate.py` (~300) — claim classification, run classes, repair events, line-skip rate, doubt lexical flag; per-run `validated_outputs.jsonl` + `summary_tables.json` importing `wilson`/`cluster_boot_rate` from `expa_global_expansion.py` (or a factored `stats_common.py`).
6. `cruxeval_arm.py` (~200) — filter (int-only, loop-light, deterministic), render, execute sandboxed, same pipeline.
7. `progtrace_tests.py` — fixture suite in the `expa_tests.py` pattern (parser, cf-executor, both-match exclusion, repair detection, truncation).
8. **Shared infra the plans forgot (feasibility G1):** `src/vllm_gen.py` (~120 lines; offline vLLM engine, chat-template + string/token-id prefix continuation — 0.5–1 day, benefits Plans A/F too) with fallback `scripts/shard_launch.sh` (CUDA_VISIBLE_DEVICES round-robin over the existing HF batch-1 loop, hours). `src/doubt_judge_v2.py` (~100 lines; G3 adapter: judge over arbitrary `validated_outputs.jsonl` with a per-regime prompt registry).
9. `paper_artifacts.py` additions (lock rule: every paper number flows through it): `table_progtrace_main()`, `table_regime_map_matched_strictness()` (subsumes the D14 header fix), dose-response panel in `make_workshop_figures.py`.
10. `PLAN_PROGTRACE.md` + entry in the master `PLAN2.md` confirmatory family; results to `results/EXPG_PROGTRACE{_PILOT}/`; extend RESULTS_LOCK with hashes post-run; locked dirs never touched.

**Reused as-is:** double-filter cohort pattern, `wilson`/`cluster_boot_rate`/Holm convention, string-prefix injection mechanics (`arith_pilot.py:104–110`), judge prompt scaffold (`doubt_judge.py:18–24`), report/lock/changelog conventions, figure pipeline. **Not reusable (stated to avoid Plan B's overclaim):** `validator.py`/`perturb.py` internals are PrOntoQA-sentence-specific; there is no vLLM anywhere in the repo today.

**N + compute (8 idle H200s claimed; estimates given per-GPU-hour so they survive feasibility G2's hardware doubt):** ~13k generations × ~400–600 gen-tokens ≈ 6–8M tokens. HF batch-1 harness (30–60 tok/s/GPU): **45–70 GPU-h ≈ under 1 day wall sharded on 8 GPUs**; after the vLLM port: **~3–6 GPU-h**. 32B judge pass over ~13k continuations: 10–20 GPU-h HF, ~1–2 h vLLM. Optional Stage-3 32B dose-response replication (8 cells × 150 at mid + gold): 30–60 GPU-h HF, a few hours vLLM TP=2. `gsm8k_gradient.py` secondary: ~1.5k gens ≈ 2–5 GPU-h. Engineering dominates: ~5–8 working days.

**Staged path with gates:**
- **Stage 0 (days 1–3):** vllm_gen.py (or shard fallback); generator+interpreter+format+tests; 100-program solve pilot; tune knobs to greedy solve ≥ 70%, parse ≥ 95%; 25 hand-audited validator classifications (publishable spot-check; feeds validatorPlan's Appendix B protocol).
- **Stage 1 pilot (day 4):** mid position, n=35/cell over {benign, true-interruption, adjacent, op-free k_r=1, op-free k_r=8, onehop k_c=1, deep k_c=5}. **Gate G-A:** parse ≥ 0.90 and solve ≥ 0.60, else loop Stage 0. **Gate G-B (floor gate, skeptic W5-1):** adjacent or op-free k_r=1 shows judge-doubt ≥ 0.10 or absorption ≤ 0.85 → full grid; else run only the compact floor-documentation package (6 families + Axis-R × mid × n=150) and take the pre-registered scope-bounding branch. Freeze doubt lexicon; commit + timestamp `PLAN_PROGTRACE.md`.
- **Stage 2 (days 5–8):** full 7B grid + certificates + verify satellite + CRUXEval + judge pass + `gsm8k_gradient.py`; summaries; lock; regenerate paper artifacts.
- **Stage 3 (optional; feeds ICLR criterion #4 hardening):** 32B Axis-R/Axis-C replication; Llama-3.1-8B/OLMo-2 spot cells at mid (identical visible-trace protocol — one N12 protocol note; R1-Distill excluded from this regime or visible-body-only); full k_r×k_c factorial; redundancy interaction.

**Sprawl budget (binding, per storyPlan/completeness W2):** main text gets exactly one subsection (§5.3, ~0.75–1 pp): two-sentence correspondence, the matched-strictness regime map, one two-panel dose-response figure (Axis-R, Axis-C), the confirmatory result, two sentences on the certificate replication. Families table, CRUXEval, verify satellite, funnel, validator spec, GSM8K gradient → new Appendix K. Abstract gains at most one clause, and only in the replication branch.

---

## 5. GSM8K / chained arithmetic disposition

- **Chained arithmetic (locked, n=99): promote by absorption, never re-run.** It is the degenerate corner of EXPG — inline operands (staleness ≈ 0), k_c=1, mandatory-use dataflow, zero redundancy — and its 0.99 propagation is the regime's own strongest floor prior, which is exactly why the op-free k_c=0 row exists. Retire it as a standalone "brittle task" section; re-present `tab:regime` as a **regime map** whose axes are check type (re-read vs recompute) × derivational redundancy, with EXPG cells populating the interior between PrOntoQA and arithmetic.
- **GSM8K (locked, n=57): keep, demote, do not promote to audited regime** — planted-step truth is op-level auditable but continuations have no full-trace validator, and promoting it re-opens the audit attack the paper survived. Reframe as naturalistic corroboration that *motivated* the §3.4 conjecture which EXPG then tests causally. **Upgrade cheaply by running `gsm8k_gradient.py`** (secondary above), appendix-only, with the goal-distance-not-refutation-distance caveat printed.
- **Regime-map table fixes (mandatory):** validatorPlan D14 header fix ("Task-validated completion", per-row metric footnotes — the current header "Closure-valid" over final-answer rates is a live mislabel) plus the new matched-strictness columns (§6, S2).
- **Text edits on landing:** Limitations "two synthetic families plus GSM8K" → "three formally validated regimes (one with external interpreter ground truth and an externally-authored arm) plus GSM8K corroboration"; replace the Lean promise at `main.tex:488–489` with the concrete future-work sketch; abstract clause per the decision tree.

---

## 6. Skeptic attacks — named, and the design change that kills each

| # | Attack (source) | Design change |
|---|---|---|
| S1 | **"Predictable absorption floor; the fallback concedes W5 rather than fixing it"** (skeptic W5-attack-1) | Op-free declarative row (k_c=0 copy/alias chains) as a first-class dose-response axis — falsification is a zero-compute re-read, the genuine d≈0 analogue; Stage-1 floor gate G-B decides full-grid vs floor-documentation; the floor branch is pre-registered as a decisive boundary result ("absorbs even zero-compute re-reads"), not a consolation narrative |
| S2 | **"The two regimes aren't comparable / trace-valid vs closure-valid strictness mismatch"** (skeptic W5-attack-2; the validator-parity attack) | Matched strictness reported **both ways in one table**: PrOntoQA hop-sound-valid (validatorPlan M2, zero-GPU from locked logs) beside EXPG trace-valid, and permissive final-output-valid beside closure-valid; replication criteria are ordering claims only; a printed correspondence table (world=listing↔rules, claim↔entity fact, re-read/recompute↔rule application) plus the honest disanalogy sentence (EXPG *decomposes* the PrOntoQA construct; replication is at the gradient-signature level); D14 header fix so no column mixes metrics |
| S3 | **"Absorption is faithful scratchpad-following, arguably correct — stronger in the execution framing"** (skeptic W5 residual) | Three converging counters: (i) **repair-event metric** elevated to a headline column — observable overwrites prove correction is behaviorally available, and their concentration at low k_r is itself the gradient; (ii) adjacent-contradiction cell — the trace conflicts with *itself*, so "faithful following" is undefined and the model must arbitrate; (iii) verify-instruction satellite — if explicitly told to verify each value and it still absorbs (the PrOntoQA `results_verify` precedent: instruction helps only where evidence is local), the faithful-by-design reading dies. Framing sentence: the context-vs-derivation arbitration *is* the phenomenon, identically present in PrOntoQA |
| S4 | **"CRUXEval arm supports direction claims only"** (skeptic W5 residual) | n≈150/cell, 2 cells, appendix, direction-only language, never in abstract; its job is solely to kill "authors wrote the grammar" |
| S5 | **"Per-regime doubt lexicons make cross-regime doubt levels incomparable"** (W5 residual) + **"doubt instrument weakest where the doubt story lives"** (N10) | Judge-rated doubt is the primary DV for the confirmatory contrast; lexicon frozen at pre-registration (with the inflection fixes the legacy regex lacks); per-cell judge/lexical agreement published; only within-regime doubt contrasts are ever compared |
| S6 | **"Fig-1-style unparsed-rate artifact"** (N1 — fatal-if-shipped class) | Rigid newline-delimited format engineered for parse rate; per-cell parse rate is a pre-registered manipulation check required ≥0.95 and matched across cells of the primary contrast; dual-denominator reporting; the confirmatory pair must survive best/worst-case unparsed imputation bounds |
| S7 | **"Confirmatory-claim proliferation, no global alpha budget"** (N3) | EXPG contributes exactly one Holm-corrected confirmatory pair to the master PLAN2 family (the skeptic's own nomination, "B's k_r contrast"); everything else explicitly secondary/exploratory |
| S8 | **"Pre-registration provenance attackable"** (N6) | `PLAN_PROGTRACE.md` committed, hashed, OSF-timestamped before Stage 2; lineage paragraph states this is a new confirmatory replication of an exploratory finding |
| S9 | **"New-generator distribution shift"** (N11) | Benign-paraphrase and true-interruption anchors with an equivalence check against the unperturbed cohort ceiling (R1) as a manipulation check |
| S10 | **"Cross-model protocol equivalence"** (N12) | Stage-3 models run the identical visible-trace protocol; one protocol note; R1-Distill excluded from EXPG (its deliberative channel makes trace-overwrite a different intervention) |
| S11 | **"vLLM harness doesn't exist; compute 3–5x optimistic; 8×H200 unverifiable"** (feasibility G1/G2, §3-1/2) | `vllm_gen.py` is a named, budgeted task (0.5–1 day) with an HF shard-script fallback; all compute quoted in GPU-hours under both throughput assumptions (45–70 GPU-h HF worst case ≈ <1 node-day; ~3–6 GPU-h after port) |
| S12 | **"Value-bound underspecification with × ops"** (feasibility §3-3) | Explicit generator constraints: |v|≤999 in both worlds by rejection, × operands ∈ [2,9], k_c≥3 cells +/− with ≤1 × |
| S13 | **"Overlooked asset `gsm8k_gradient.py`"** (feasibility §3-4) | Adopted as the cheap secondary; appendix corroboration with honest k-semantics caveat |
| S14 | **"Sprawl recreated"** (skeptic W2) | Hard main-text budget: one subsection + one table + one figure; Appendix K holds everything else; EXPG name avoids the EXPD/EXPE/EXPF collisions |
| S15 | **"Lean promise left unkept"** (regime.md obligation; verified live at `main.tex:488–489`) | Promise replaced by a concrete, feasibility-honest future-work sketch (decidable-proposition intermediates, prover-model protocol variant) |

**Bottom line.** Primary = `EXPG_PROGTRACE` (interpreter-validated program-execution traces with the op-free re-read axis and matched-strictness reporting); secondary = run the existing `exp/src/gsm8k_gradient.py`; SymPy and Lean rejected this cycle with printed reasons; arithmetic absorbed as the regime's degenerate corner; GSM8K demoted to corroboration. Under the gates above the regime either replicates the gradient in a second, better-instrumented symbol system (ICLR criterion #4 satisfied) or converts a floor outcome into a decisive, pre-registered scope boundary — in both branches weakness #5 is answered rather than conceded.