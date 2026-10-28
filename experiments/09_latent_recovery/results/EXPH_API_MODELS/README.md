# EXPH_API_MODELS README: API-model generality arm (descriptive / exploratory)

Generated: 2026-07-06 (skampere2). Status: **descriptive/exploratory; NOT confirmatory** — no pre-registered hypothesis is scored here.

**History.** The pre-stated bridge gate (section 2) FAILED on both arms → per the stop rule, the original 6-model Stage 2 was halted with $0 spent. Author decision (2026-07-06): proceed with a **REDUCED Stage 2, prefill mode only** — `claude-haiku-4-5` and `claude-sonnet-4-5` (the two API models that accept assistant-turn prefill, i.e. where no instruction-format shift exists) — and **exclude all instruct-mode rows** (claude-sonnet-5, claude-opus-4-8, gpt-4.1, gpt-5.1, and haiku-instruct), with the bridge data as the measured justification.

**Headline.** At the API-model capability tier, the paper's absorption phenomena vanish in the same harness that produces them robustly on Qwen2.5-7B: across all 12 cells x 2 models (n=650 rows/model), derivational injection-dependence is **0.000 everywhere**, order-independent plant-dependent content is **0 rows anywhere** (including cat_false_usable_d1, where Qwen prefill absorbs at 0.863), and closure-valid recovery is **0.97–1.00 in every cell** (Qwen: 0.36 on global falsehood, 0.09 on usable worlds). Both models perform the d=1 refutation check on usable plants that Qwen never executes. Total API spend: **$4.48**.

---

## 1. Reduced Stage 2 (prefill mode, own-trace golds, mid injections)

Mechanics: assistant turn carries the model's own proof-so-far + planted step; user turn is byte-identical to the `src/common.py` prompt string; prefill is byte-identical to the local `answer_prefix` modulo its single leading space (carried by the turn boundary; verified on 3 locked rows before any gold spend, `prefill_parity_check.json`, PASS). Decoding: temperature=0, max_tokens=300. All planted claims audited (global/one-hop/reuse FALSE; paraphrase/interruption TRUE); reuse cells assert measured d==1 from the prefix; injected statements are rebuilt with the standard EXPA machinery on each model's own gold steps (machinery reproduces the locked EXPA mid injections 100/100 on Qwen's traces).

### Gold funnels (double filter: solved AND validator-valid AND >=3 intermediate entity steps)

| model | attempts | solve rate | validator-valid fail | too-few-steps | eligible | worlds eligible |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 300 | **1.000** | 11 | 131 | 158 (target 150 reached) | 150/150 |
| claude-sonnet-4-5 | 350 | **1.000** | 4 | 185 | 161 (target 150 reached) | 150/150 |

Both models solve every attempted problem; ineligibility is dominated by **proof compression** (fewer than 3 intermediate entity facts — the models skip hops), so the perturbation cohort skews toward longer chains. Disclosed.

### claude-haiku-4-5 (prefill; n=125/core cell, 75/reuse cell)

| cell | n | valid | deriv inj-dep | echo | stated-comp | doubt | unparsed | parse |
|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 125 | 0.992 | 0.000 | 0.192 | 0.000 | 0.176 | 0.000 | 1.00 |
| true_interruption | 125 | 1.000 | 0.000 | 0.024 | 0.000 | 0.712 | 0.000 | 1.00 |
| one_hop_falsehood | 125 | 0.992 | 0.000 | 0.000 | 0.888 | 0.448 | 0.000 | 1.00 |
| global_falsehood | 125 | 0.992 | 0.000 | 0.000 | 0.000 | 0.520 | 0.000 | 1.00 |
| cat_false_usable_d1 | 75 | 0.973 | 0.000 | 0.000 | 0.333 | 1.000 | 0.027 | 0.97 |
| cat_false_inert_d1 | 75 | 0.987 | 0.000 | 0.000 | 0.000 | 0.987 | 0.000 | 1.00 |

### claude-sonnet-4-5 (prefill; n=125/core cell, 75/reuse cell)

| cell | n | valid | deriv inj-dep | echo | stated-comp | doubt | unparsed | parse |
|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 125 | 1.000 | 0.000 | 0.216 | 0.000 | 0.176 | 0.000 | 1.00 |
| true_interruption | 125 | 0.968 | 0.000 | 0.000 | 0.000 | 0.472 | 0.000 | 1.00 |
| one_hop_falsehood | 125 | 1.000 | 0.000 | 0.000 | 0.944 | 0.808 | 0.000 | 1.00 |
| global_falsehood | 125 | 0.992 | 0.000 | 0.000 | 0.000 | 0.496 | 0.000 | 1.00 |
| cat_false_usable_d1 | 75 | 0.973 | 0.000 | 0.000 | 0.000 | 0.560 | 0.000 | 1.00 |
| cat_false_inert_d1 | 75 | 1.000 | 0.000 | 0.000 | 0.000 | 0.373 | 0.000 | 1.00 |

Qwen2.5-7B reference (locked, same machinery): global falsehood valid 0.360 / poisoned 0.373; one-hop valid 0.640 / doubt 0.273; EXPD usable d1 derivational 0.863 / valid 0.093; inert derivational 0.000.

### Reading (descriptive)

1. **No absorption at this tier.** deriv inj-dep = 0.000 in all 12 cells; the order-independent audit (`api/api_diagnosis.json`) confirms **zero rows with any plant-dependent content**, including the usable-bridge cells built to make reuse maximally attractive. The paper's absorption/poisoning phenomena are capability-bound, not harness artifacts: the identical harness, injection machinery, and validator produce Qwen's 0.36–0.86 effects and the API models' zeros.
2. **The d=1 check happens.** On usable plants, haiku explicitly derives the refuting rule ("Wait... Every rahupus is not a hetpus. So Stella is not a hetpus.") before restarting — the one-hop verification step that Qwen-scale models never performed (EXPD: checking flat/absent at every d>=1). stated-complement on one-hop: haiku 0.888, sonnet 0.944 (explicit correction of the planted negation).
3. **Recovery is restart-shaped.** Both models frequently rewrite the proof rather than continue it (prefix-restatement: haiku 0.23–1.00, sonnet 0.21–0.98 per cell) — but unlike the Qwen instruct bridge, the planted falsehood NEVER carries into the rewrite (echo 0.000 in every falsehood cell; benign echo 0.19–0.22 is restatement of the TRUE paraphrase content and is counted valid by closure semantics).
4. **Doubt is not falsity-specific for chat models.** The lexical doubt regex fires on TRUE plants too (true_interruption: haiku 0.712, sonnet 0.472) because restart hedging ("Wait, let me restart") matches it. Falsity-specific signals here are stated-complement and the (null) derivational channel; doubt should be read as interruption-sensitivity, not error detection.

---

## 2. Bridge arm and the instruct-mode exclusion (measured justification)

Pre-stated gate, run on Qwen2.5-7B-Instruct (pinned `a09a3545…`, vLLM greedy) with the LOCKED EXPA rows — same questions, gold prefixes, and injected statements; only the prompt format changed from in-stream prefill to chat-instruct continuation (n=100/cell mid; usable/inert on audited generated worlds n=60/cell):

| cell (instruct v1) | valid | deriv inj-dep | echo | stated-comp | doubt | locked prefill anchor |
|---|---|---|---|---|---|---|
| benign_paraphrase | 0.910 | 0.000 | 0.840 | 0.000 | 0.000 | valid 0.880, doubt 0.127 |
| true_interruption | 0.890 | 0.000 | 0.570 | 0.000 | 0.000 | valid 0.867 |
| one_hop_falsehood | 0.400 | **0.460** | 0.510 | 0.280 | 0.010 | valid 0.640, poisoned 0.020, doubt 0.273 |
| global_falsehood | 0.300 | 0.640 | 0.690 | 0.000 | 0.000 | valid 0.360, poisoned 0.373 |
| cat_false_usable_d1 | 0.850 | **0.150** | 0.150 | 0.000 | 0.000 | EXPD deriv 0.863, valid 0.093 |
| cat_false_inert_d1 | 0.917 | 0.067 | 0.083 | 0.000 | 0.000 | EXPD deriv 0.000 |

**Gate: FAIL** — inj-dep(one-hop) 0.46 > 0.10; usable−inert gap 0.083 < 0.30 (orderings, global inj-dep, parse rate passed). Diagnosis (`bridge/diagnosis.json`):

- 100% of instruct-format "poisoning" is the continuation *re-asserting the planted line* while rewriting proof segments (64/64 global, 46/46 one-hop, 9/9 usable rows), not first-introduction downstream reuse; one-hop plants cannot feed rules by grammar, so its inj-dep can only be echo.
- Echo-corrected (order-independent) global-plant reuse is format-robust: **0.35/0.36 vs 0.373 locked**.
- EXPD usable-bridge absorption genuinely collapses: **0.863 -> 0.15 -> 0.00** (v2 anti-restate variant); the model re-plans through the canonical chain (valid 0.85 -> 1.00 vs 0.093 prefill). Absorption is carried by in-stream continuation pressure, which chat re-presentation removes.
- Doubt collapses (0.273 -> 0.01) and reroutes to silent correction (stated-complement 0.28/0.36).
- The v2 anti-restate diagnostic shows the two failures trade off (suppressing restatement kills absorption entirely): **not fixable by prompt tightening**.

Consequently (author decision): instruct-mode rows for claude-sonnet-5, claude-opus-4-8, gpt-4.1, gpt-5.1 and haiku-instruct are **excluded** — chat re-presentation changes the phenomenon (re-planning replaces continuation), so those rows would not test the paper's mechanisms. This extends the pre-existing exclusion rationale (Claude Fable 5 / o-series / R1: deliberation-channel non-equivalence) with direct same-model measurements.

**Two format-robust facts** survive the shift and can be cited: (a) the closure-valid ordering global < one-hop < benign holds in sign in every regime tested (locked prefill, instruct v1, instruct v2); (b) echo-corrected global-plant reuse reproduces almost exactly (0.35/0.36 vs 0.373).

---

## 3. Cost ledger

| model | requests | input tok | output tok | USD |
|---|---|---|---|---|
| claude-haiku-4-5 | 1,389 | 672,381 | 136,349 | $1.35 |
| claude-sonnet-4-5 | 1,152 | 574,129 | 93,519 | $3.13 |
| local vLLM (Qwen bridge + diagnostics, 1 GPU) | 1,190 gens | — | — | $0.00 |
| **total** | | | | **$4.48** |

Target ~$25–60, hard stop $100: spend is 7% of the lower target bound (the 4 excluded instruct models account for the difference). Ledger includes smoke tests and the 300 aborted mixed-pool gold attempts (see disclosure 3). Token-level detail in `cost_ledger.json`.

## 4. Disclosures

1. **Prefill is a chat-turn boundary, not one token stream.** The user turn is byte-identical to `common.py`'s user string and the assistant prefill to its `answer_prefix` (modulo the leading space) — but Anthropic's chat rendering inserts turn structure between "A:" and the prefill. This is the closest available regime; parity documented in `prefill_parity_check.json`.
2. **Own-trace golds:** injections are built on each model's own greedy (temperature=0) rollouts; condition availability was 100% (650/650 manifest rows per model, zero injection-audit rejects).
3. **In-grammar pool restriction (added after first gold wave):** the legacy candidate pool mixes composed-grammar worlds (disjunction/conjunction rules) that the validator cannot certify by construction; the locked EXPA run burned 3,032 UNCAPPED attempts against the mixed pool (5.0% eligibility) and all 150 of its eligible problems are strictly in-grammar. EXPH restricts the pool to the 651/5,819 in-grammar candidates so the 400-attempt cap is meaningful; this is equivalent to the locked run's implicit filter. The 300 pre-restriction haiku attempts (both grammars) are retained in raw/ledger; on in-grammar candidates haiku was 21/33 eligible vs 0/267 on composed.
4. **Cohort skew:** ineligibility is dominated by proof compression (haiku 131/300, sonnet 185/350 below the >=3-intermediate-steps floor), so perturbed problems over-represent long chains. Solve rate itself is 1.000 for both models.
5. **Doubt regex** fires on restart hedging (0.47–0.71 on TRUE interruptions) — read as interruption-sensitivity, not error detection; stated-complement is the falsity-specific verbal channel here.
6. **max_tokens=300** (legacy local runs: 192 HF tokens); truncations would surface as derailed and are counted (none material: valid 0.97–1.00, parse 0.97–1.00).
7. **Bridge continuations**: greedy vLLM, max_new_tokens=256; continuation post-processing everywhere is limited to stripping a leading "A:" and truncating at a regenerated "Q:" block (flagged per row).
8. Wilson + cluster bootstrap CIs (problem-cluster for legacy, family-cluster for worlds) in `summary_tables.json`; rows are one-sample descriptive estimates.

## 5. Artifacts

- `api/` — per-model raw gold/perturb, cohorts, manifests, validated rows, funnels, `api_diagnosis.json` (restatement + order-independent plant-dependence).
- `bridge/`, `bridge_diag_v2/` — bridge arm (registered + diagnostic variant), `bridge_gate.json`, `diagnosis.json`.
- `worlds/` — 150 audited generated cat worlds (75 families x usable/inert, d=1; 0 audit failures).
- `summary_tables.json`, `validated_outputs.jsonl`, `manifest.jsonl`, `raw_generations.jsonl`, `run_metadata.json` (prompt block + sha, per-model API param policy; **no keys**), `cost_ledger.json`, `prefill_parity_check.json`, `model_resolution.json` (exclusion records).
- Code: `improvement_plan/exph/{exph_common.py, api_gen.py, bridge_run.py, exph_run.py}` (+ `.venv-exph`).
