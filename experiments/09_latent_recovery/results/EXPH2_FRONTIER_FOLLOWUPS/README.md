# EXPH2_FRONTIER_FOLLOWUPS README: frontier follow-ups (descriptive / exploratory)

Generated: 2026-07-06 (skampere2). Status: **descriptive/exploratory; NOT confirmatory.** Author-approved follow-up package to EXPH (results/EXPH_API_MODELS/). Cumulative spend across EXPH+EXPH2: **$30.81** (Anthropic $27.30, OpenAI $3.52; hard stop $100).

**One-paragraph summary.** (A) Frontier rejection generalizes to the EXPG execution-trace regime only in the *memory* channel: both Claude 4.5-tier models reject re-readable copy errors and adjacent contradictions that Qwen absorbs at 0.74–1.00, but absorb *re-computable* falsehoods at 0.98–1.00 — even a one-addition recompute — exactly like Qwen; all of it silent (doubt 0, repair 0). (B) The 2025-mid Anthropic tier (claude-opus-4-1, prefill) already rejects everything in the PrOntoQA regime — the absorption→rejection transition in the Claude lineage happened at or before opus-4-1. (C) In the SAME instruct format as the Qwen bridge anchor, current OpenAI chat models still silently reuse audited-false planted categories at or above Qwen-7B levels (echo-corrected global reuse: gpt-4.1 0.580, gpt-4o 0.340, gpt-5.1-reasoning-off 0.340 vs Qwen 0.35), while Anthropic models do not (sonnet-5 0.010, opus-4-8 0.000) — a lab-level, not capability-level, split. (D) The completions endpoint is alive; gpt-3.5-turbo-instruct under TRUE text-completion continuation (protocol-identical to local prefill) is the deepest absorber measured: global valid 0.187, echo-corrected reuse 0.571.

---

## A. Second-regime frontier check (regime2/; EXPG program traces, prefill, Anthropic)

Reuses `improvement_plan/expg/` verbatim (generator with pilot Stage-0 final knobs + pilot seed bases, trace format, injector, two-world interpreter validator). Own-trace golds: 250 programs/model (alloc kr1 70 / kr8 60 / kc1 60 / kc5 60), **solve rate 1.000 for both models on all shapes** (Qwen pilot: 0.83 overall, 0.45 on kc5). Parity: user turn byte-identical to the EXPG prompt builder; gold prefill "line 1:"; continuation prefill = manifest prefix_text with the trailing newline stripped (Anthropic rejects trailing whitespace; flagged per row; digit-gluing measured at 0.000 everywhere).

| cell | haiku: trace-valid / next-read abs / final-out abs | sonnet-4-5: trace-valid / next-read abs / final-out abs | Qwen pilot next-read abs |
|---|---|---|---|
| benign_paraphrase | 1.000 / n.m. / n.m. | 1.000 / n.m. / n.m. | n.m. |
| true_interruption | 0.786 / n.m. / n.m. | 0.871 / n.m. / n.m. | n.m. |
| adjacent_contradiction | 0.800 / **0.000** / 0.000 | 0.614 / **0.000** / 0.000 | 0.743 |
| opfree_kr1 | 0.943 / **0.043** / 0.043 | 0.957 / **0.043** / 0.043 | 1.000 |
| opfree_kr8 | 0.700 / **0.217** / 0.217 | 0.950 / **0.050** / 0.050 | 1.000 |
| onehop_kc1 | 0.000 / **1.000** / 1.000 | 0.017 / **0.983** / 0.983 | 0.943 |
| deep_kc5 | 0.000 / **1.000** / 1.000 | 0.000 / **1.000** / 1.000 | 1.000 |

n = 70/70/70/70/60/60/60 per model. Repair events 0.000 and doubt (frozen + broad lexicons) 0.000 in every cell for both models — all rejection is *silent true-value use*, all absorption is silent too. Unparsed concentrated in the note-insertion cells (haiku 0.20–0.21, sonnet up to 0.386 on adjacent_contradiction; retained in denominators, house convention).

**Generalization verdict:** frontier rejection is NOT PrOntoQA-shaped — it transfers to the execution regime — but it is **verification-channel-specific**. Where the true value can be RE-READ (opfree: k_c=0, planted value contradicts a value stated k_r lines up), the frontier models reject near-totally (0.043 at k_r=1) — and a real k_r gradient appears for haiku (0.043→0.217) that flattens for sonnet (0.043→0.050). Where the true value must be RE-COMPUTED (onehop_kc1 = one addition; deep_kc5 = 5-op chain), absorption is total (0.98–1.00), indistinguishable from Qwen. The paper's production-vs-verification asymmetry survives at the frontier in the compute channel.

## B. Anthropic generation ladder (ladder/; prefill, same six EXPH cells)

Availability probe: `claude-opus-4-1` served (retires 2026-08); `claude-opus-4-0`, `claude-sonnet-4-0` return 404 (recorded in probes.json) — ladder = opus-4-1 only.

claude-opus-4-1 (prefill; gold: 300 attempts, solve **1.000**, 123 eligible — target 150 not reached at cap, cells bounded at n=100; worlds 150/150):

| cell | n | valid | deriv inj-dep | echo | plant-dep | stated-comp | doubt |
|---|---|---|---|---|---|---|---|
| benign_paraphrase | 100 | 1.000 | 0.000 | 0.450 | 0.000 | 0.000 | 0.450 |
| true_interruption | 100 | 0.990 | 0.000 | 0.000 | 0.000 | 0.000 | 0.280 |
| one_hop_falsehood | 100 | 0.990 | 0.000 | 0.000 | 0.000 | 0.130 | 0.100 |
| global_falsehood | 100 | 0.990 | 0.000 | 0.000 | 0.000 | 0.000 | 0.220 |
| cat_false_usable_d1 | 75 | 0.907 | 0.000 | 0.000 | 0.000 | 0.000 | 0.133 |
| cat_false_inert_d1 | 75 | 0.867 | 0.000 | 0.000 | 0.000 | 0.000 | 0.067 |

**Transition verdict:** the 2025-mid Opus tier does NOT absorb — zero derivational reuse and zero plant-dependent content in all six cells, same as the 4.5 tier (EXPH). Within the Anthropic lineage the absorption→rejection transition therefore happened at or before claude-opus-4-1; the earlier (4.0) rungs are no longer served, so the boundary cannot be bracketed more tightly on this ladder. Note opus-4-1's high benign-cell echo (0.45) + doubt (0.45): it restates and hedges much more than 4.5-tier models while remaining derivationally clean.

## C. Cross-lab SAME-FORMAT instruct panel (instruct_panel/; appendix-only)

Registered bridge instruct prompt (byte-identical to `EXPH_API_MODELS/bridge/`, sha-pinned); own-trace golds per model under the instruct format; 4 core cells x mid x n<=100. **NOT comparable to prefill rows except via the anchor**: the Qwen bridge showed the echo-corrected (order-independent) global-reuse metric is format-robust (instruct 0.35 vs locked prefill 0.373), so that column is the PRIMARY panel metric; validity ordering in sign is SECONDARY.

| model (instruct) | gold solve / eligible | global: valid | global: **echo-corrected reuse** | one-hop: valid | one-hop: stated-comp | benign: valid |
|---|---|---|---|---|---|---|
| Qwen2.5-7B (bridge anchor) | locked cohort | 0.300 | **0.350** | 0.400 | 0.280 | 0.910 |
| gpt-4.1 | 1.000 / 119 | 0.360 | **0.580** | 0.590 | 0.290 | 0.970 |
| gpt-4o | 0.995 / 124 | 0.640 | **0.340** | 0.800 | 0.300 | 0.970 |
| gpt-5.1 (reasoning_effort=none) | 0.990 / 118 | 0.640 | **0.340** | 0.930 | 0.330 | 0.980 |
| claude-sonnet-5 (thinking disabled) | 1.000 / 109 | 0.980 | **0.010** | 1.000 | 1.000 | 0.990 |
| claude-opus-4-8 | 1.000 / 101 | 1.000 | **0.000** | 1.000 | 0.980 | 1.000 |

Validity ordering global < one-hop < benign holds in sign for Qwen and all three OpenAI models; it is degenerate at ceiling for the two Anthropic models (all cells 0.98–1.00). Doubt ~0 for every panel model (instruct-format doubt collapse, as in the bridge). gpt-5.1 accepted reasoning_effort="none" at runtime (no substitution needed).

**Panel reading:** in the identical format, on their own traces, with audited-false plants, the current OpenAI chat lineage still silently walks planted false categories into proofs at Qwen-7B-or-higher rates (gpt-4.1 exceeds the 7B anchor), while the current Anthropic lineage rejects near-totally. Cross-lab generality of the paper's absorption phenomenon: it is not a small-model artifact; it is present in currently-shipping frontier chat models of one major lab and absent in the other's.

## D. Completions-endpoint probe (completions_probe/)

Probe: `v1/completions` is ALIVE for both `gpt-3.5-turbo-instruct` and `davinci-002` (probes.json). gpt-3.5-turbo-instruct run under TRUE text-completion continuation — prompt = instruction+fewshot+question+"\nA: "+proof-so-far+planted step, exactly the `common.py` rendering with no chat template: the only OpenAI model protocol-identical to the local prefill regime, and the oldest generation point (2023). Gold: 250 attempts, solve 0.808, 91 eligible (target 100 not reached at cap; cells n=91).

| cell | n | valid | deriv inj-dep | echo | plant-dep | stated-comp | doubt | parse |
|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 91 | 0.758 | 0.000 | 0.066 | 0.000 | 0.000 | 0.000 | 0.90 |
| true_interruption | 91 | 0.615 | 0.000 | 0.187 | 0.000 | 0.000 | 0.000 | 0.97 |
| one_hop_falsehood | 91 | 0.692 | 0.000 | 0.033 | 0.000 | 0.044 | 0.000 | 0.99 |
| global_falsehood | 91 | **0.187** | 0.527 | 0.132 | **0.571** | 0.000 | 0.000 | 0.99 |

The deepest absorber measured in the whole project (echo-corrected reuse 0.571 vs Qwen prefill 0.373); validity ordering holds in sign (0.187 < 0.692 < 0.758). Being protocol-identical to prefill, this row can join the main-table lineage as the 2023 generation point.

## Cross-experiment picture (echo-corrected global reuse, all regimes)

| model (year, mode) | echo-corrected global reuse |
|---|---|
| gpt-3.5-turbo-instruct (2023, TRUE completion) | 0.571 |
| Qwen2.5-7B (2024 open, local prefill, locked) | 0.373 |
| Qwen2.5-7B (instruct bridge anchor) | 0.350 |
| gpt-4o (2024, instruct) | 0.340 |
| gpt-4.1 (2025, instruct) | 0.580 |
| gpt-5.1 (2025, instruct, reasoning off) | 0.340 |
| claude-opus-4-1 (2025-mid, prefill) | 0.000 |
| claude-haiku-4-5 / sonnet-4-5 (2025, prefill, EXPH) | 0.000 |
| claude-sonnet-5 / opus-4-8 (2026, instruct) | 0.010 / 0.000 |

Mode differs across rows (prefill vs instruct vs completion); rows are linked through the format-robustness anchor and should be read as a descriptive gradient, not a single controlled ladder.

## Cost ledger (cumulative EXPH + EXPH2)

| model | requests | in tok | out tok | USD |
|---|---|---|---|---|
| claude-haiku-4-5 | 2,099 | 1,277,313 | 242,348 | $2.49 |
| claude-sonnet-4-5 | 1,862 | 1,179,061 | 199,542 | $6.53 |
| claude-opus-4-1 | 1,000 | 498,153 | 71,374 | $12.83 |
| claude-opus-4-8 | 650 | 472,653 | 40,522 | $3.38 |
| claude-sonnet-5 | 650 | 474,520 | 43,635 | $2.08 |
| gpt-4.1 | 599 | 301,227 | 45,072 | $0.96 |
| gpt-4o | 600 | 300,371 | 41,461 | $1.17 |
| gpt-5.1 | 598 | 299,567 | 45,304 | $0.83 |
| gpt-3.5-turbo-instruct | 613 | 274,987 | 73,553 | $0.56 |
| **total** | | | | **$30.81** |

EXPH2 additional spend: ~$26.33 (target ~$40); hard stop $100 enforced per-request on the cumulative multi-ledger sum.

## Disclosures

1. All rows descriptive; one-sample greedy-where-allowed estimates (sonnet-5/opus-4-8 reject temperature; gpt-5.1 run with reasoning off — determinism not guaranteed).
2. Regime2: continuation prefill drops the trailing newline of the local rendering (API constraint; flagged per row; measured digit-gluing 0.000); the 32B judge pass was not run — doubt is the two frozen/broad lexicons (all zero) and rejection is measured derivationally (next-read/final-output/repair channels).
3. Regime2 note-insertion cells carry elevated unparsed rates (up to 0.386, sonnet adjacent_contradiction) — the models often comment on the planted note in non-trace-shaped lines; runs retained in denominators.
4. Ladder: gold target 150 not reached at the 300-attempt cap (123 eligible; eligibility bound = proof compression, solve rate 1.000); opus-4-0/sonnet-4-0 are no longer served.
5. Panel: appendix-only cross-format rows; prefill and instruct rows are not directly comparable (bridge-measured format shift); the echo-corrected reuse column is the format-robust bridge metric.
6. Completions: gold target 100 not reached at cap (91; solve 0.808); stop sequence "\nQ:"; max_tokens 300.
7. Legacy pool = the 651 in-grammar candidates (EXPH disclosure 3 applies); worlds pool = the audited EXPH 150-world cat pool.
8. Wilson + cluster bootstrap CIs (problem/family/program cluster) in `summary_bcd.json` / `regime2/summary_regime2.json`; per-model API params in run_metadata.json; keys never logged.
