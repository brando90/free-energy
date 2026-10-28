# E10 Track A — API panel: recompute-side absorption across the API roster

**Run date:** 2026-07-27 · Lead: Elyas Obbad · Spec commit `36e620a6…` (E10_WIDEN.md, verified).
**API spend this program: $0.00.** No new API calls were made — see Provenance.

## Provenance / why $0 spend (read first)

The complete E10 §4 API roster was **already executed on 2026-07-06** under
`results/EXPH2_FRONTIER_FOLLOWUPS/`, on exactly the substrates E10 §4 designates:

- **Prefill (Anthropic), EXPG program-trace cells** → `regime2/` (`expg_api_run.py`): `claude-haiku-4-5`, `claude-sonnet-4-5`. Byte-parity prefill (`prefill_parity_check.json`), temp 0.
- **NL substrate, "EXPH channels stand in" (E10 §2/§4)** → `instruct_panel/` + `completions_probe/`: `gpt-4.1`, `gpt-4o`, `gpt-3.5-turbo-instruct`, plus frontier extras `gpt-5.1`, `claude-opus-4-8`, `claude-sonnet-5`.

Because an identical panel (same generator/cells/validator, same roster, same DVs the
spec names) already exists, re-running would duplicate it and burn budget for no gain.
This report **computes the E10 deliverable from that validated data** (`e10_api_analyze.py`,
uniform k/n_measured rates + Wilson CIs) rather than re-issuing calls. Cost ledger
(`results/EXPH_API_MODELS/cost_ledger.json`) unchanged; $85 program stop not approached.
Outputs: `improvement_plan/iclr_exec/recompute_wall/e10_api/{e10_api_summary.json,E10_API_REPORT.md}`.

**Decision point for the parent:** the two OpenAI/completions models were run on the NL
entity-chain substrate (per the E10 §4 "EXPH channels stand in" clause), **not** on the EXPG
program-trace cells (§1's "EXPG cell set on every model"). If a strict EXPG-substrate run of
`gpt-4.1`/`gpt-4o`/`gpt-3.5-turbo-instruct` is wanted, that is a genuine gap and needs a new
run (new harness bridge + budget). Flagged, not silently assumed. Also: per-cell API n is
60–100 (E10 §5.2 "50≤n<100: scored, CI-widened, flagged" band), below the n≥150 target —
a top-up would need explicit authorization.

## 1. EXPG substrate — the recompute wall on frontier prefill models (primary)

Recompute-side = `final_output_absorbed` pooled over `opfree_kr1/kr8, onehop_kc1, deep_kc5`.
Readable-side = `adjacent_contradiction` `final_output_absorbed`.

| model | readable (adj) | opfree_kr1 | opfree_kr8 | onehop_kc1 | deep_kc5 | pooled recompute | margin | replicates (≥.15) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 0.00 (n70) | 0.043 (n70) | 0.217 (n60) | **1.00** (n60) | **1.00** (n60) | 0.544 (n250) | +0.544 | **yes** |
| claude-sonnet-4-5 | 0.00 (n70) | 0.043 (n70) | 0.050 (n60) | **0.983** (n60) | **1.00** (n60) | 0.500 (n250) | +0.500 | **yes** |

**Finding — the wall is sharper than the pilot, and sits exactly at "requires an operation."**
Both frontier prefill models **catch the copy/alias cells** (`opfree`, a stated root re-read at
distance k_r) at 0.04–0.22, yet **absorb the genuinely computed cells** (`onehop_kc1` = 1 op,
`deep_kc5` = 5 ops) at ~1.00. The pooled 0.50–0.54 is bimodal (≈0 on opfree, ≈1.0 on the op
cells), not a middling rate. Readable contradiction is caught 100% (absorption 0.00). This is the
readable-vs-recompute asymmetry replicating, and it localizes the boundary to re-execution of an
operation rather than mere re-read distance. (Qwen-7B pilot absorbed opfree too, at 0.83–0.91;
frontier prefill is stricter on copies but equally blind on computed values.)

## 2. NL substrate — "EXPH channels stand in" (descriptive; not pooled with §1)

Recompute-side = `inj_derivational` (class=="poisoned") on `global_falsehood` (poisoning-measurable).
Readable-side = `stated_complement` (explicit correction) on `one_hop_falsehood`. Caveats printed
per E10 §4 (instruct chat re-presentation can substitute re-planning for continuation; NL entity-chain
substrate, not the EXPG cells; never pooled with prefill/open-weights).

| model | mode | recompute inj_deriv | readable correction | global_falsehood class dist |
|---|---|---|---|---|
| gpt-4.1 | instruct | 0.60 (n100) | 0.29 | poisoned 60 / valid 36 / parrot 2 / unparsed 2 |
| gpt-4o | instruct | 0.34 (n100) | 0.30 | valid 64 / poisoned 34 / parrot 1 / unparsed 1 |
| gpt-5.1 | instruct | 0.33 (n100) | 0.33 | valid 64 / poisoned 33 / derailed 3 |
| claude-opus-4-8 | instruct | **0.00** (n100) | **0.98** | valid 100 |
| claude-sonnet-5 | instruct | **0.01** (n100) | **1.00** | valid 98 / derailed 1 / poisoned 1 |
| gpt-3.5-turbo-instruct | completions | 0.53 (n91) | 0.044 | poisoned 48 / parrot 17 / valid 17 / derailed 8 / unparsed 1 |

**Findings.** (a) **The two newest Anthropic instruct models CHECK at this tier** — `claude-opus-4-8`
and `claude-sonnet-5` show ~0 derivational absorption and ~1.0 readable correction. Per E10 §6 this
is a **positive finding ("checks at this tier")** that makes the asymmetry *smaller* and counts
*against* the program-level claim for those two models — reported as-is, no reframing. (b) The GPT
instruct models partially absorb the derivational plant (0.33–0.60). (c) `gpt-3.5-turbo-instruct`
(the cleanest single-token-stream API analogue, via legacy completions) is the most wall-like API
row: 0.53 derivational absorption with almost no readable correction (0.044). NB: the NL readable
channel is a *correction* rate, not an absorption rate, so §1-style margins are not computed here.

## 3. Refusals — the parent's separate ask

**Zero refusals to continue traces across all 8 models (~3,200 rollouts).** The refusal scan
(start-of-turn "I can't/cannot/unable to continue/comply", "I must decline", etc.) returned 0 hits
for every model on both substrates. No model that previously continued now refuses. Non-refusal
reliability noise only: haiku 6 derailed, gpt-5.1 3 derailed (on global_falsehood), gpt-3.5 1
generation failure, sonnet/others 0. **Nothing to log as a refusal finding — continuation
compliance was universal.**

| model (substrate) | refusals | rows | genfail | derailed |
|---|---|---|---|---|
| claude-haiku-4-5 (EXPG) | 0 | 460 | 0 | 6 |
| claude-sonnet-4-5 (EXPG) | 0 | 460 | 0 | 0 |
| gpt-4.1 (NL) | 0 | 399 | 0 | — |
| gpt-4o (NL) | 0 | 400 | 0 | — |
| gpt-5.1 (NL) | 0 | 398 | 0 | — |
| claude-opus-4-8 (NL) | 0 | 400 | 0 | — |
| claude-sonnet-5 (NL) | 0 | 400 | 0 | — |
| gpt-3.5-turbo-instruct (NL) | 0 | 364 | 1 | — |

## 4. Phrasing guardrails honored (RECOMPUTE_SCOOP_CHECK §3)

Numbers are **measured absorption rates on the named roster**, never a universal law. The
asymmetry is reported per model; it **replicates on the two prefill models** and **fails to replicate
(models check) on opus-4-8/sonnet-5** — reported both ways. Single-trace in-stream framing kept;
no memory/staleness or asked-to-grade wording.
