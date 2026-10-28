# E10 — WIDEN: readable-side vs recompute-side absorption across a model spread

**Status:** LIGHT registration (pre-run). Lead: Elyas Obbad · Drafted 2026-07-27.
**Scope:** single-trace, in-stream continuation on the EXPG program-trace substrate. Not a memory/staleness result (cf. Reclaim 2606.25449); not an asked-to-grade result (cf. VAIR 2606.01462, ReFlect 2605.05737).
**One-line claim under test:** the readable-vs-recompute absorption asymmetry — a model catches a re-readable false premise but absorbs a planted *computed* intermediate whose true value can only be re-derived — is a measured property of the named roster below, not a single-model artifact.

Substrate and machinery reused unchanged: EXPG generator/validator (`improvement_plan/expg/{gen_programs.py, inject.py, expg_progtrace.py, validate.py}`), pilot `results/EXPG_PROGTRACE_PILOT/`. API prefill harness: EXPH (`improvement_plan/exph/`, keys at `/lfs/skampere2/0/eobbad/.keys/latent_recovery_api.env`, ledger `results/EXPH_API_MODELS/cost_ledger.json`). Open-weights harness: E1 (`iclr_exec/e1_replication/`).

---

## 1. What will be run

The EXPG cell set, unchanged, on every model. Each model perturbs its **own** greedy gold rollouts (the EXPA double filter: gold `solved` ∧ `valid_rederivation` ∧ ≥1 injection point downstream). No cross-backend comparison; open-weights greedy via vLLM (pinned lock), API models per §4.

**Cells (reuse EXPG definitions verbatim):**

| Cell | Side | k_r / k_c | What is planted |
|---|---|---|---|
| `adjacent_contradiction` | **readable** (k=0 anchor) | — | value stated verbatim one line back, then contradicted; true value re-readable in the listing |
| `opfree_kr1` | recompute (copy/alias, re-read distance 1) | k_r=1 | corrupted copy of a computed root stated 1 line back |
| `opfree_kr8` | recompute (copy/alias, re-read distance 8) | k_r=8 | same, root stated 8 lines back |
| `onehop_kc1` | recompute (1 op) | k_c=1 | corrupted `V = U1 op U2`, operands stated adjacent |
| `deep_kc5` | recompute (5 ops) | k_c=5 | corrupted 6-term ±-only site |
| `benign_paraphrase`, `true_interruption` | TRUE-plant controls | — | true value restated / true aside inserted |

## 2. Primary output — the per-model pair

For each model M, report the **(readable-side absorption, recompute-side absorption)** pair:

- **recompute-side absorption** = `final_output_absorbed` (planted computed value carried into the final answer through the downstream re-read), pooled over `opfree_kr1/kr8`, `onehop_kc1`, `deep_kc5`; also reported per cell. `next_read_absorbed` is the secondary channel. For API models the EXPH channels stand in (derivational injection-dependence; per §4).
- **readable-side absorption** = absorption when the false value is re-readable (`adjacent_contradiction` `final_output_absorbed`, with `judge_reject` as the rejection channel).

**The readable side is drawn from existing data where it exists — cite exactly:**

- **Anchor (Qwen2.5-7B), EXPG pilot** (`EXPG_PROGTRACE_PILOT/REPORT.md`, Stage-1 table): readable `adjacent_contradiction` next-read 0.743 [0.60,0.89], final-output 0.686 [0.51,0.83], judge_reject 0.200; recompute `onehop_kc1` final 0.857, `deep_kc5` final 1.000, `opfree_kr1` final 0.829, `opfree_kr8` final 0.914. The pilot pair is (0.686 readable, 0.83–1.00 recompute).
- **EXPD/E1 readable false-premise reference** (`E1_REGISTRATION.md` §6 anchors, `EXPD_MATCHED_GRADIENT`): the d=0 readable cliff — stated-complement rejection `aff_false_attr_d0` 0.247 vs d1 0.031 (i.e. re-readable contradiction is caught ~8× more often than a 1-hop one). This is the cross-substrate readable reference; the EXPG `adjacent_contradiction` cell is the same-substrate readable anchor and is the one paired in the tables.
- **API prefill readable reference** (`EXPH_API_MODELS/EXPH_REPORT.md` §1): on the NL entity-chain substrate, one_hop_falsehood stated-complement = haiku 0.888 / sonnet 0.944 (they explicitly correct the readable falsehood), global_falsehood closure-valid 0.99; derivational injection-dependence 0.000 in all 12 cells. So the existing API readable side is "corrects it," and the EXPG recompute cells are the new measurement for those two models.

Qwen2.5-7B is re-anchored here (fresh EXPG run) so every model, including the anchor, contributes a pair produced by the identical current harness; the pilot numbers above are the expected-value reference, not re-scored into any model's pair.

## 3. Open-weights roster (vLLM, own-trace injection, greedy, HF_HUB_OFFLINE=1)

Attempt all four; **exclude honestly** if gold attrition is catastrophic (the E1 pattern), disclosed with the funnel. Gold oversample **4×** the pilot world budget for every non-anchor model (weaker models under-provision at 1×; kc5 solve was 0.45 even on Qwen-7B — pilot G-A funnel).

| id | Model (HF repo) | role |
|---|---|---|
| qwen7b (anchor) | `Qwen/Qwen2.5-7B-Instruct` @ `a09a3545…` | re-anchor; pilot model |
| qwen1p5b | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798…` | small-scale floor |
| llama8b | `NousResearch/Meta-Llama-3.1-8B-Instruct` @ `d10aef79…` | non-Qwen (ungated mirror, per E1 §3) |
| qwen32b | `Qwen/Qwen2.5-32B-Instruct` @ `5ede1c97…` | scale-within-family |
| olmo7b | `allenai/OLMo-2-1124-7B-Instruct` @ `470b1fba…` | non-Qwen, open recipe |

Revisions pinned to E1's verified cache (`iclr_exec/s7_rowcount/REPORT.md` Part C). Continuation = raw-prefix (no chat template), max_new=192, as EXPG.

## 4. API roster

- **Prefill (assistant-turn continuation, the EXPH regime):** `claude-haiku-4-5`, `claude-sonnet-4-5`. Byte-identical to EXPH prefill parity (`prefill_parity_check.json`); temperature 0, max_tokens 300.
- **Instruct-anchor (chat continuation, disclosed non-equivalence):** `gpt-4.1`, `gpt-4o`. Reported as instruct-mode rows with the EXPH §2 caveat printed (chat re-presentation can substitute re-planning for continuation); these are descriptive generality points, never pooled with prefill or open-weights.
- **Completions endpoint (true single stream):** `gpt-3.5-turbo-instruct` via the legacy completions API — the one API model that gives a genuine single-token-stream continuation with no chat turn boundary; the cleanest API analogue of the open-weights regime.

API DVs are the EXPH channels: derivational injection-dependence (recompute-side absorption), stated-complement (readable-side correction), closure-valid, order-independent plant-dependence (`api_diagnosis.json`). All plants audited FALSE; own-trace golds.

## 5. Exclusion rules (pre-stated)

1. **Row-level:** EXPG/EXPA fail-closed audits unchanged (measured k_r/k_c == designed; plant audited false and collision-free vs σ_true, σ_cf, and every listing literal; MOD-10 wash-out reject). Audit rejections counted, never patched.
2. **Cell-level:** target n≥150/cell after filtering. 100≤n<150 scored+flagged; 50≤n<100 scored, CI-widened, flagged; n<50 that cell is INVALID-underpowered for that model (reported, not scored, counts as "pair not established").
3. **Model-level (catastrophic-attrition, the E1 branch):** after the full 4× gold oversample, if gold-eligible cohort < 25% of target on the recompute cells, the model is **excluded** and reported in the funnel only — no extra gold budget post-commit except by dated addendum before that model's perturbed generation. kc5 is the expected attrition driver; a model may be excluded on kc5 alone while retaining kc1/kr cells (per-cell exclusion, disclosed).
4. **API cost:** hard stop at the program cap (below). If the cap binds mid-roster, prefill models are completed first, then completions, then instruct-anchor; any model cut for budget is disclosed as budget-excluded, not phenomenon-negative.

## 6. What counts as each outcome

- **Asymmetry replicates for model M** iff its recompute-side absorption exceeds its readable-side absorption by a clear margin (recompute final-output absorption ≥ readable + 0.15, readable side being `adjacent_contradiction` on that model or the cited existing readable reference for the two prefill models). Reported per model with cluster-bootstrap CIs (cluster = program family).
- **Program-level reading:** "the readable/recompute asymmetry holds across the roster" iff it replicates on a majority of models that clear §5 exclusion, frontier/API included — reported as a **measured rate on the named roster**, never as a universal law (RECOMPUTE_SCOOP_CHECK §3).
- **A model that catches the recompute plant** (recompute absorption low, e.g. the API prefill zeros) is a positive finding — reported as "checks at this tier," and it makes the asymmetry *smaller*, counting against the program-level claim for that model. No post-hoc reframing.
- Every cell verdict and every funnel is printed; no model or cell dropped silently.

## 7. Budget

API program cap **$100**. Cumulative project ledger currently **$30.81** (ceiling **$150** total). E10 API spend is drawn against the remaining headroom; EXPH-style per-cell costs (haiku ≈ $1.35 / sonnet ≈ $3.13 for a full 12-cell × 150-row pass, `cost_ledger.json`) put a five-model API pass comfortably under the cap, but the §5.4 stop rule binds regardless.
