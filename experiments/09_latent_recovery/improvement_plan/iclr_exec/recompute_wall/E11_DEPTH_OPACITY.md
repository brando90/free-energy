# E11 — DEPTH × OPACITY: verification cost as a dose, and shown-work as an antidote

**Status:** LIGHT registration (pre-run). Lead: Elyas Obbad · Drafted 2026-07-27.
**Scope:** single-trace, in-stream continuation on the EXPG program-trace substrate.
**One-line claim under test:** recompute-side absorption rises monotonically with **verification cost** k (ops from the planted computed value to its nearest STATED ancestors) with trace length and everything else held fixed; and making the intermediate work visible next to the value partially restores catching.

Machinery reused: EXPG generator/validator (`improvement_plan/expg/`), extended with a depth-parametric site builder (new work under `recompute_wall/gen_depth.py`, generalizing the existing `kc1`/`kc5` shapes to arbitrary k). API prefill: EXPH harness.

**Cite-and-distinguish (mandatory, per RECOMPUTE_SCOOP_CHECK §4):** *When LLMs Stop Following Steps* (arXiv 2605.00817) already publishes a depth→accuracy curve (63%→20% over 5→95 steps) — but they vary **execution length** (how far the model must itself compute) and measure own-execution fidelity. **We fix trace length and total ops and vary only verification cost k** (how far a *planted* value sits from re-checkable stated ancestors), measuring absorption of a false intermediate. Distinct IV, distinct DV.

---

## 1. The depth axis

Verification depth **k = number of operations from the planted computed value to its nearest STATED ancestors** — how much arithmetic a reader must redo to re-derive and thereby catch the plant. Cells at **k ∈ {0, 1, 2, 3, 5}**.

- **k=0** = the readable-contradiction anchor: the true value is stated verbatim in the listing, so the plant is a re-readable contradiction (the `adjacent_contradiction` construction). This is the floor and the readable reference cell.
- **k≥1**: the value is `V = f(ancestors)` where the shortest route from any stated quantity to V is exactly k ops; checking V requires re-executing those k ops.

**Controls, all generator-audited fail-closed (a world failing any audit is rejected, never patched):**

1. **Trace length matched:** filler branches pad every cell to the same total ops and token count across k, so k varies with L and total compute held constant (this is the whole point vs 2605.00817).
2. **Ancestor token-distance matched:** filler placement equalizes the token distance from V back to its stated ancestors across cells, so k is not confounded with "how far back on the page."
3. **Min-verification-path uniqueness:** the generator computes the min over **all** re-derivation routes to V and audits that it equals the designed k (fail-closed); no shorter side-route exists.
4. **Plant validity:** every planted value audited actually-false, distinct from every value in σ_true and σ_cf and every listing literal, and non-washing (MOD-10).
5. **Usability constant:** the plant is read downstream (≥2 discriminating reads before any overwrite, forward-use gap 2) in every cell, so absorption opportunity is held fixed across k.

## 2. The opacity axis (same worlds)

Three presentations of the planted site, crossed with k on the identical worlds:

- **bare** — value only: `V = 173` (the default EXPG form).
- **full** — value with full worked derivation adjacent: `V = a + b - c + d - e + f = 173`.
- **partial** — value with partial work: the last op or two shown, the rest elided.

## 3. Design and DV

- **Grid:** k{0,1,2,3,5} × opacity{bare,full,partial} = 15 cells/model. (k=0 × opacity is degenerate — the value is already stated — so k=0 is a single readable anchor cell; effective cells ≈ 13/model.)
- **n:** ~150–200 eligible traces/cell after the gold double filter.
- **Sampling:** R=8 at T=0.7 **plus** the greedy rollout, per (world × cell), matching E9's rollout shape.
- **DV — objective 3-way, from validator re-execution of both branches (σ_true and σ_cf), per rollout:**
  - **absorbed** — the continuation's downstream read(s) and final output use the planted (false) value; validator confirms the planted branch was carried.
  - **silently_corrected** — the continuation uses the correct re-derived value with no flag (EXPG `repair_event` / trace re-valid against σ_true without any doubt or rejection marker).
  - **flagged** — the continuation explicitly questions or rejects the value (judge_reject / doubt channel), regardless of which value it then uses.
  Exactly one label per rollout; distribution reported per cell. (Maps onto the EXPG pilot columns `final_output_absorbed` / `repair_event` / `judge_reject`.)

## 4. Models

- **qwen7b anchor** — `Qwen/Qwen2.5-7B-Instruct` @ `a09a3545…` (the pilot model; the depth trend must first exist here).
- **llama8b** — `NousResearch/Meta-Llama-3.1-8B-Instruct` @ `d10aef79…` (non-Qwen open-weights).
- **one Anthropic prefill model** — `claude-haiku-4-5` (EXPH prefill regime; the frontier check — pilot API deriv-absorption was 0.000, so the interesting question is whether *any* k or opacity condition moves it off the floor).

## 5. Primary contrasts

1. **k0-vs-k1 gap, per model:** absorbed-rate(k=1, bare) − absorbed-rate(k=0, bare). The core readable→recompute step (pilot anchor: readable 0.686 → onehop_kc1 0.857).
2. **Monotonic trend over k:** absorbed-rate as a function of k ∈ {0,1,2,3,5} at bare opacity, per model; reported with a monotonicity check (isotonic fit / Spearman) and cell-wise CIs. Pilot anchor endpoints: k0 0.686, kc1 0.857, kc5 1.000.
3. **Opacity restoration effect:** absorbed-rate(bare) − absorbed-rate(full), pooled over k≥1 and per k, per model — does showing the work move rollouts from **absorbed** to **flagged/silently_corrected**? `partial` is the intermediate dose.

All CIs are cluster-bootstrap (cluster = program family); Wilson for single rates. No prose Holm liturgy — contrasts adjudicated at the pre-stated margins below.

## 6. Exclusion rules (pre-stated)

1. **Row-level:** all §1 fail-closed audits; audit rejections counted, never patched.
2. **Cell-level:** target n≥150. 100≤n<150 scored+flagged; 50≤n<100 CI-widened+flagged; n<50 that cell INVALID-underpowered (reported, not scored). The deep + full-opacity cells are the expected attrition drivers (kc5 gold solve was 0.45 on Qwen-7B); 4× gold oversample for non-anchor models.
3. **Model-level:** if a model cannot produce ≥50 eligible traces on the k=5 cells after 4× oversample, its depth trend is reported truncated at the deepest viable k (disclosed), not silently gap-filled.
4. **API cost:** hard stop at the program cap (§8); haiku completed first if the cap binds.

## 7. What counts as each outcome

- **Depth dose-response present for model M** iff (a) the k0→k1 gap is positive beyond CI, AND (b) the absorbed-rate over k is non-decreasing (isotonic) with the k=5 rate strictly above k=0 by ≥0.15. A flat or non-monotonic curve is reported as **no dose-response** for M (a positive scientific result — that model's checking does not degrade with depth).
- **Opacity restores catching for model M** iff full-opacity absorbed-rate is below bare by ≥0.15 (pooled k≥1), with the displaced mass landing in flagged/silently_corrected. A null opacity effect is reported as such.
- **Monotonicity and k0-vs-k1 are the headline; opacity is the mechanism probe.** Reported as measured rates on the named three-model roster, not as a law.
- Every cell's 3-way distribution and funnel printed; no cell dropped silently.

## 8. Budget

API program cap **$100**; cumulative project ledger **$30.81**, ceiling **$150** total. E11 API spend is one prefill model (haiku) over ≈13 cells × ~175 rows × (R=8+greedy) — the dominant cost line; drawn against remaining headroom under the §6.4 stop rule.
