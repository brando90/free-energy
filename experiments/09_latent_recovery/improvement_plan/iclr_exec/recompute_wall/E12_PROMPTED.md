# E12 — PROMPTED: does an explicit verify-instruction breach the recompute wall?

**Status:** LIGHT registration (pre-run). Lead: Elyas Obbad · Drafted 2026-07-27.
**Scope:** single-trace, in-stream continuation on the EXPG program-trace substrate.
**One-line claim under test:** the recompute-side absorption seen under a neutral instruction is (or is not) removed by an explicit standing instruction to verify each computed value before continuing — i.e. is the wall a *propensity* gap that a prompt closes, or a deeper limitation?

This is the propensity-vs-capability guardrail made into an experiment (RECOMPUTE_SCOOP_CHECK §3, capability-vs-propensity): the nearest strong numbers in the literature (ReFlect 2605.05737, VAIR 2606.01462) are all **prompted**; E12 measures exactly what prompting buys on our substrate so the paper can say "unprompted" with a measured contrast behind it.

Machinery reused: **E9's crossed verification-pressure factor** verbatim — `iclr_exec/e9_motive/e9_run.py` `VERIF_INSTR` dict and the `--verification low high` rollout path (§3.1 of `E9_REGISTRATION.md`). Substrate/cells from EXPG.

---

## 1. What will be run

The two recompute cells of interest, run under a crossed instruction factor, fully within-world (both instruction levels on the identical eligible worlds and golds):

- **Cells:** `onehop_kc1` (k=1, the shallow recompute cell) and `deep_kc5` (k=5, the deep cell). Optionally `adjacent_contradiction` (k=0 readable) as a ceiling reference — a prompt should not need to fix what is already re-readable.
- **Instruction factor (crossed, from E9):**
  - **neutral (low):** the base EXPG instruction, unchanged (`VERIF_INSTR["low"] = ""`).
  - **verify (high):** E9's high-verification prefix, adapted from fact-derivability to computed-value framing: *"Before continuing, verify each computed value by re-deriving it from the earlier lines."* (E9 ships the fact-form string `"Before answering, verify that each supplied fact is derivable from the rules."` in `VERIF_INSTR["high"]`; E12 registers the value-form variant and pins the exact bytes in the run manifest before generation.)
- **Sampling:** R=8 at T=0.7 + greedy per (world × cell × instruction), matching E9/E11.

## 2. DV

Same objective 3-way label as E11, from validator re-execution of both branches per rollout: **absorbed** / **silently_corrected** / **flagged**. Exactly one per rollout; distribution reported per (cell × instruction).

## 3. Models

- **qwen7b anchor** — `Qwen/Qwen2.5-7B-Instruct` @ `a09a3545…`.
- **llama8b** — `NousResearch/Meta-Llama-3.1-8B-Instruct` @ `d10aef79…`.
- **one Anthropic prefill model** — `claude-haiku-4-5` (EXPH prefill regime).

## 4. Primary contrast

**Instruction breach = absorbed-rate(neutral) − absorbed-rate(verify), per cell per model.** A positive, CI-clear drop means the instruction breaches the wall (the gap was propensity, not capability). Reported per cell so the depth interaction is visible:

- **k=1 (`onehop_kc1`):** does "verify" move the shallow recompute plant from absorbed toward flagged/silently_corrected? (neutral anchor: pilot absorbed ≈ 0.857.)
- **k=5 (`deep_kc5`):** does it survive depth? A prompt that fixes k=1 but not k=5 is the informative outcome — instruction helps only where re-derivation is cheap. (neutral anchor: pilot absorbed ≈ 1.000.)

Secondary: where the displaced mass lands (flagged vs silently_corrected) — does the instruction produce visible checking or silent correction?

## 5. Exclusion rules (pre-stated)

1. **Row-level:** EXPG fail-closed audits unchanged; the instruction changes only the prompt prefix, not the world, so both instruction levels share the same audited gold cohort (within-world pairing; a world enters only if eligible under the double filter). Audit rejections counted, never patched.
2. **Cell-level:** target n≥150/cell/instruction. 100≤n<150 scored+flagged; 50≤n<100 CI-widened; n<50 INVALID-underpowered (reported, not scored). 4× gold oversample for non-anchor models; kc5 is the attrition driver.
3. **Parse-rate guard:** the verify instruction can change output format/length; if parse rate differs between instruction levels by >5pp within a cell, report dual-denominator with best/worst-case bounds and mark the contrast instrument-limited if a bound crosses zero.
4. **API cost:** hard stop at the program cap (§7).

## 6. What counts as each outcome

- **Instruction breaches the wall for model M at cell c** iff absorbed-rate drops by ≥0.15 from neutral to verify with the one-sided CI clear of zero. Reported per (M, c).
- **Wall holds against prompting for M at c** iff the drop is null or below margin — a positive finding: the recompute limitation is not a mere propensity a standing instruction fixes (the paper's "unprompted" framing is then the weaker, capability-adjacent claim for that model/cell, disclosed).
- **Depth interaction:** if verify breaches at k=1 but not k=5 on a model, that is the headline for that model — prompting rescues cheap re-derivation only. Reported explicitly, not averaged away.
- All rates are measured on the named three-model roster; every (cell × instruction) distribution and funnel printed; nothing dropped silently.

## 7. Budget

API program cap **$100**; cumulative project ledger **$30.81**, ceiling **$150** total. E12 API spend is one prefill model (haiku) over 2 cells × 2 instruction levels × ~175 rows × (R=8+greedy), drawn against remaining headroom under the §5.4 stop rule.
