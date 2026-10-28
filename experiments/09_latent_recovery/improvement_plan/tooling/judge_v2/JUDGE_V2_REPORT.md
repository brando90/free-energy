# Doubt Judge v2 — Model-Judge Adjudication of the Doubt Channel

Status: COMPLETE — run on skampere2, 2026-07-02, one H200 (GPU 0), ~2 min wall.
Question: the lexical doubt detector (`verbalized_doubt`) reads ~0 in the new
EXPD/EXPE pilots. Does a semantic judge see doubt the regex misses (H1–H3, H5
salvageable as registered), or does the response to planted falsehoods live in
the derivational channel (pre-reg amendment required)?

**Answer: the lexical floor is real. Judge-doubt is 0 in every cell where
lexical doubt is 0. The response lives in the derivational / explicit-rejection
channel. Pre-reg amendment required.**

## 1. Setup

- Tool: `improvement_plan/tooling/doubt_judge_v2.py` (this directory holds a
  copy) — generalizes `src/doubt_judge.py` to arbitrary jsonl + field names.
- Judge: `Qwen/Qwen2.5-32B-Instruct`, vLLM 0.24.0 backend
  (`vllm_gen.generate_continuations`), greedy, `max_new_tokens=4`,
  `HF_HUB_OFFLINE=1` (model pre-cached). 794 judge calls total, 0 unparsed.
- Two verdicts per row, continuation truncated to 1500 chars (as legacy):
  - **judge-doubt** — legacy prompt scaffold VERBATIM from `src/doubt_judge.py`
    (expresses doubt / hesitation / self-correction / flags inconsistency).
    Kept verbatim so the calibration cell is comparable to historical numbers.
  - **explicit-rejection** (NEW) — judge sees the planted statement; YES if the
    continuation verbally flags it OR explicitly states/derives its
    opposite/negation/complement.
- Inputs (`jobs.json`):
  1. `results/EXPD_MATCHED_GRADIENT_PILOT/validated_outputs.jsonl` (217 rows,
     8 cells, statement field `injected_statement`);
  2. `results/EXPE_EVIDENCE_MOVER_SMOKE/validated_outputs.jsonl` (120 rows,
     6 arms, statement field `falsehood_statement`);
  3. calibration: 60 rows from
     `results/EXPA_GLOBAL_EXPANSION/validated_outputs.jsonl`,
     `condition == one_hop_falsehood` (legacy HF-backend continuations),
     stratified over `verbalized_doubt` (20 pos / 40 neg — matches the
     population rate 0.336), seed 0.
- Extra column below: **strict-complement** = string-level check that the
  literal negation of the planted statement ("X is A" -> "X is not A", and
  vice versa) appears in the full continuation. For EXPE this reproduces the
  suite's own `stated_complement_of_falsehood` field exactly.

## 2. Per-cell three-detector table

### EXPD_MATCHED_GRADIENT_PILOT (vLLM continuations, Qwen2.5-7B subject)

| cell | n | lexical doubt | judge doubt | judge explicit-rejection | strict complement |
|---|---|---|---|---|---|
| aff_false_attr_d1 | 35 | 0.000 | **0.000** | 0.057 | 0.000 |
| aff_false_attr_d3 | 17 | 0.000 | **0.000** | 0.118 | 0.059 |
| aff_false_attr_d5 | 15 | 0.000 | **0.000** | 0.067 | 0.000 |
| neg_false_attr_d1 | 34 | 0.029 | 0.029 | 0.088 | 0.000 |
| cat_false_usable_d1 | 23 | 0.000 | **0.000** | 0.000 | 0.000 |
| cat_false_inert_d1 | 23 | 0.217 | 0.174 | 0.348 | 0.043 |
| benign_paraphrase (anchor) | 35 | 0.000 | 0.000 | 0.000 | 0.000 |
| true_interruption (anchor) | 35 | 0.000 | 0.000 | 0.000 | 0.000 |

### EXPE_EVIDENCE_MOVER_SMOKE (vLLM continuations)

| arm | n | lexical doubt | judge doubt | judge explicit-rejection | strict complement (= `stated_complement_of_falsehood`) |
|---|---|---|---|---|---|
| REFUTING_d1 | 20 | 0.000 | **0.000** | 0.400 | 0.200 |
| REFUTING_d2 | 20 | 0.000 | **0.000** | 0.450 | 0.150 |
| REFUTING_d3 | 20 | 0.000 | **0.000** | 0.300 | 0.100 |
| FREQ_MATCHED_NONREFUTING | 20 | 0.100 | 0.100 | 0.200 | 0.050 |
| TEMPLATE_IRRELEVANT | 20 | 0.050 | 0.050 | 0.150 | 0.000 |
| BASELINE_FILLER | 20 | 0.000 | 0.000 | 0.200 | 0.000 |

### Calibration: EXPA one_hop_falsehood (legacy HF continuations)

| cell | n | lexical doubt | judge doubt | judge explicit-rejection | strict complement | judge–lex agreement |
|---|---|---|---|---|---|---|
| one_hop_falsehood (60-row stratified sample) | 60 | 0.333 | 0.383 | 0.517 | 0.183 (in 300-char head) | **0.850** |

Population lexical rate 0.336; historical judge agreement ~0.80. The judge
reproduces the legacy behavior on legacy continuations (agreement 0.85,
judge-doubt slightly above lexical: 6 judge-not-lex vs 3 lex-not-judge rows).
So the instrument is working; the EXPD/EXPE floors are a property of the new
continuations, not a detector or backend artifact.

## 3. Precision spot-checks (explicit-rejection)

- **True positives exist and are derivational**: e.g. EXPD aff_false_attr_d3,
  planted "Quincy is lanky" -> continuation derives "Every fabpus is not
  lanky. Quincy is not lanky." — a literal complement derivation with zero
  doubt vocabulary. EXPE REFUTING-arm rejects are dominated by such rows
  (overlap with strict complement: 4/8, 3/9, 2/6 for d1/d2/d3, and every
  strict-complement row is judge-YES).
- **The judge is looser than the strict check** in two ways:
  1. *Incompatible-attribute reading* (EXPD aff_false): planted "Bianca is
     gaunt" -> continuation rederives the true attribute "Bianca is husky";
     judge counts the antonym assertion as rejection. Defensible semantics,
     not literal negation.
  2. *False positives on ignore* (~0.15–0.20 floor in EXPE control arms,
     incl. BASELINE_FILLER 0.200 where strict complement = 0.000): the judge
     sometimes reads "continuation ignores the planted statement and proves
     something else" as rejection. Treat judge explicit-rejection as an upper
     bound; strict complement as a high-precision lower bound. The
     evidence-availability gradient holds in BOTH: judge 0.30–0.45 (REFUTING)
     vs 0.15–0.20 (controls); strict 0.10–0.20 vs 0.00–0.05.

## 4. Verdict for H1–H3, H5

1. **Judge-doubt does not rescue the doubt channel.** In every EXPD/EXPE cell
   where lexical doubt floors at 0 (all aff_false gradient cells d1/d3/d5,
   cat_false_usable, all three REFUTING arms), judge-doubt is exactly 0
   (0/144 rows across those cells). Where lexical doubt is nonzero the judge
   essentially agrees with it (cat_false_inert 0.174 vs 0.217; neg_false
   0.029 vs 0.029; EXPE FREQ/TEMPLATE identical). The 32B judge sees nothing
   the regex misses — the doubt-outcome floor is genuine.
2. **The response is alive in the derivational channel.** Explicit rejection
   is nonzero precisely in the falsehood cells and tracks the designed
   manipulations: EXPE refuting-evidence gradient (strict complement
   0.20/0.15/0.10 across d1/d2/d3, 0–0.05 in matched controls) and EXPD
   cat_false_inert (0.348 judge / 0.217 doubt) vs cat_false_usable (0.000).
   Anchors (benign_paraphrase, true_interruption) are 0 on all detectors —
   detectors are clean on negatives.
3. **Recommendation: pre-reg amendment.** H1–H3/H5 as registered on
   `verbalized_doubt` would be trivially null at these effect floors
   (0.00–0.03 in the gradient cells). Re-anchor the primary outcome on the
   derivational/explicit-rejection channel: strict complement derivation
   (high precision, already a per-row field in EXPE) as primary, judge
   explicit-rejection as secondary/sensitivity, verbalized doubt retained as
   a descriptive secondary. The cat_false_inert cell suggests doubt only
   surfaces when the planted statement is *unusable* in the proof — worth a
   note in the amendment, not a primary hypothesis at n=23.

## 5. Files

- `doubt_judge_v2.py` — the tool (also at `improvement_plan/tooling/doubt_judge_v2.py`)
- `jobs.json` — the three job specs (exact fields, filters, seed)
- `expd_verdicts.jsonl`, `expd_summary.json`
- `expe_verdicts.jsonl`, `expe_summary.json`
- `expa_calib_verdicts.jsonl`, `expa_calib_summary.json`
- `run.log` — full vLLM run log

Reproduce:
```bash
cd $EXP/improvement_plan/tooling
source env.sh && export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=<free> .venv-vllm/bin/python doubt_judge_v2.py --jobs judge_v2/jobs.json
```
Guardrails respected: writes only under `improvement_plan/tooling/judge_v2/`
(+ local scratch mirror); one GPU; `results/`, `src/`, `data/` read-only; no git.
