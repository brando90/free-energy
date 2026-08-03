"""E7 prep: extract + format the two judge_v2 completion-pass row sets.

Row set (a): EXPD_MATCHED_GRADIENT full grid -- ALL rows (the pre-registered
             SECONDARY judge DV was never run over this experiment; see
             EXPD_FULL_REPORT.md L205: "it was not run in this pass").
Row set (b): EXPB_LOCAL_CERT_FLIP rows with class == "unparsed" -- the 118
             rows the mechanical strict validator could not parse (W3: raises
             unparsed 4.0% -> 31.7% under LOCAL_CERTIFICATE). The 32B judge
             gives these an informed verdict where the mechanical validator
             gave none.

CPU only. Writes materialized input jsonl copies under
  $EXP/improvement_plan/iclr_exec/e7_judge/inputs/
(never touches results/ -- read-only source), a doubt_judge_v2-compatible
jobs_e7.json pointing at them, and a prompt-preview file for 5 sample rows
(2-3 from each set) built with the PROMPT_DOUBT / PROMPT_REJECT scaffolds
copied VERBATIM from doubt_judge_v2.py so the preview is byte-identical to
what the judge will actually see.

Usage (CPU, no GPU env needed):
    /lfs/skampere2/0/eobbad/free-energy/.venv/bin/python prepare_inputs.py
"""
import json
import os

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
OUT_ROOT = os.path.join(EXP, "improvement_plan/iclr_exec/e7_judge")
INPUTS_DIR = os.path.join(OUT_ROOT, "inputs")
OUTPUTS_DIR = os.path.join(OUT_ROOT, "outputs")

EXPD_SRC = os.path.join(EXP, "results/EXPD_MATCHED_GRADIENT/validated_outputs.jsonl")
EXPB_SRC = os.path.join(EXP, "results/EXPB_LOCAL_CERT_FLIP/validated_outputs.jsonl")

# --- prompt scaffolds, copied VERBATIM from
# improvement_plan/tooling/judge_v2/doubt_judge_v2.py -- do not edit. ---
CONT_TRUNC = 1500

PROMPT_DOUBT = """You will see the continuation of a step-by-step logic proof. Decide whether the text EXPRESSES DOUBT, HESITATION, SELF-CORRECTION, or flags an inconsistency about any earlier step (e.g., questioning a premise, noting a contradiction, revising a claim). Mechanical proof steps with no commentary = NO.

Continuation:
---
{cont}
---
Answer with exactly one word: YES or NO."""

PROMPT_REJECT = """A statement was planted into the middle of a step-by-step logic proof about fictional creatures. You will see the planted statement and the continuation of the proof that follows it. Decide whether the continuation REJECTS the planted statement. Answer YES if the continuation does EITHER of the following:
(1) verbally flags, questions, doubts, or comments on the planted statement (e.g., noting it is wrong, inconsistent, or contradicts the premises), OR
(2) explicitly states or derives the OPPOSITE (negation/complement) of the planted statement -- e.g., the planted statement says "Stella is shiny" and the continuation asserts "Stella is not shiny", or the planted statement says "Sam is a numpus" and the continuation asserts "Sam is not a numpus".
Answer NO if the continuation merely ignores the planted statement, repeats it, builds on it, or consists of proof steps that never contradict it.

Planted statement:
---
{stmt}
---
Continuation:
---
{cont}
---
Answer with exactly one word: YES or NO."""


def read_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main():
    os.makedirs(INPUTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    report_lines = []

    def log(s):
        print(s)
        report_lines.append(s)

    # ---- (a) EXPD full grid: all rows ----
    expd_rows = read_jsonl(EXPD_SRC)
    log(f"[EXPD] read {len(expd_rows)} rows from {EXPD_SRC}")
    assert len(expd_rows) == 11577, (
        f"expected 11577 EXPD rows per RESULTS_LOCK / EXPD_FULL_REPORT.md, got {len(expd_rows)}"
    )
    for r in expd_rows:
        assert not r["failed_generation"] and r.get("continuation"), (
            f"unexpected empty/failed EXPD row {r.get('problem_id')}/{r.get('run_id')}"
        )
    expd_out = os.path.join(INPUTS_DIR, "expd_full_grid.jsonl")
    write_jsonl(expd_out, expd_rows)
    log(f"[EXPD] wrote {len(expd_rows)} rows -> {expd_out}")

    # ---- (b) EXPB unparsed rows: class == "unparsed" ----
    expb_rows = read_jsonl(EXPB_SRC)
    expb_unparsed = [r for r in expb_rows if r.get("class") == "unparsed"]
    log(f"[EXPB] read {len(expb_rows)} total rows from {EXPB_SRC}; "
        f"{len(expb_unparsed)} have class=='unparsed'")
    assert len(expb_unparsed) == 118, (
        f"expected 118 unparsed EXPB rows per W3 (unparsed 4.0%->31.7%), got {len(expb_unparsed)}"
    )
    for r in expb_unparsed:
        assert not r["failed_generation"] and r.get("continuation"), (
            f"unexpected empty/failed EXPB unparsed row {r.get('problem_id')}/{r.get('run_id')}"
        )
    # per-condition breakdown (cross-check against W3's 4.0% -> 31.7%)
    from collections import Counter
    cond_counts = Counter(r["condition"] for r in expb_unparsed)
    cond_totals = Counter(r["condition"] for r in expb_rows)
    for cond in sorted(cond_totals):
        n_unp = cond_counts.get(cond, 0)
        n_tot = cond_totals[cond]
        log(f"  {cond}: {n_unp}/{n_tot} unparsed = {n_unp/n_tot:.1%}")
    expb_out = os.path.join(INPUTS_DIR, "expb_unparsed.jsonl")
    write_jsonl(expb_out, expb_unparsed)
    log(f"[EXPB] wrote {len(expb_unparsed)} rows -> {expb_out}")

    # ---- jobs_e7.json (doubt_judge_v2 --jobs format) ----
    jobs = [
        {
            "name": "EXPD_MATCHED_GRADIENT_FULL",
            "input": expd_out,
            "continuation_field": "continuation",
            "statement_field": "injected_statement",
            "cell_field": "condition",
            "lexical_field": "verbalized_doubt",
            "id_fields": ["problem_id", "run_id"],
            "out_prefix": os.path.join(OUTPUTS_DIR, "expd_full"),
        },
        {
            "name": "EXPB_UNPARSED_118",
            "input": expb_out,
            "continuation_field": "continuation",
            "statement_field": "injected_statement",
            "cell_field": "condition",
            "lexical_field": "verbalized_doubt",
            "id_fields": ["problem_id", "run_id"],
            "out_prefix": os.path.join(OUTPUTS_DIR, "expb_unparsed"),
        },
    ]
    jobs_path = os.path.join(OUT_ROOT, "jobs_e7.json")
    with open(jobs_path, "w") as fh:
        json.dump(jobs, fh, indent=2)
    log(f"[jobs] wrote {jobs_path}")
    log(f"[jobs] total rows across both jobs: {len(expd_rows) + len(expb_unparsed)} "
        f"({len(expd_rows) + len(expb_unparsed)} rows x 2 prompts/row = "
        f"{2*(len(expd_rows) + len(expb_unparsed))} judge calls)")

    # ---- prompt-construction dry run: 5 sample rows (3 EXPD, 2 EXPB) ----
    samples = []
    # spread EXPD samples across distinct conditions for a representative preview
    seen_conds = set()
    for r in expd_rows:
        if r["condition"] not in seen_conds:
            samples.append(("EXPD", r))
            seen_conds.add(r["condition"])
        if len(samples) == 3:
            break
    for r in expb_unparsed[:2]:
        samples.append(("EXPB", r))

    preview_path = os.path.join(OUT_ROOT, "prompt_preview_5rows.txt")
    with open(preview_path, "w") as fh:
        for i, (src, r) in enumerate(samples):
            cont = str(r.get("continuation", ""))[:CONT_TRUNC]
            stmt = str(r.get("injected_statement", ""))
            header = (f"=== sample {i+1}/5  src={src}  condition={r.get('condition')}  "
                      f"problem_id={r.get('problem_id')}  run_id={r.get('run_id')} ===\n")
            doubt_prompt = PROMPT_DOUBT.format(cont=cont)
            reject_prompt = PROMPT_REJECT.format(stmt=stmt, cont=cont)
            fh.write(header)
            fh.write("--- PROMPT_DOUBT ---\n" + doubt_prompt + "\n\n")
            fh.write("--- PROMPT_REJECT ---\n" + reject_prompt + "\n\n\n")
    log(f"[preview] wrote 5-sample prompt preview -> {preview_path}")

    # ---- output-path wiring check ----
    log(f"[wiring] inputs dir exists: {os.path.isdir(INPUTS_DIR)} ({INPUTS_DIR})")
    log(f"[wiring] outputs dir exists: {os.path.isdir(OUTPUTS_DIR)} ({OUTPUTS_DIR})")
    log(f"[wiring] expd out_prefix -> {OUTPUTS_DIR}/expd_full_verdicts.jsonl + _summary.json")
    log(f"[wiring] expb out_prefix -> {OUTPUTS_DIR}/expb_unparsed_verdicts.jsonl + _summary.json")

    summary_path = os.path.join(OUT_ROOT, "prepare_inputs_report.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(report_lines) + "\n")
    log(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
