"""doubt_judge_v2 -- generalized LLM-judge over arbitrary validated_outputs.jsonl.

Generalizes src/doubt_judge.py (which was hard-wired to the legacy results/
runs.jsonl layout) to judge any jsonl of rows given field names. Two verdicts
per row, both from Qwen/Qwen2.5-32B-Instruct via the vLLM backend
(improvement_plan/tooling/vllm_gen.py), greedy:

  (a) verbalized_doubt  -- does the continuation express doubt / hesitation /
      self-correction / flag an inconsistency about earlier reasoning?
      Prompt scaffold kept VERBATIM from the legacy judge (src/doubt_judge.py)
      so calibration against historical judge numbers is meaningful.
  (b) explicit_rejection -- does the continuation verbally flag the planted
      statement OR explicitly state/derive its complement/negation? The judge
      reads the planted statement and checks whether the continuation asserts
      its opposite. NEW in v2 (captures the derivational-rejection channel
      that the doubt question misses).

Input : jsonl path + field names (continuation field, planted-statement field,
        cell/condition field, lexical-doubt field, id fields), optional
        row filter and stratified subsampling.
Output: <out_prefix>_verdicts.jsonl  (one judged row per input row)
        <out_prefix>_summary.json    (per-cell + overall rates for
                                      lexical doubt, judge doubt,
                                      explicit rejection, agreement)

Single job via CLI flags, or several jobs in one engine load via --jobs
jobs.json (a list of dicts whose keys mirror the CLI flags; the 32B engine
is loaded once and reused).

Usage (skampere2):
    cd $EXP/improvement_plan/tooling
    source env.sh                       # caches on /lfs, one GPU
    export HF_HUB_OFFLINE=1             # model already cached; never touch Hub
    CUDA_VISIBLE_DEVICES=<free> .venv-vllm/bin/python doubt_judge_v2.py \
        --jobs judge_v2/jobs.json
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
CONT_TRUNC = 1500  # legacy judge truncated continuations to 1500 chars; kept

# ---------------------------------------------------------------- prompts
# (a) VERBATIM legacy scaffold from src/doubt_judge.py -- do not edit.
PROMPT_DOUBT = """You will see the continuation of a step-by-step logic proof. Decide whether the text EXPRESSES DOUBT, HESITATION, SELF-CORRECTION, or flags an inconsistency about any earlier step (e.g., questioning a premise, noting a contradiction, revising a claim). Mechanical proof steps with no commentary = NO.

Continuation:
---
{cont}
---
Answer with exactly one word: YES or NO."""

# (b) NEW: rejection-of-planted-statement judge (flag OR assert complement).
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
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                d = json.loads(line)
                d["_line_idx"] = i
                rows.append(d)
    return rows


def apply_filter(rows, filt):
    """filt: dict of field -> required value (str compare)."""
    if not filt:
        return rows
    out = []
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in filt.items()):
            out.append(r)
    return out


def stratified_sample(rows, n, stratify_field, seed):
    """Proportional (largest-remainder) allocation over strata; deterministic."""
    if n is None or n >= len(rows):
        return rows
    strata = defaultdict(list)
    for r in rows:
        strata[str(r.get(stratify_field))].append(r)
    keys = sorted(strata)
    quotas = {k: n * len(strata[k]) / len(rows) for k in keys}
    alloc = {k: int(quotas[k]) for k in keys}
    rem = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: quotas[k] - alloc[k], reverse=True)[:rem]:
        alloc[k] += 1
    rng = random.Random(seed)
    picked = []
    for k in keys:
        pool = sorted(strata[k], key=lambda r: r["_line_idx"])
        picked += rng.sample(pool, min(alloc[k], len(pool)))
    picked.sort(key=lambda r: r["_line_idx"])
    return picked


def parse_verdict(text):
    """-> (bool verdict, bool parsed). Unparsed counts as NO but is tallied."""
    t = (text or "").strip().upper()
    if t.startswith("YES"):
        return True, True
    if t.startswith("NO"):
        return False, True
    return False, False


def run_job(job, engine_model, max_new_tokens):
    from vllm_gen import generate_continuations

    name = job["name"]
    cont_f = job.get("continuation_field", "continuation")
    stmt_f = job.get("statement_field", "injected_statement")
    cell_f = job.get("cell_field", "condition")
    lex_f = job.get("lexical_field", "verbalized_doubt")
    id_fs = job.get("id_fields", ["problem_id", "run_id"])
    extra_fs = job.get("extra_rate_fields", [])

    rows = read_jsonl(job["input"])
    rows = apply_filter(rows, job.get("filter"))
    rows = stratified_sample(rows, job.get("sample_n"),
                             job.get("stratify_field", lex_f),
                             job.get("seed", 0))
    n = len(rows)
    print(f"[{name}] {n} rows from {job['input']}", flush=True)

    prompts = []
    for r in rows:
        cont = str(r.get(cont_f, ""))[:CONT_TRUNC]
        stmt = str(r.get(stmt_f, ""))
        prompts.append({"messages": [{"role": "user",
                                      "content": PROMPT_DOUBT.format(cont=cont)}]})
        prompts.append({"messages": [{"role": "user",
                                      "content": PROMPT_REJECT.format(stmt=stmt, cont=cont)}]})

    outs = generate_continuations(engine_model, prompts,
                                  max_new_tokens=max_new_tokens)
    assert len(outs) == 2 * n

    verdicts = []
    agg = defaultdict(lambda: defaultdict(int))
    unparsed = 0
    for i, r in enumerate(rows):
        jd, ok1 = parse_verdict(outs[2 * i]["text"])
        er, ok2 = parse_verdict(outs[2 * i + 1]["text"])
        unparsed += (not ok1) + (not ok2)
        lex = bool(r.get(lex_f))
        cell = str(r.get(cell_f))
        v = {k: r.get(k) for k in id_fs}
        v.update({"_line_idx": r["_line_idx"], "cell": cell,
                  "lexical_doubt": lex, "judge_doubt": jd,
                  "explicit_rejection": er,
                  "judge_doubt_raw": outs[2 * i]["text"].strip(),
                  "explicit_rejection_raw": outs[2 * i + 1]["text"].strip(),
                  "planted_statement": r.get(stmt_f),
                  "continuation_head": str(r.get(cont_f, ""))[:300]})
        for k in extra_fs:
            v[k] = bool(r.get(k))
        verdicts.append(v)
        a = agg[cell]
        a["n"] += 1
        a["lex"] += int(lex)
        a["judge"] += int(jd)
        a["reject"] += int(er)
        a["agree_judge_lex"] += int(jd == lex)
        a["judge_and_lex"] += int(jd and lex)
        a["judge_not_lex"] += int(jd and not lex)
        a["reject_not_judge"] += int(er and not jd)
        for k in extra_fs:
            a[f"extra::{k}"] += int(bool(r.get(k)))

    def cell_summary(a):
        m = a["n"]
        s = {"n": m,
             "lexical_doubt_rate": round(a["lex"] / m, 4),
             "judge_doubt_rate": round(a["judge"] / m, 4),
             "explicit_rejection_rate": round(a["reject"] / m, 4),
             "judge_lex_agreement": round(a["agree_judge_lex"] / m, 4),
             "judge_and_lex": a["judge_and_lex"],
             "judge_not_lex": a["judge_not_lex"],
             "reject_not_judge": a["reject_not_judge"]}
        for k in extra_fs:
            s[f"{k}_rate"] = round(a[f"extra::{k}"] / m, 4)
        return s

    total = defaultdict(int)
    for a in agg.values():
        for k, v_ in a.items():
            total[k] += v_
    summary = {"job": name, "input": job["input"], "n_rows": n,
               "judge_model": engine_model,
               "continuation_field": cont_f, "statement_field": stmt_f,
               "cell_field": cell_f, "lexical_field": lex_f,
               "filter": job.get("filter"), "sample_n": job.get("sample_n"),
               "stratify_field": job.get("stratify_field"),
               "seed": job.get("seed", 0),
               "n_unparsed_verdicts": unparsed,
               "cells": {c: cell_summary(agg[c]) for c in sorted(agg)},
               "overall": cell_summary(total)}

    prefix = job["out_prefix"]
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    with open(prefix + "_verdicts.jsonl", "w") as fh:
        for v in verdicts:
            fh.write(json.dumps(v) + "\n")
    with open(prefix + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{name}] done. unparsed={unparsed}", flush=True)
    for c in sorted(agg):
        s = summary["cells"][c]
        print(f"  {c}: n={s['n']} lex={s['lexical_doubt_rate']} "
              f"judge={s['judge_doubt_rate']} reject={s['explicit_rejection_rate']}",
              flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", default=None,
                    help="JSON file: list of job dicts (keys mirror the flags)")
    ap.add_argument("--input", default=None)
    ap.add_argument("--continuation-field", default="continuation")
    ap.add_argument("--statement-field", default="injected_statement")
    ap.add_argument("--cell-field", default="condition")
    ap.add_argument("--lexical-field", default="verbalized_doubt")
    ap.add_argument("--id-fields", default="problem_id,run_id")
    ap.add_argument("--filter", default=None,
                    help="comma-separated field=value row filter")
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--stratify-field", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    if args.jobs:
        jobs = json.load(open(args.jobs))
    else:
        if not (args.input and args.out_prefix):
            ap.error("need --jobs, or --input + --out-prefix")
        filt = None
        if args.filter:
            filt = dict(kv.split("=", 1) for kv in args.filter.split(","))
        jobs = [{"name": args.name or os.path.basename(args.input),
                 "input": args.input,
                 "continuation_field": args.continuation_field,
                 "statement_field": args.statement_field,
                 "cell_field": args.cell_field,
                 "lexical_field": args.lexical_field,
                 "id_fields": args.id_fields.split(","),
                 "filter": filt, "sample_n": args.sample_n,
                 "stratify_field": args.stratify_field or args.lexical_field,
                 "seed": args.seed, "out_prefix": args.out_prefix}]

    for job in jobs:
        run_job(job, args.judge_model, args.max_new_tokens)
    print("DONE doubt_judge_v2", flush=True)


if __name__ == "__main__":
    main()
