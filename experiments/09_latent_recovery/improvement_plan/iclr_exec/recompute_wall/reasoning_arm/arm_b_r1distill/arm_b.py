"""ARM B -- DeepSeek-R1-Distill-Qwen-7B on the recompute-wall computed cells,
with a VISIBLE think channel. Local vLLM on one free skampere1 GPU.

Same worlds as ARM A / the E10 bridge (regime2 programs.jsonl), three cells:
adjacent_contradiction (readable), onehop_kc1 (1-op), deep_kc5 (5-op).

GOLD: own-trace, RAW completion seeded with GOLD_PREFILL="line 1:" (no think
tags -- the mid-line seed keeps the model tracing rather than opening <think>).
4x oversample (1 greedy + 3 sampled); keep the first solved; report attrition.

B1  continuation, RAW completion, NO think tags: user render + injected prefix
    as one token stream; the model continues the visible trace. Standard
    continuation regime, comparable to the open-weight roster. n>=100/cell, R=4.

B2  continuation, CHAT template WITH <think>: the injected partial trace is
    presented in the user turn (same instruction as the bridge); R1's template
    opens <think>, so the model deliberates before continuing. n>=60/cell, R=4.
    Final answer (post-</think>) graded by classify_run (re-execution);
    the raw think text is mechanically scanned for re-derivation of the plant.

All grading is the SAME EXPG two-world validator as every other arm.

Usage: arm_b.py [--gpu 7] [--reps 4] [--b2-max-tokens 3000]
Single process: loads the model once, runs gold -> B1 -> B2, writes JSONL + a
summary_arm_b.json. Resumable via the raw JSONL files.
"""
import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CT = "/var/tmp/eobbad/recompute_wall/free-energy/experiments/09_latent_recovery/improvement_plan"
for p in (os.path.join(CT, "expg"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import trace_format as tf        # noqa: E402
import inject as inj_mod         # noqa: E402
import validate as val_mod       # noqa: E402
import gen_programs as gp        # noqa: E402
from interp import parse_program  # noqa: E402
import rederive_scan as RS        # noqa: E402

MODEL_PATH = ("/lfs/skampere1/0/shared_hf_cache/models--deepseek-ai--"
              "DeepSeek-R1-Distill-Qwen-7B/snapshots/"
              "09db661939bdd5a289fea1b3c7cbd4c555562bca")
PROGRAMS = os.path.join(HERE, "programs.jsonl")
OUT_DIR = os.path.join(HERE, "out")
CELL_SHAPE = {"adjacent_contradiction": "kr1", "onehop_kc1": "kc1", "deep_kc5": "kc5"}
CELLS = ("adjacent_contradiction", "onehop_kc1", "deep_kc5")
DS_EOS = "<｜end▁of▁sentence｜>"  # DeepSeek end-of-sentence token
OUTPUT_LINE_RE = re.compile(r"print\(out\)|output\s*=")


def now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def sha(parts):
    return hashlib.sha1("|".join(str(x) for x in parts).encode()).hexdigest()[:24]


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def clip_trace(text):
    """Keep trace lines through the first output-claim line; drop the rest
    (R1 tends to keep talking after the answer). Format-preserving."""
    if not text:
        return text
    lines = text.split("\n")
    out = []
    for ln in lines:
        out.append(ln)
        if OUTPUT_LINE_RE.search(ln):
            break
    return "\n".join(out)


def programs_by_shape():
    progs = [json.loads(l) for l in open(PROGRAMS)]
    by = {}
    for p in progs:
        by.setdefault(p["shape"], []).append(p)
    return {p["program_id"]: p for p in progs}, by


# ---------------------------------------------------------------- vLLM
def make_llm():
    from vllm import LLM
    return LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=8192, enforce_eager=False, trust_remote_code=True)


def gen_raw(llm, prompts, n, temperature, max_tokens, seed=0):
    """Raw completion generate. Returns list (per prompt) of list-of-n texts."""
    from vllm import SamplingParams
    sp = SamplingParams(n=n, temperature=temperature, top_p=0.95,
                        max_tokens=max_tokens, seed=seed,
                        stop=[DS_EOS, "\nQ:", "\nProgram:"])
    outs = llm.generate(prompts, sp)
    return [[o.text for o in r.outputs] for r in outs]


# ---------------------------------------------------------------- stages
def stage_gold(llm, by_shape, reps_over=4):
    """Own-trace gold via the model's OWN think channel (R1-Distill needs its
    <think> channel to reliably solve the multi-op kc5 traces; raw-completion
    gold yields ~0 on kc5). Gold is still the model's own solution -> the prefix
    shown in B1/B2 stays in-distribution. Post-</think> trace lines are the gold.
    4x oversample; keep first solved; attrition reported."""
    raw_path = os.path.join(OUT_DIR, "gold_raw.jsonl")
    coh_path = os.path.join(OUT_DIR, "gold_cohort.json")
    want_shapes = sorted(set(CELL_SHAPE.values()))
    progs = [p for s in want_shapes for p in by_shape.get(s, [])]
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
                  [{"role": "user", "content": tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))}],
                  tokenize=False, add_generation_prompt=True)
               for p in progs]
    greedy = gen_raw(llm, prompts, 1, 0.0, 3200, seed=0)
    sampled = gen_raw(llm, prompts, reps_over - 1, 0.6, 3200, seed=1)
    cohort = {}
    rawrows = []
    for i, p in enumerate(progs):
        cands = greedy[i] + sampled[i]
        chosen = None
        evals = []
        for c in cands:
            after = c.partition("</think>")[2] if "</think>" in c else ""
            gfull = clip_trace(after)
            ev = inj_mod.gold_solve_eval(p, gfull)
            evals.append({"solved": ev["solved"], "reason": (ev["first_error"] or {}).get("type"),
                          "think_closed": "</think>" in c})
            if ev["solved"] and chosen is None:
                chosen = gfull
        rawrows.append({"program_id": p["program_id"], "shape": p["shape"],
                        "n_cands": len(cands), "solved": chosen is not None,
                        "gold_full_text": chosen, "cand_evals": evals})
        if chosen is not None:
            cohort[p["program_id"]] = chosen
    write_jsonl(raw_path, rawrows)
    per_shape = {}
    for s in want_shapes:
        rs = [r for r in rawrows if r["shape"] == s]
        per_shape[s] = {"n": len(rs), "solved": sum(r["solved"] for r in rs),
                        "solve_rate": round(sum(r["solved"] for r in rs) / len(rs), 4) if rs else None}
    write_json(coh_path, {"n_programs": len(rawrows), "n_solved": len(cohort),
                          "by_shape": per_shape, "eligible_ids": sorted(cohort.keys())})
    print("[gold] solved %d/%d  by_shape=%s" % (len(cohort), len(rawrows), json.dumps(per_shape)))
    return cohort


def build_manifest(progs, by_shape, cohort):
    """One injection per (cell, eligible world of the matching shape)."""
    man = []
    for cell in CELLS:
        shape = CELL_SHAPE[cell]
        for p in by_shape.get(shape, []):
            pid = p["program_id"]
            if pid not in cohort:
                continue
            try:
                b = inj_mod.build_injection(p, cell, cohort[pid])
            except inj_mod.InjectError:
                continue
            row = {"program_id": pid, "cell": cell, "shape": shape}
            row.update({k: b[k] for k in ("prefix_text", "cf", "planted_var",
                        "planted_value", "true_value", "force_after_line",
                        "expected_lines_from", "injected_trace_line")})
            man.append(row)
    return man


def grade(prog, m, continuation):
    inj = {"cf": m["cf"], "planted_var": m["planted_var"],
           "planted_value": m["planted_value"], "force_after_line": m["force_after_line"],
           "expected_lines_from": m["expected_lines_from"]}
    return val_mod.classify_run(parse_program(prog["stmt_texts"]), prog, inj, continuation)


def stage_b1(llm, progs, manifest, reps):
    raw_path = os.path.join(OUT_DIR, "b1_validated.jsonl")
    prompts, meta = [], []
    for m in manifest:
        p = progs[m["program_id"]]
        prompt = tf.make_user_message(tf.make_listing_text(p["stmt_texts"])) + "\n" + m["prefix_text"]
        prompts.append(prompt)
        meta.append(m)
    gens = gen_raw(llm, prompts, reps, 0.6, 640, seed=11)
    rows = []
    for i, m in enumerate(meta):
        p = progs[m["program_id"]]
        for rep, cont in enumerate(gens[i]):
            clipped = clip_trace(cont)
            met = grade(p, m, clipped)
            rec = dict(m)
            rec.update({"protocol": "B1", "rep": rep, "continuation": clipped,
                        "raw_continuation": cont, **met})
            rows.append(rec)
    write_jsonl(raw_path, rows)
    print("[B1] rows=%d (worlds=%d reps=%d)" % (len(rows), len(meta), reps))
    return rows


def stage_b2(llm, progs, manifest, reps, max_tokens):
    raw_path = os.path.join(OUT_DIR, "b2_validated.jsonl")
    tok = llm.get_tokenizer()
    prompts, meta = [], []
    for m in manifest:
        p = progs[m["program_id"]]
        listing = tf.make_listing_text(p["stmt_texts"])
        user = (tf.INSTRUCTION + tf.EXEMPLARS + "Program:\n" + listing +
                "\n\nA partial trace of this program has already been written; it is "
                "shown below under \"Trace:\". Continue it. Write only the remaining "
                "trace lines, in order, in the exact same format, picking up "
                "immediately after the last line shown. Do not repeat lines that are "
                "already written.\n\nTrace:\n" + m["prefix_text"].rstrip("\n"))
        chat = tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        prompts.append(chat)
        meta.append(m)
    gens = gen_raw(llm, prompts, reps, 0.6, max_tokens, seed=21)
    rows = []
    for i, m in enumerate(meta):
        p = progs[m["program_id"]]
        for rep, full in enumerate(gens[i]):
            if "</think>" in full:
                think, _, after = full.partition("</think>")
                think_truncated = False
            else:
                think, after, think_truncated = full, "", True
            cont = clip_trace(after)
            met = grade(p, m, cont)
            rec = dict(m)
            rec.update({"protocol": "B2", "rep": rep, "think_truncated": think_truncated,
                        "think_chars": len(think), "continuation": cont, **met})
            if m["cf"]:
                sc = RS.scan(think, m["planted_var"], m["planted_value"], m["true_value"])
                rec["rederive"] = sc
                rec["rederive_bucket"] = RS.taxonomy_bucket(sc, met.get("final_output_absorbed"))
                rec["rederive_bucket_loose"] = RS.taxonomy_bucket_loose(sc, met.get("final_output_absorbed"))
            rec["think_text"] = think[:4000]
            rows.append(rec)
    write_jsonl(raw_path, rows)
    print("[B2] rows=%d (worlds=%d reps=%d)" % (len(rows), len(meta), reps))
    return rows


def _rate(k, n):
    return {"count": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def cell_summary(rows):
    live = [r for r in rows if not r.get("no_output_claim", False) or True]
    cf_rows = [r for r in rows if r.get("cf")]
    fo = [r for r in cf_rows if r.get("final_output_absorbed") is not None]
    absorbed = silent = flagged = other = 0
    for r in cf_rows:
        if r.get("doubt_lex"):
            flagged += 1
        elif r.get("final_output_absorbed"):
            absorbed += 1
        elif r.get("final_output_valid"):
            silent += 1
        else:
            other += 1
    m = len(cf_rows)
    return {"n": len(rows), "n_cf": m,
            "final_output_absorbed": _rate(sum(bool(r.get("final_output_absorbed")) for r in fo), len(fo)),
            "absorbed": _rate(absorbed, m), "silently_corrected": _rate(silent, m),
            "flagged": _rate(flagged, m), "other": _rate(other, m),
            "doubt_lex": _rate(sum(bool(r.get("doubt_lex")) for r in cf_rows), m),
            "no_output_claim": _rate(sum(bool(r.get("no_output_claim")) for r in rows), len(rows))}


def summarize(gold_cohort, b1_rows, b2_rows):
    out = {"created_at": now(), "model": "DeepSeek-R1-Distill-Qwen-7B",
           "cells": list(CELLS), "B1": {}, "B2": {}, "B2_rederive_taxonomy": {}}
    for cell in CELLS:
        out["B1"][cell] = cell_summary([r for r in b1_rows if r["cell"] == cell])
        out["B2"][cell] = cell_summary([r for r in b2_rows if r["cell"] == cell])
    for cell in CELLS:
        cf = [r for r in b2_rows if r["cell"] == cell and r.get("cf") and "rederive_bucket" in r]
        from collections import Counter
        strict = Counter(r["rederive_bucket"] for r in cf)
        loose = Counter(r["rederive_bucket_loose"] for r in cf)
        n = len(cf)
        out["B2_rederive_taxonomy"][cell] = {
            "n": n,
            "think_truncated": sum(1 for r in cf if r.get("think_truncated")),
            "strict_assign": {k: {"count": strict.get(k, 0), "rate": round(strict.get(k, 0) / n, 4) if n else None}
                              for k in ("rederive_and_correct", "rederive_and_absorb", "never_rederive")},
            "loose_token": {k: {"count": loose.get(k, 0), "rate": round(loose.get(k, 0) / n, 4) if n else None}
                            for k in ("rederive_and_correct", "rederive_and_absorb", "never_rederive")}}
    out["gold"] = gold_cohort
    write_json(os.path.join(OUT_DIR, "summary_arm_b.json"), out)
    print(json.dumps({k: out[k] for k in ("B1", "B2", "B2_rederive_taxonomy")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--b2-max-tokens", type=int, default=3000)
    ap.add_argument("--gold-oversample", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    progs, by_shape = programs_by_shape()
    llm = make_llm()
    # GOLD (resume if cohort exists)
    coh_path = os.path.join(OUT_DIR, "gold_cohort.json")
    if os.path.exists(coh_path):
        cj = json.load(open(coh_path))
        goldraw = {r["program_id"]: r["gold_full_text"] for r in read_jsonl(os.path.join(OUT_DIR, "gold_raw.jsonl")) if r["solved"]}
        cohort = goldraw
        print("[gold] resumed cohort n=%d" % len(cohort))
    else:
        cohort = stage_gold(llm, by_shape, args.gold_oversample)
        cj = json.load(open(coh_path))
    manifest = build_manifest(progs, by_shape, cohort)
    from collections import Counter
    print("[manifest] cells=%s" % json.dumps(dict(Counter(m["cell"] for m in manifest))))
    b1 = stage_b1(llm, progs, manifest, args.reps)
    b2 = stage_b2(llm, progs, manifest, args.reps, args.b2_max_tokens)
    summarize(cj, b1, b2)


if __name__ == "__main__":
    main()
